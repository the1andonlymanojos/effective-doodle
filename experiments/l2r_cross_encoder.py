#!/usr/bin/env python3
"""
BM25 + L2R + Cross-Encoder experiment.

Three-stage ranking pipeline:
1. BM25 retrieval (candidate generation)
2. L2R reranking (lexical + embedding features)
3. Cross-encoder reranking (top-k from L2R)

Trains on queries.train.tsv + qrels.train.tsv, evaluates on queries.dev.tsv + qrels.dev.tsv.

Usage:
    python experiments/l2r_cross_encoder.py --train-queries 5000 --dev-queries 1000
    python experiments/l2r_cross_encoder.py --help
"""

import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.shared import (
    get_elasticsearch_client,
    load_queries,
    load_qrels,
    bm25_search,
    tokenize,
    lexical_features,
    eval_metric,
    fetch_passage_embeddings,
    l2_normalize_rows,
)


def build_train_pool(
    es,
    index: str,
    queries: pd.DataFrame,
    qid_to_rels: Dict[str, Set[str]],
    n_queries: int,
    candidate_k: int,
    seed: int,
) -> Tuple[List[dict], Dict[str, str], Dict[str, List[str]]]:
    """Build BM25 candidate pool for training (shuffled)."""
    rng = random.Random(seed)
    qrows = list(queries[["qid", "query"]].itertuples(index=False, name=None))
    rng.shuffle(qrows)
    qrows = qrows[:n_queries]

    rows: List[dict] = []
    qid_to_text: Dict[str, str] = {}
    qid_to_tokens: Dict[str, List[str]] = {}

    for qid, qtext in qrows:
        hits = bm25_search(es, index, str(qtext), candidate_k)
        if not hits:
            continue

        qid_s = str(qid)
        qid_to_text[qid_s] = str(qtext)
        qid_to_tokens[qid_s] = tokenize(str(qtext))

        for pid, passage, score in hits:
            rel = 1 if str(pid) in qid_to_rels.get(qid_s, set()) else 0
            rows.append(
                {
                    "qid": qid_s,
                    "pid": str(pid),
                    "passage": passage,
                    "bm25": float(score),
                    "label": rel,
                }
            )

    return rows, qid_to_text, qid_to_tokens


def build_dev_pool(
    es,
    index: str,
    queries: pd.DataFrame,
    qid_to_rels: Dict[str, Set[str]],
    n_queries: int,
    candidate_k: int,
) -> Tuple[List[dict], Dict[str, str], Dict[str, List[str]]]:
    """Build evaluation pool (deterministic, no shuffle)."""
    qrows = list(queries[["qid", "query"]].itertuples(index=False, name=None))[
        :n_queries
    ]

    rows: List[dict] = []
    qid_to_text: Dict[str, str] = {}
    qid_to_tokens: Dict[str, List[str]] = {}

    for qid, qtext in qrows:
        hits = bm25_search(es, index, str(qtext), candidate_k)
        if not hits:
            continue

        qid_s = str(qid)
        qid_to_text[qid_s] = str(qtext)
        qid_to_tokens[qid_s] = tokenize(str(qtext))

        for pid, passage, score in hits:
            rel = 1 if str(pid) in qid_to_rels.get(qid_s, set()) else 0
            rows.append(
                {
                    "qid": qid_s,
                    "pid": str(pid),
                    "passage": passage,
                    "bm25": float(score),
                    "label": rel,
                }
            )

    return rows, qid_to_text, qid_to_tokens


def featurize_lexical(
    rows: List[dict],
    qid_to_tokens: Dict[str, List[str]],
    use_prox: bool = False,
) -> np.ndarray:
    """Build lexical feature matrix."""
    n_rows = len(rows)
    n_lex = 6 + (1 if use_prox else 0)
    X = np.zeros((n_rows, n_lex), dtype=np.float64)

    for i, row in enumerate(rows):
        q_tokens = qid_to_tokens[row["qid"]]
        d_tokens = tokenize(row["passage"])
        feats = lexical_features(q_tokens, d_tokens, use_prox=use_prox)
        X[i, 0] = row["bm25"]
        X[i, 1:] = feats

    return X


def add_embedding_features(
    df: pd.DataFrame,
    qid_to_text: Dict[str, str],
    es,
    index: str,
    emb_field: str,
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Add embedding features. Returns (n_rows, 3) array."""
    # Get unique queries and passages
    uqids = df["qid"].unique().tolist()
    upids = df["pid"].unique().tolist()

    print(f"Fetching passage embeddings...")
    pid_to_vec, missing = fetch_passage_embeddings(
        es, index, upids, emb_field, batch_size=batch_size
    )
    print(f"  Passages: {len(pid_to_vec)} found, {missing}/{len(upids)} missing")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    # Map qid/pid to indices
    qid_to_idx = {q: i for i, q in enumerate(uqids)}
    pid_to_idx = {p: i for i, p in enumerate(upids)}

    # Encode UNIQUE queries only
    q_texts = [qid_to_text[q] for q in uqids]
    print(f"  Encoding {len(uqids)} unique queries...")
    q_emb_unique = model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = q_emb_unique.shape[1]

    # Build passage embedding matrix
    p_emb = np.zeros((len(upids), dim), dtype=np.float64)
    for i, pid in enumerate(upids):
        if pid in pid_to_vec:
            v = pid_to_vec[pid]
            if v.shape[0] == dim:
                p_emb[i] = v

    p_emb = l2_normalize_rows(p_emb)

    # Map to all rows
    qi = np.array([qid_to_idx[q] for q in df["qid"]])
    pi = np.array([pid_to_idx[p] for p in df["pid"]])

    q_emb = q_emb_unique[qi]
    p_emb_rows = p_emb[pi]

    emb_cos = np.sum(q_emb * p_emb_rows, axis=1)
    emb_l2 = np.linalg.norm(q_emb - p_emb_rows, axis=1)
    emb_dot = np.sum(q_emb * p_emb_rows, axis=1)

    return np.column_stack([emb_cos, emb_l2, emb_dot])


def cross_encoder_rerank(
    query_text: str,
    passages: List[str],
    pids: List[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> List[Tuple[str, float]]:
    """Rerank passages with cross-encoder. Returns list of (pid, score)."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)
    pairs = [[query_text, p] for p in passages]
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)

    ranked = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)
    return ranked


def train_l2r(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: List[int],
    seed: int,
) -> Tuple[StandardScaler, XGBRanker]:
    """Train L2R model. Returns (scaler, model)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = XGBRanker(
        objective="rank:pairwise",
        random_state=seed,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
    )
    model.fit(X_train_scaled, y_train, group=groups_train)

    return scaler, model


def evaluate_three_stage(
    dev_rows: List[dict],
    dev_df: pd.DataFrame,
    dev_qids: List[str],
    dev_pids_per_qid: Dict[str, List[str]],
    X_dev: np.ndarray,
    scaler: StandardScaler,
    l2r_model: XGBRanker,
    qid_to_text: Dict[str, str],
    qid_to_rels: Dict[str, Set[str]],
    cross_encoder_model: str,
    cross_encoder_k: int,
    batch_size: int,
    device: str,
    eval_metric_name: str,
    eval_k: int,
) -> Tuple[float, float, float]:
    """Evaluate BM25 / L2R / Cross-Encoder. Returns (bm25, l2r, ce) scores."""
    X_dev_scaled = scaler.transform(X_dev)
    l2r_preds = l2r_model.predict(X_dev_scaled)

    groups_dev = dev_df.groupby("qid").size().tolist()

    offset = 0
    bm25_scores = []
    l2r_scores = []
    ce_scores = []

    ce_available = cross_encoder_model is not None

    for i, qid in enumerate(dev_qids):
        g = groups_dev[i]
        pids = dev_pids_per_qid[qid]
        query_text = qid_to_text[qid]

        bm25_raw = X_dev[offset : offset + g, 0]
        l2r_raw = l2r_preds[offset : offset + g]

        bm25_order = np.argsort(-bm25_raw)
        l2r_order = np.argsort(-l2r_raw)

        rels = [1 if pid in qid_to_rels.get(qid, set()) else 0 for pid in pids]

        bm25_rels = [rels[j] for j in bm25_order[:eval_k]]
        l2r_rels = [rels[j] for j in l2r_order[:eval_k]]

        bm25_scores.append(eval_metric(bm25_rels, eval_metric_name, eval_k))
        l2r_scores.append(eval_metric(l2r_rels, eval_metric_name, eval_k))

        if ce_available:
            top_k_idx = l2r_order[:cross_encoder_k]
            top_k_pids = [pids[j] for j in top_k_idx]
            top_k_passages = [dev_rows[offset + j]["passage"] for j in top_k_idx]

            ce_ranked = cross_encoder_rerank(
                query_text,
                top_k_passages,
                top_k_pids,
                cross_encoder_model,
                batch_size,
                device,
            )

            ce_order = [top_k_pids.index(pid) for pid, _ in ce_ranked[:eval_k]]
            ce_rels = [rels[top_k_idx[j]] for j in ce_order]
            ce_scores.append(eval_metric(ce_rels, eval_metric_name, eval_k))

        offset += g

    if ce_available:
        return (
            float(np.mean(bm25_scores)),
            float(np.mean(l2r_scores)),
            float(np.mean(ce_scores)),
        )
    else:
        return float(np.mean(bm25_scores)), float(np.mean(l2r_scores)), 0.0


def main():
    parser = argparse.ArgumentParser(
        description="BM25 + L2R + Cross-Encoder experiment"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--train-queries", type=int, default=5000, help="Number of training queries"
    )
    parser.add_argument(
        "--dev-queries", type=int, default=1000, help="Number of dev queries"
    )
    parser.add_argument(
        "--candidates", type=int, default=100, help="BM25 candidates per query"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--metric", default="mrr", choices=["mrr", "ndcg"], help="Evaluation metric"
    )
    parser.add_argument("--metric-k", type=int, default=10, help="Evaluation cutoff")
    parser.add_argument("--index", default="msmarco", help="Elasticsearch index")
    parser.add_argument("--emb-field", default="embedding", help="ES embedding field")
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model",
    )
    parser.add_argument(
        "--ce-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        help="Cross-encoder model",
    )
    parser.add_argument(
        "--ce-k", type=int, default=100, help="Cross-encoder top-k from L2R"
    )
    parser.add_argument("--no-ce", action="store_true", help="Disable cross-encoder")
    parser.add_argument("--prox", action="store_true", help="Use proximity features")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    es = get_elasticsearch_client()
    print(f"ES ping: {es.ping()}")

    train_queries_path = os.path.join(args.data_dir, "queries.train.tsv")
    train_qrels_path = os.path.join(args.data_dir, "qrels.train.tsv")
    dev_queries_path = os.path.join(args.data_dir, "queries.dev.tsv")
    dev_qrels_path = os.path.join(args.data_dir, "qrels.dev.tsv")

    print("Loading training data...")
    train_queries = load_queries(train_queries_path)
    _, train_qid_to_rels = load_qrels(train_qrels_path)

    print("Loading dev data...")
    dev_queries = load_queries(dev_queries_path)
    _, dev_qid_to_rels = load_qrels(dev_qrels_path)

    train_valid_qids = set(train_qid_to_rels.keys())
    train_queries = train_queries[train_queries["qid"].isin(train_valid_qids)].copy()
    print(f"Train queries with qrels: {len(train_queries)}")

    dev_valid_qids = set(dev_qid_to_rels.keys())
    dev_queries = dev_queries[dev_queries["qid"].isin(dev_valid_qids)].copy()
    print(f"Dev queries with qrels: {len(dev_queries)}")

    print(f"Building training pool (n={args.train_queries}, k={args.candidates})...")
    train_rows, train_qid_to_text, train_qid_to_tokens = build_train_pool(
        es,
        args.index,
        train_queries,
        train_qid_to_rels,
        args.train_queries,
        args.candidates,
        args.seed,
    )
    print(f"Training rows: {len(train_rows)}")

    print(f"Building dev pool (n={args.dev_queries}, k={args.candidates})...")
    dev_rows, dev_qid_to_text, dev_qid_to_tokens = build_dev_pool(
        es,
        args.index,
        dev_queries,
        dev_qid_to_rels,
        args.dev_queries,
        args.candidates,
    )
    print(f"Dev rows: {len(dev_rows)}")

    train_df = pd.DataFrame(train_rows)
    dev_df = pd.DataFrame(dev_rows)

    # Filter out queries with no positive labels (no relevant doc in BM25 pool)
    before_train = len(train_df)
    train_df = train_df[train_df.groupby("qid")["label"].transform("sum") > 0].copy()
    print(
        f"Train rows after filtering queries with no positives: {len(train_df)} (removed {before_train - len(train_df)})"
    )

    before_dev = len(dev_df)
    dev_df = dev_df[dev_df.groupby("qid")["label"].transform("sum") > 0].copy()
    print(
        f"Dev rows after filtering queries with no positives: {len(dev_df)} (removed {before_dev - len(dev_df)})"
    )

    # Update rows to match filtered dataframes
    train_rows = train_df.to_dict("records")
    dev_rows = dev_df.to_dict("records")

    train_qids = train_df["qid"].unique().tolist()
    dev_qids = dev_df["qid"].unique().tolist()

    print(f"Train queries: {len(train_qids)}, Dev queries: {len(dev_qids)}")

    all_qid_to_tokens = {**train_qid_to_tokens, **dev_qid_to_tokens}

    print("Featurizing (lexical)...")
    X_train_lex = featurize_lexical(train_rows, all_qid_to_tokens, use_prox=args.prox)
    X_dev_lex = featurize_lexical(dev_rows, all_qid_to_tokens, use_prox=args.prox)

    print("Adding embedding features (train)...")
    X_train_emb = add_embedding_features(
        train_df,
        train_qid_to_text,
        es,
        args.index,
        args.emb_field,
        args.model,
        args.batch_size,
        args.device,
    )
    print("Adding embedding features (dev)...")
    X_dev_emb = add_embedding_features(
        dev_df,
        dev_qid_to_text,
        es,
        args.index,
        args.emb_field,
        args.model,
        args.batch_size,
        args.device,
    )

    X_train = np.hstack([X_train_lex, X_train_emb])
    X_dev = np.hstack([X_dev_lex, X_dev_emb])

    y_train = train_df["label"].values

    groups_train = train_df.groupby("qid").size().tolist()
    dev_pids_per_qid = {
        qid: dev_df[dev_df["qid"] == qid]["pid"].tolist() for qid in dev_qids
    }

    print("Training L2R model...")
    scaler, l2r_model = train_l2r(X_train, y_train, groups_train, args.seed)

    cross_encoder_model = None if args.no_ce else args.ce_model

    print(f"Evaluating (CE {'disabled' if args.no_ce else 'enabled'})...")
    bm25_score, l2r_score, ce_score = evaluate_three_stage(
        dev_rows,
        dev_df,
        dev_qids,
        dev_pids_per_qid,
        X_dev,
        scaler,
        l2r_model,
        dev_qid_to_text,
        dev_qid_to_rels,
        cross_encoder_model,
        args.ce_k,
        args.batch_size,
        args.device,
        args.metric,
        args.metric_k,
    )

    print(f"\n{'=' * 50}")
    print(f"Results ({args.metric.upper()}@{args.metric_k}) on DEV:")
    print(f"  BM25:   {bm25_score:.4f}")
    print(f"  L2R:    {l2r_score:.4f} (Δ {l2r_score - bm25_score:+.4f})")
    if not args.no_ce:
        print(f"  CE:     {ce_score:.4f} (Δ {ce_score - l2r_score:+.4f})")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
