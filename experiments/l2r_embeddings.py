#!/usr/bin/env python3
"""
BM25 + L2R experiment with embedding features.

Trains on queries.train.tsv + qrels.train.tsv, evaluates on queries.dev.tsv + qrels.dev.tsv.

Usage:
    python experiments/l2r_embeddings.py --train-queries 5000 --dev-queries 1000
    python experiments/l2r_embeddings.py --help
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
) -> Tuple[np.ndarray, List[str]]:
    """Build lexical feature matrix."""
    n_rows = len(rows)
    X = np.zeros((n_rows, 6), dtype=np.float64)

    for i, row in enumerate(rows):
        q_tokens = qid_to_tokens[row["qid"]]
        d_tokens = tokenize(row["passage"])
        feats = lexical_features(q_tokens, d_tokens, use_prox=False)
        X[i, 0] = row["bm25"]
        X[i, 1:] = feats

    feat_cols = [
        "bm25",
        "coverage",
        "doc_len_log",
        "phrase_exact",
        "tri_overlap",
        "bi_jacc",
    ]
    return X, feat_cols


def add_embedding_features(
    df: pd.DataFrame,
    qid_to_text: Dict[str, str],
    es,
    index: str,
    emb_field: str,
    model_name: str,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, List[str]]:
    """Add embedding features. Returns (emb_features, emb_cols)."""
    # Get unique queries and passages
    uqids = df["qid"].unique().tolist()
    upids = df["pid"].unique().tolist()

    print(f"Fetching passage embeddings from ES (field: {emb_field})...")
    pid_to_vec, missing = fetch_passage_embeddings(
        es, index, upids, emb_field, batch_size=batch_size
    )
    print(f"  Passages: {len(pid_to_vec)} found, {missing}/{len(upids)} missing")

    # Map qid/pid to row indices
    qid_to_idx = {q: i for i, q in enumerate(uqids)}
    pid_to_idx = {p: i for i, p in enumerate(upids)}

    print(f"Computing query embeddings with {model_name}...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

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

    emb_features = np.column_stack([emb_cos, emb_l2, emb_dot])
    emb_cols = ["emb_cos", "emb_l2", "emb_dot"]

    return emb_features, emb_cols


def train_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: List[int],
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    groups_dev: List[int],
    qid_to_rels: Dict[str, Set[str]],
    dev_qids: List[str],
    dev_pids_per_qid: Dict[str, List[str]],
    eval_metric_name: str,
    eval_k: int,
    seed: int,
) -> Tuple[float, float, float]:
    """Train XGBRanker and evaluate. Returns (bm25_score, cosine_score, l2r_score)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)

    model = XGBRanker(
        objective="rank:pairwise",
        random_state=seed,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
    )
    model.fit(X_train_scaled, y_train, group=groups_train)

    dev_preds = model.predict(X_dev_scaled)

    emb_cos_idx = 6

    offset = 0
    bm25_scores = []
    cosine_scores = []
    l2r_scores = []

    for i, qid in enumerate(dev_qids):
        g = groups_dev[i]
        pids = dev_pids_per_qid[qid]

        bm25_scores_raw = X_dev[offset : offset + g, 0]
        cosine_scores_raw = X_dev[offset : offset + g, emb_cos_idx]
        l2r_scores_raw = dev_preds[offset : offset + g]

        bm25_order = np.argsort(-bm25_scores_raw)
        cosine_order = np.argsort(-cosine_scores_raw)
        l2r_order = np.argsort(-l2r_scores_raw)

        rels = [1 if pid in qid_to_rels.get(qid, set()) else 0 for pid in pids]

        bm25_rels = [rels[j] for j in bm25_order[:eval_k]]
        cosine_rels = [rels[j] for j in cosine_order[:eval_k]]
        l2r_rels = [rels[j] for j in l2r_order[:eval_k]]

        bm25_scores.append(eval_metric(bm25_rels, eval_metric_name, eval_k))
        cosine_scores.append(eval_metric(cosine_rels, eval_metric_name, eval_k))
        l2r_scores.append(eval_metric(l2r_rels, eval_metric_name, eval_k))

        offset += g

    return (
        float(np.mean(bm25_scores)),
        float(np.mean(cosine_scores)),
        float(np.mean(l2r_scores)),
    )


def main():
    parser = argparse.ArgumentParser(description="BM25 + L2R with embeddings")
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
    X_train_lex, feat_cols = featurize_lexical(train_rows, all_qid_to_tokens)
    X_dev_lex, _ = featurize_lexical(dev_rows, all_qid_to_tokens)

    print("Adding embedding features (train)...")
    X_train_emb, emb_cols = add_embedding_features(
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
    X_dev_emb, _ = add_embedding_features(
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

    all_cols = feat_cols + emb_cols
    print(f"Features: {all_cols}")

    y_train = train_df["label"].values
    y_dev = dev_df["label"].values

    groups_train = train_df.groupby("qid").size().tolist()
    groups_dev = dev_df.groupby("qid").size().tolist()

    dev_pids_per_qid = {
        qid: dev_df[dev_df["qid"] == qid]["pid"].tolist() for qid in dev_qids
    }

    print("Training XGBRanker...")

    bm25_score, cosine_score, l2r_score = train_and_eval(
        X_train,
        y_train,
        groups_train,
        X_dev,
        y_dev,
        groups_dev,
        dev_qid_to_rels,
        dev_qids,
        dev_pids_per_qid,
        args.metric,
        args.metric_k,
        args.seed,
    )

    print(f"\n{'=' * 50}")
    print(f"Results ({args.metric.upper()}@{args.metric_k}) on DEV:")
    print(f"  BM25:     {bm25_score:.4f}")
    print(f"  Cosine:   {cosine_score:.4f}")
    print(f"  L2R:      {l2r_score:.4f}")
    print(f"  L2R-Δ:    {l2r_score - bm25_score:+.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
