#!/usr/bin/env python3
"""
BM25 + L2R experiment (lexical features only).

Trains on queries.train.tsv + qrels.train.tsv, evaluates on queries.dev.tsv + qrels.dev.tsv.

Features:
- BM25 score
- Per-query BM25 features: bm25_norm, inv_rank, rank_pct
- Lexical: coverage, doc_len_log, phrase_exact, tri_overlap, bi_jacc
- Proximity (optional)

Usage:
    python experiments/bm25_l2r.py --train-queries 5000 --dev-queries 1000
    python experiments/bm25_l2r.py --help
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
    add_per_query_bm25_features,
    eval_metric,
)


LEX_COLS = ["coverage", "doc_len_log", "phrase_exact", "tri_overlap", "bi_jacc"]
PROX_COLS = [
    "prox_min_window",
    "prox_ordered_window",
    "prox_pair_avg",
    "prox_cluster_density",
]
PER_QUERY_COLS = ["bm25_norm", "inv_rank", "rank_pct"]


def build_train_pool(
    es,
    index: str,
    queries: pd.DataFrame,
    qid_to_rels: Dict[str, Set[str]],
    n_queries: int,
    candidate_k: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]]]:
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

    return pd.DataFrame(rows), qid_to_text, qid_to_tokens


def build_dev_pool(
    es,
    index: str,
    queries: pd.DataFrame,
    qid_to_rels: Dict[str, Set[str]],
    n_queries: int,
    candidate_k: int,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, List[str]], List[str]]:
    """Build evaluation pool (deterministic, no shuffle).

    Returns (df, qid_to_text, qid_to_tokens, sampled_qids).
    sampled_qids includes ALL sampled queries (even those with no BM25 hits).
    """
    qrows = list(queries[["qid", "query"]].itertuples(index=False, name=None))[
        :n_queries
    ]
    sampled_qids = [str(qid) for qid, _ in qrows]

    rows: List[dict] = []
    qid_to_text: Dict[str, str] = {}
    qid_to_tokens: Dict[str, List[str]] = {}

    for qid, qtext in qrows:
        hits = bm25_search(es, index, str(qtext), candidate_k)
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

    return pd.DataFrame(rows), qid_to_text, qid_to_tokens, sampled_qids


def add_lexical_features(
    df: pd.DataFrame, qid_to_tokens: Dict[str, List[str]], use_prox: bool = False
) -> List[str]:
    """Add lexical features to dataframe (in-place). Returns feature column names."""
    n_rows = len(df)
    n_lex = 5 + (4 if use_prox else 0)
    lex_arr = np.zeros((n_rows, n_lex), dtype=np.float64)

    for i, row in enumerate(df.itertuples()):
        q_tokens = qid_to_tokens[row.qid]
        d_tokens = tokenize(row.passage)
        feats = lexical_features(q_tokens, d_tokens, use_prox=use_prox)
        lex_arr[i, : len(feats)] = feats

    cols = LEX_COLS[:]
    df[cols] = lex_arr[:, :5]

    if use_prox:
        prox_cols = PROX_COLS[:4]
        for j, c in enumerate(prox_cols):
            df[c] = lex_arr[:, 5 + j]
        cols.extend(prox_cols)

    return cols


def train_and_eval(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    feat_cols: List[str],
    scale_cols: List[str],
    all_dev_qids: List[str],
    dev_qids_with_hits: List[str],
    dev_pids_per_qid: Dict[str, List[str]],
    qid_to_rels: Dict[str, Set[str]],
    eval_metric_name: str,
    eval_k: int,
    seed: int,
) -> Tuple[float, float, int, int]:
    """Train XGBRanker and evaluate on dev.

    Returns (bm25_score, l2r_score, n_queries, n_no_hits).
    - n_queries: total number of sampled dev queries
    - n_no_hits: queries where BM25 returned no candidates
    """
    # Sort by qid, then by bm25 descending (for group assignment)
    train_df = train_df.sort_values(["qid", "bm25"], ascending=[True, False])
    dev_df = dev_df.sort_values(["qid", "bm25"], ascending=[True, False])

    X_train = train_df[feat_cols].to_numpy(np.float64)
    y_train = train_df["label"].to_numpy(np.float32)
    groups_train = train_df.groupby("qid").size().tolist()

    X_dev = dev_df[feat_cols].to_numpy(np.float64)
    groups_dev = dev_df.groupby("qid").size().tolist()

    scaler = StandardScaler()
    X_train[:, scale_cols_idx] = scaler.fit_transform(X_train[:, scale_cols_idx])
    X_dev[:, scale_cols_idx] = scaler.transform(X_dev[:, scale_cols_idx])

    model = XGBRanker(
        objective="rank:pairwise",
        random_state=seed,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
    )
    model.fit(X_train, y_train, group=groups_train)

    dev_preds = model.predict(X_dev)

    # Build lookup for queries with hits
    dev_qid_to_idx = {qid: i for i, qid in enumerate(dev_qids_with_hits)}

    bm25_scores = []
    l2r_scores = []
    n_no_hits = 0

    offset = 0
    for qid in all_dev_qids:
        if qid not in dev_qid_to_idx:
            # BM25 returned no hits for this query
            n_no_hits += 1
            bm25_scores.append(0.0)
            l2r_scores.append(0.0)
            continue

        idx = dev_qid_to_idx[qid]
        g = groups_dev[idx]
        pids = dev_pids_per_qid[qid]

        bm25_raw = X_dev[offset : offset + g, 0]
        l2r_raw = dev_preds[offset : offset + g]

        bm25_order = np.argsort(-bm25_raw)
        l2r_order = np.argsort(-l2r_raw)

        rels = [1 if pid in qid_to_rels.get(qid, set()) else 0 for pid in pids]

        bm25_rels = [rels[j] for j in bm25_order[:eval_k]]
        l2r_rels = [rels[j] for j in l2r_order[:eval_k]]

        bm25_scores.append(eval_metric(bm25_rels, eval_metric_name, eval_k))
        l2r_scores.append(eval_metric(l2r_rels, eval_metric_name, eval_k))

        offset += g

    n_queries = len(bm25_scores)
    return float(np.mean(bm25_scores)), float(np.mean(l2r_scores)), n_queries, n_no_hits


# Global index for scale columns
scale_cols_idx = None


def main():
    global scale_cols_idx

    parser = argparse.ArgumentParser(
        description="BM25 + L2R experiment (lexical features)"
    )
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument(
        "--train-queries", type=int, default=5000, help="Number of training queries"
    )
    parser.add_argument(
        "--dev-queries", type=int, default=1000, help="Number of dev queries for eval"
    )
    parser.add_argument(
        "--candidates", type=int, default=100, help="BM25 candidates per query"
    )
    parser.add_argument("--prox", action="store_true", help="Use proximity features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--metric", default="mrr", choices=["mrr", "ndcg"], help="Evaluation metric"
    )
    parser.add_argument("--metric-k", type=int, default=10, help="Evaluation cutoff")
    parser.add_argument("--index", default="msmarco", help="Elasticsearch index")
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
    train_df, train_qid_to_text, train_qid_to_tokens = build_train_pool(
        es,
        args.index,
        train_queries,
        train_qid_to_rels,
        args.train_queries,
        args.candidates,
        args.seed,
    )
    print(f"Training rows: {len(train_df)}")

    print(f"Building dev pool (n={args.dev_queries}, k={args.candidates})...")
    dev_df, dev_qid_to_text, dev_qid_to_tokens, sampled_dev_qids = build_dev_pool(
        es,
        args.index,
        dev_queries,
        dev_qid_to_rels,
        args.dev_queries,
        args.candidates,
    )
    print(f"Dev rows: {len(dev_df)}")
    print(f"Sampled dev queries: {len(sampled_dev_qids)}")

    # Filter training queries with no positives (need at least 1 relevant doc to train)
    before_train = len(train_df)
    train_df = train_df[train_df.groupby("qid")["label"].transform("sum") > 0].copy()
    print(
        f"Train rows after filtering: {len(train_df)} (removed {before_train - len(train_df)})"
    )

    # For dev: keep queries with positives for feature extraction, but track all sampled qids
    before_dev = len(dev_df)
    dev_df = dev_df[dev_df.groupby("qid")["label"].transform("sum") > 0].copy()
    print(
        f"Dev rows after filtering: {len(dev_df)} (removed {before_dev - len(dev_df)})"
    )

    train_qids = train_df["qid"].unique().tolist()
    dev_qids_with_hits = dev_df["qid"].unique().tolist()
    n_no_hits = len(sampled_dev_qids) - len(dev_qids_with_hits)
    print(f"Train queries: {len(train_qids)}")
    print(
        f"Dev queries with BM25 hits: {len(dev_qids_with_hits)}, no hits: {n_no_hits}"
    )

    # Combine token dicts
    all_qid_to_tokens = {**train_qid_to_tokens, **dev_qid_to_tokens}

    # Add lexical features
    print("Adding lexical features...")
    lex_cols = add_lexical_features(train_df, all_qid_to_tokens, use_prox=args.prox)
    add_lexical_features(dev_df, all_qid_to_tokens, use_prox=args.prox)

    # Add per-query BM25 features
    print("Adding per-query BM25 features...")
    add_per_query_bm25_features(train_df)
    add_per_query_bm25_features(dev_df)

    # Define feature columns: BM25, per-query, lexical
    feat_cols = ["bm25"] + PER_QUERY_COLS + lex_cols
    scale_cols = [c for c in feat_cols if c != "bm25"]
    scale_cols_idx = [feat_cols.index(c) for c in scale_cols]

    print(f"Features: {feat_cols}")

    # Build dev pids mapping (only for queries with hits)
    dev_pids_per_qid = {
        qid: dev_df[dev_df["qid"] == qid]["pid"].tolist() for qid in dev_qids_with_hits
    }

    print("Training XGBRanker...")
    bm25_score, l2r_score, n_queries, n_empty = train_and_eval(
        train_df,
        dev_df,
        feat_cols,
        scale_cols,
        sampled_dev_qids,
        dev_qids_with_hits,
        dev_pids_per_qid,
        dev_qid_to_rels,
        args.metric,
        args.metric_k,
        args.seed,
    )

    print(f"\n{'=' * 50}")
    print(f"Results ({args.metric.upper()}@{args.metric_k}) on DEV:")
    print(f"  Total queries: {n_queries}")
    print(f"  Queries with BM25 hits: {n_queries - n_empty}")
    print(f"  Queries with no hits: {n_empty}")
    print(f"  BM25: {bm25_score:.4f}")
    print(f"  L2R:  {l2r_score:.4f}")
    print(f"  Delta: {l2r_score - bm25_score:+.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
