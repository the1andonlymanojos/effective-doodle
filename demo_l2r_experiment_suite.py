#!/usr/bin/env python3
"""
L2R experiment suite: baseline + BM25 variants, IDF/TF, proximity, coverage tiers,
dense extras, per-query normalization, interactions, optional negative sampling,
K sweep, and rank objective aligned with eval metric (MRR → rank:pairwise, NDCG → rank:ndcg).

Stages (01→09): baseline → +log/norm BM25 → +IDF overlap & TF → +proximity → +coverage tiers
→ +dense (cos, L2, dot) → +cos² & BM25×cos → +per-query max-norm (BM25/cos/log)
→ +explicit interactions (BM25×cov, BM25×overlap, cos×cov). Then 10: negative sampling.

Builds one BM25 pool (max-k per query), encodes queries + loads passage vectors once, then sweeps
K_eval and feature stacks so embedding cost is not repeated per experiment.

Writes progress to a log file with flush after each line — use: tail -f your_log.txt

Usage:
  python demo_l2r_experiment_suite.py
  python demo_l2r_experiment_suite.py --out runs/my_run.txt --queries 2000 --max-k 500
  python demo_l2r_experiment_suite.py --quick --out runs/smoke.txt
"""
from __future__ import annotations

import argparse
import math
import os
import random
import re
import sys
from collections import defaultdict
import heapq
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
def load_dotenv_like(path: str) -> Dict[str, str]:
    env = dict(os.environ)
    var_ref = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def expand(value: str) -> str:
        return var_ref.sub(lambda m: env.get(m.group(1), ""), value)

    with open(path, "r") as f:
        for line in f:
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.strip().split("=", 1)
            env[k.strip()] = expand(v.strip().strip('"'))
    return env


def tokenize(x) -> List[str]:
    return str(x).lower().split()


def ndcg(rels: List[int], k: int = 10) -> float:
    rels = rels[:k]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(rels: List[int], k: int = 10) -> float:
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def eval_metric(rels: List[int], metric: str, k: int) -> float:
    if metric == "ndcg":
        return ndcg(rels, k=k)
    if metric == "mrr":
        return mrr_at_k(rels, k=k)
    raise ValueError(metric)


def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def fetch_passage_embeddings_from_es(
    client: Elasticsearch,
    index: str,
    pids: List[str],
    field: str,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], int]:
    pid_to_vec: Dict[str, np.ndarray] = {}
    missing = 0
    for i in range(0, len(pids), batch_size):
        batch = pids[i : i + batch_size]
        resp = client.mget(index=index, ids=batch, source_includes=[field])
        for pid, doc in zip(batch, resp["docs"]):
            if not doc.get("found"):
                missing += 1
                continue
            src = doc.get("_source") or {}
            raw = src.get(field)
            if raw is None:
                missing += 1
                continue
            pid_to_vec[pid] = np.asarray(raw, dtype=np.float64).ravel()
    return pid_to_vec, missing


def compute_embedding_block(
    df: pd.DataFrame,
    qid_to_query_text: Dict[str, str],
    model_name: str,
    batch_size: int,
    seed: int,
    es_client: Elasticsearch,
    passage_index: str,
    passage_embedding_field: str,
    mget_batch_size: int,
) -> Tuple[np.ndarray, List[str]]:
    from sentence_transformers import SentenceTransformer

    np.random.seed(seed)
    model = SentenceTransformer(model_name, device="cpu")

    uqids = df["qid"].unique().tolist()
    q_texts = [qid_to_query_text[q] for q in uqids]
    qid_to_row = {q: i for i, q in enumerate(uqids)}

    pid_first = df.drop_duplicates(subset=["pid"], keep="first")
    upids = pid_first["pid"].tolist()
    pid_to_row = {p: i for i, p in enumerate(upids)}

    q_emb = model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    pid_to_vec, n_missing = fetch_passage_embeddings_from_es(
        es_client,
        passage_index,
        upids,
        passage_embedding_field,
        mget_batch_size,
    )
    if n_missing:
        pass  # logged at suite level if needed

    dim = None
    for p in upids:
        v = pid_to_vec.get(p)
        if v is not None:
            dim = int(v.shape[0])
            break
    if dim is None:
        raise RuntimeError(
            f"No passage embeddings in {passage_index!r} field {passage_embedding_field!r}"
        )

    p_emb = np.zeros((len(upids), dim), dtype=np.float64)
    for j, p in enumerate(upids):
        v = pid_to_vec.get(p)
        if v is not None and v.shape[0] == dim:
            p_emb[j] = v
        elif v is not None:
            raise ValueError(f"Inconsistent dim for pid={p!r}")

    p_emb = _l2_normalize_rows(p_emb)

    qi = np.array([qid_to_row[q] for q in df["qid"].to_numpy()], dtype=np.int64)
    pi = np.array([pid_to_row[p] for p in df["pid"].to_numpy()], dtype=np.int64)
    qv = q_emb[qi]
    pv = p_emb[pi]
    emb_cos = np.sum(qv * pv, axis=1)
    emb_l2 = np.linalg.norm(qv - pv, axis=1)
    emb_dot = np.sum(qv * pv, axis=1)  # same as cos when normalized; kept for spec
    names = ["emb_cos", "emb_l2", "emb_dot"]
    return np.column_stack([emb_cos, emb_l2, emb_dot]).astype(np.float64), names


# ---------------------------------------------------------------------------
# IDF from candidate pool (per dataset build)
# ---------------------------------------------------------------------------
def build_idf_from_passages(passages: Sequence[str]) -> Dict[str, float]:
    N = max(1, len(passages))
    df_t: Dict[str, int] = defaultdict(int)
    for text in passages:
        seen = set(tokenize(text))
        for t in seen:
            df_t[t] += 1
    out: Dict[str, float] = {}
    for t, dfi in df_t.items():
        out[t] = math.log((N + 1.0) / (1.0 + dfi)) + 1.0
    return out


# ---------------------------------------------------------------------------
# Proximity (distinct query terms in query order)
# ---------------------------------------------------------------------------
def distinct_query_terms(q_tokens: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for t in q_tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def term_positions(doc_tokens: List[str]) -> Dict[str, List[int]]:
    pos: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(doc_tokens):
        pos[t].append(i)
    return pos


def min_window_covering_all_terms(
    q_distinct: List[str], pos_map: Dict[str, List[int]]
) -> Optional[int]:
    """Smallest inclusive window length covering one occurrence per distinct term (smallest range, k lists)."""
    lists: List[List[int]] = []
    for t in q_distinct:
        if t not in pos_map or not pos_map[t]:
            return None
        lists.append(sorted(pos_map[t]))
    if not lists:
        return None
    pq: List[Tuple[int, int, int]] = [(lst[0], i, 0) for i, lst in enumerate(lists)]
    heapq.heapify(pq)
    max_val = max(lst[0] for lst in lists)
    best = max_val - pq[0][0] + 1
    while True:
        _min_val, row, idx = heapq.heappop(pq)
        if idx + 1 >= len(lists[row]):
            return best
        nxt = lists[row][idx + 1]
        heapq.heappush(pq, (nxt, row, idx + 1))
        max_val = max(max_val, nxt)
        cur = max_val - pq[0][0] + 1
        if cur < best:
            best = cur


def first_positions(q_distinct: List[str], pos_map: Dict[str, List[int]]) -> Optional[List[int]]:
    fp: List[int] = []
    for t in q_distinct:
        if t not in pos_map or not pos_map[t]:
            return None
        fp.append(pos_map[t][0])
    return fp


def min_pairwise_distance(positions: List[int]) -> float:
    if len(positions) < 2:
        return 0.0
    m = float("inf")
    for a in range(len(positions)):
        for b in range(a + 1, len(positions)):
            m = min(m, abs(positions[a] - positions[b]))
    return float(m)


def ordered_subsequence_match(q_tokens: List[str], d_tokens: List[str]) -> int:
    """1 if query tokens appear in order as a subsequence in doc (not necessarily adjacent)."""
    if not q_tokens:
        return 1
    j = 0
    for t in d_tokens:
        if j < len(q_tokens) and t == q_tokens[j]:
            j += 1
    return 1 if j == len(q_tokens) else 0


# ---------------------------------------------------------------------------
# Feature bundle → column list and builder
# ---------------------------------------------------------------------------
@dataclass
class FeatureSpec:
    name: str
    bm25_variants: bool = False
    idf_tf: bool = False
    proximity: bool = False
    coverage_tiers: bool = False
    dense: bool = False
    dense_extras: bool = False  # cos^2, bm25*cos (needs dense)
    interactions: bool = False


def build_feature_matrix(
    df: pd.DataFrame,
    qid_to_tokens: Dict[str, List[str]],
    idf_map: Dict[str, float],
    spec: FeatureSpec,
    bm25_col: str = "bm25",
) -> Tuple[pd.DataFrame, List[str]]:
    """Returns df with new columns and list of feature column names (in order)."""
    rows: List[List[float]] = []
    cols: List[str] = []

    base = ["overlap", "coverage", "doc_len", "phrase"]
    if spec.bm25_variants:
        base = ["bm25_raw", "bm25_log", "bm25_norm_q"] + base
    else:
        base = [bm25_col, "overlap", "coverage", "doc_len", "phrase"] + []

    # Reset base construction cleanly
    if spec.bm25_variants:
        ordered = ["bm25_raw", "bm25_log", "bm25_norm_q", "overlap", "coverage", "doc_len", "phrase"]
    else:
        ordered = [bm25_col, "overlap", "coverage", "doc_len", "phrase"]

    if spec.idf_tf:
        ordered += ["idf_overlap", "tf_max", "tf_sum", "tf_mean"]

    if spec.proximity:
        ordered += ["prox_min_span", "prox_min_pair", "prox_ordered"]

    if spec.coverage_tiers:
        ordered += ["cov_full", "cov_gt_0_5", "cov_gt_0_8"]

    if spec.dense:
        ordered += ["emb_cos", "emb_l2", "emb_dot"]

    if spec.dense_extras:
        ordered += ["emb_cos_sq", "bm25_x_cos"]

    if spec.interactions:
        ordered += ["bm25_x_cov", "bm25_x_overlap", "cos_x_cov"]

    # Precompute per-query max bm25 for norm
    gbm25 = df.groupby("qid")[bm25_col].transform("max").replace(0, np.nan)
    gbm25 = gbm25.fillna(1.0)
    df = df.copy()
    df["_max_bm25_q"] = gbm25

    for _, r in df.iterrows():
        qid = r["qid"]
        passage = r["passage"]
        bm = float(r[bm25_col])
        q_tokens = qid_to_tokens[qid]
        d_tokens = tokenize(passage)
        d_set = set(d_tokens)
        pos_map = term_positions(d_tokens)
        q_dist = distinct_query_terms(q_tokens)

        overlap = sum(1 for t in q_tokens if t in d_set)
        cov = overlap / max(1, len(q_tokens))
        doc_len = float(len(d_tokens))
        q_phrase = " ".join(q_tokens)
        d_text = " ".join(d_tokens)
        phrase = float(int(q_phrase in d_text))

        idf_overlap = sum(idf_map.get(t, 0.0) for t in q_tokens if t in d_set)
        tf_counts = [d_tokens.count(t) for t in q_tokens]
        tf_max = float(max(tf_counts)) if tf_counts else 0.0
        tf_sum = float(sum(tf_counts))
        tf_mean = tf_sum / max(1, len(q_tokens))

        ms = min_window_covering_all_terms(q_dist, pos_map)
        if ms is None:
            prox_span = 1e6
        else:
            prox_span = float(ms)
        fp = first_positions(q_dist, pos_map)
        prox_pair = min_pairwise_distance(fp) if fp is not None else 1e6
        prox_ord = float(ordered_subsequence_match(q_tokens, d_tokens))

        cov_full = float(int(abs(cov - 1.0) < 1e-12))
        cov_gt_05 = float(cov > 0.5)
        cov_gt_08 = float(cov > 0.8)

        bm_log = math.log1p(max(0.0, bm))
        bm_nq = bm / max(1e-12, float(r["_max_bm25_q"]))

        vec: Dict[str, float] = {}
        if spec.bm25_variants:
            vec["bm25_raw"] = bm
            vec["bm25_log"] = bm_log
            vec["bm25_norm_q"] = bm_nq
        else:
            vec[bm25_col] = bm

        vec["overlap"] = float(overlap)
        vec["coverage"] = float(cov)
        vec["doc_len"] = doc_len
        vec["phrase"] = phrase

        if spec.idf_tf:
            vec["idf_overlap"] = idf_overlap
            vec["tf_max"] = tf_max
            vec["tf_sum"] = tf_sum
            vec["tf_mean"] = tf_mean

        if spec.proximity:
            vec["prox_min_span"] = math.log1p(prox_span)
            vec["prox_min_pair"] = math.log1p(prox_pair)
            vec["prox_ordered"] = prox_ord

        if spec.coverage_tiers:
            vec["cov_full"] = cov_full
            vec["cov_gt_0_5"] = cov_gt_05
            vec["cov_gt_0_8"] = cov_gt_08

        emb_cos = float(r.get("emb_cos", 0.0))
        emb_l2 = float(r.get("emb_l2", 0.0))
        emb_dot = float(r.get("emb_dot", emb_cos))

        if spec.dense:
            vec["emb_cos"] = emb_cos
            vec["emb_l2"] = emb_l2
            vec["emb_dot"] = emb_dot

        if spec.dense_extras:
            vec["emb_cos_sq"] = emb_cos**2
            vec["bm25_x_cos"] = bm * emb_cos

        if spec.interactions:
            vec["bm25_x_cov"] = bm * cov
            vec["bm25_x_overlap"] = bm * float(overlap)
            vec["cos_x_cov"] = emb_cos * cov

        rows.append([vec[c] for c in ordered])

    out_df = pd.DataFrame(rows, columns=ordered, index=df.index)
    return out_df, ordered


def apply_per_query_feature_norm(
    feat_df: pd.DataFrame, qids: pd.Series, cols: Sequence[str]
) -> pd.DataFrame:
    """feature / max(feature) within each qid for selected columns."""
    out = feat_df.copy()
    m = pd.concat([qids.reset_index(drop=True), out[cols].reset_index(drop=True)], axis=1)
    for c in cols:
        if c not in out.columns:
            continue
        mx = m.groupby("qid")[c].transform("max").replace(0, np.nan).fillna(1.0)
        out[c] = out[c].values / mx.values
    return out


# ---------------------------------------------------------------------------
# Negative sampling (train rows only)
# ---------------------------------------------------------------------------
def add_negatives_random(
    df: pd.DataFrame,
    rel_pairs: Set[Tuple[str, str]],
    rng: random.Random,
    n_rand_per_query: int,
) -> pd.DataFrame:
    """Append random passages from other queries in df as extra candidates (label 0)."""
    by_q: Dict[str, List[str]] = defaultdict(list)
    for qid, pid in zip(df["qid"], df["pid"]):
        by_q[qid].append(pid)
    pool_pids = list({p for p in df["pid"].tolist()})
    if len(pool_pids) < 2:
        return df

    extra_rows = []
    for qid in df["qid"].unique():
        have = set(by_q[qid])
        tries = 0
        added = 0
        while added < n_rand_per_query and tries < n_rand_per_query * 20:
            tries += 1
            pid = rng.choice(pool_pids)
            if pid in have:
                continue
            if (qid, pid) in rel_pairs:
                continue
            # find passage text for pid
            sub = df[df["pid"] == pid]
            if sub.empty:
                continue
            row0 = sub.iloc[0]
            passage = row0["passage"]
            bm25 = float(row0["bm25"])
            er: dict = {
                "qid": qid,
                "pid": pid,
                "passage": passage,
                "bm25": bm25 * 0.1,
                "label": 0,
            }
            for col in ("emb_cos", "emb_l2", "emb_dot"):
                if col in row0.index:
                    er[col] = float(row0[col])
            er["_split"] = "train"
            extra_rows.append(er)
            have.add(pid)
            added += 1

    if not extra_rows:
        return df
    return pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)


def add_negatives_lowbm_highcos(
    df: pd.DataFrame,
    rng: random.Random,
    per_query: int,
) -> pd.DataFrame:
    """Within each query, take docs in bottom half of bm25 but top half of emb_cos; add if not already top."""
    if "emb_cos" not in df.columns:
        return df
    extra = []
    for qid, g in df.groupby("qid"):
        if len(g) < 4:
            continue
        g2 = g.sort_values("bm25", ascending=True)
        low = g2.head(len(g2) // 2)
        if low.empty:
            continue
        thr = np.median(low["emb_cos"])
        cand = low[low["emb_cos"] >= thr].head(per_query)
        for _, r in cand.iterrows():
            extra.append(r.to_dict())
    if not extra:
        return df
    ex = pd.DataFrame(extra)
    ex["label"] = ex["label"].clip(upper=1)  # keep true labels
    return pd.concat([df, ex], ignore_index=True)


# ---------------------------------------------------------------------------
# One experiment run
# ---------------------------------------------------------------------------
def xgb_objective_for_metric(metric: str) -> str:
    if metric == "mrr":
        return "rank:pairwise"
    return "rank:ndcg"


@dataclass
class RunConfig:
    name: str
    k_eval: int
    feature_spec: FeatureSpec
    eval_metric: str
    eval_cutoff: int
    neg_random: int = 0
    neg_hard: int = 0
    per_query_norm_bm25_cos: bool = False


def run_single_experiment(
    run: RunConfig,
    df_full: pd.DataFrame,
    qid_to_tokens: Dict[str, List[str]],
    rel_pairs: Set[Tuple[str, str]],
    idf_map: Dict[str, float],
    seed: int,
    xgb_params: dict,
    log: Callable[[str], None],
) -> Dict[str, float]:
    rng = random.Random(seed)
    metric = run.eval_metric
    cutoff = run.eval_cutoff
    K = run.k_eval

    df = df_full.copy()
    df = df.sort_values(["qid", "bm25"], ascending=[True, False])
    # keep top-K per query by bm25
    df = df.groupby("qid", group_keys=False).head(K)

    feat_spec = run.feature_spec
    if feat_spec.dense or feat_spec.dense_extras or feat_spec.interactions:
        if "emb_cos" not in df.columns:
            raise RuntimeError("Dense features requested but emb_cos missing on df")

    feat_mat, feat_cols = build_feature_matrix(df, qid_to_tokens, idf_map, feat_spec)
    df = df.drop(columns=[c for c in feat_cols if c in df.columns], errors="ignore")
    df = pd.concat([df.reset_index(drop=True), feat_mat.reset_index(drop=True)], axis=1)

    train_df = df[df["_split"] == "train"].copy()
    test_df = df[df["_split"] == "test"].copy()

    if run.neg_random > 0:
        train_df = add_negatives_random(train_df, rel_pairs, rng, run.neg_random)
    if run.neg_hard > 0:
        train_df = add_negatives_lowbm_highcos(train_df, rng, run.neg_hard)

    if run.neg_random > 0 or run.neg_hard > 0:
        tr_part, tr_cols = build_feature_matrix(train_df, qid_to_tokens, idf_map, feat_spec)
        train_df = train_df.drop(columns=[c for c in tr_cols if c in train_df.columns], errors="ignore")
        train_df = pd.concat([train_df.reset_index(drop=True), tr_part.reset_index(drop=True)], axis=1)
        feat_cols = tr_cols

    if run.per_query_norm_bm25_cos:
        norm_cols = [
            c
            for c in ("bm25_raw", "emb_cos", "bm25_log", "bm25_norm_q", "bm25")
            if c in feat_cols and c in train_df.columns
        ]
        if norm_cols:
            tr_sub = apply_per_query_feature_norm(train_df[norm_cols], train_df["qid"], norm_cols)
            for c in norm_cols:
                train_df[c] = tr_sub[c].values
            te_sub = apply_per_query_feature_norm(test_df[norm_cols], test_df["qid"], norm_cols)
            for c in norm_cols:
                test_df[c] = te_sub[c].values

    if train_df.empty or test_df.empty:
        log(f"  SKIP {run.name}: empty train/test")
        return {}

    if metric == "mrr":
        train_df = train_df[train_df.groupby("qid")["label"].transform("sum") > 0]
        test_df = test_df[test_df.groupby("qid")["label"].transform("sum") > 0]
        if train_df.empty or test_df.empty:
            log(f"  SKIP {run.name}: no positives per query for MRR filter")
            return {}

    scaler = StandardScaler()
    scaler.fit(train_df[feat_cols])
    train_df[feat_cols] = scaler.transform(train_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    X_train = train_df[feat_cols].to_numpy(np.float32)
    y_train = train_df["label"].to_numpy(np.float32)
    group_train = train_df.groupby("qid").size().to_list()

    obj = xgb_objective_for_metric(metric)
    model = XGBRanker(
        objective=obj,
        random_state=seed,
        n_jobs=-1,
        **xgb_params,
    )
    model.fit(X_train, y_train, group=group_train)

    bm25_scores = []
    cos_scores = []
    l2r_scores = []
    for _, g in test_df.groupby("qid"):
        gg = g.sort_values("bm25", ascending=False).head(K)
        rel_b = gg["label"].head(cutoff).tolist()
        bm25_scores.append(eval_metric(rel_b, metric, cutoff))
        if "emb_cos" in g.columns:
            gc = g.sort_values("emb_cos", ascending=False).head(K)
            cos_scores.append(eval_metric(gc["label"].head(cutoff).tolist(), metric, cutoff))
        Xq = gg[feat_cols].to_numpy(np.float32)
        pred = model.predict(Xq)
        gg2 = gg.assign(pred=pred).sort_values("pred", ascending=False)
        l2r_scores.append(eval_metric(gg2["label"].head(cutoff).tolist(), metric, cutoff))

    out = {
        "bm25": float(np.mean(bm25_scores)),
        "l2r": float(np.mean(l2r_scores)),
        "l2r_minus_bm25": float(np.mean(l2r_scores) - np.mean(bm25_scores)),
    }
    if cos_scores:
        out["cosine"] = float(np.mean(cos_scores))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="L2R experiment suite with flushed log file")
    ap.add_argument("--out", default="l2r_experiment_suite_results.txt", help="Log + results path")
    ap.add_argument("--queries", type=int, default=5000, help="Target number of queries (capped by file)")
    ap.add_argument("--max-k", type=int, default=1000, help="BM25 pool size per query (max K to fetch)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-metric", choices=["mrr", "ndcg"], default="mrr")
    ap.add_argument("--eval-cutoff", type=int, default=10)
    ap.add_argument("--filter-qrel", action="store_true", help="Only queries with relevant doc in top-K")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embedding-batch", type=int, default=64)
    ap.add_argument("--quick", action="store_true", help="Small grid: fewer queries and K list [100,500]")
    args = ap.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    log_path = args.out
    _out_dir = os.path.dirname(os.path.abspath(log_path))
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    _log_fp = open(log_path, "w", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"{datetime.now().isoformat(timespec='seconds')} | {msg}"
        print(line, flush=True)
        _log_fp.write(line + "\n")
        _log_fp.flush()
        sys.stdout.flush()

    log(f"Starting experiment suite → {os.path.abspath(log_path)}")
    log(f"args: {args}")

    ENV = load_dotenv_like(".env.local")
    es = Elasticsearch(ENV["ES_LOCAL_URL"], api_key=ENV["ES_LOCAL_API_KEY"])
    ES_INDEX = "msmarco"
    ES_EMB_FIELD = "embedding"

    if not es.ping():
        log("ERROR: ES ping failed")
        return

    queries = pd.read_csv(
        "queries.train.tsv", sep="\t", names=["qid", "query"], dtype=str
    )
    qrels = pd.read_csv(
        "qrels.train.tsv",
        sep="\t",
        names=["qid", "unused", "pid", "rel"],
        dtype=str,
    ).drop(columns=["unused"])
    qrels["rel"] = qrels["rel"].astype(int)
    rel_pairs = set(zip(qrels["qid"], qrels["pid"]))

    if args.filter_qrel:
        valid = set(qrels["qid"].unique())
        queries = queries[queries["qid"].isin(valid)].copy()
        log(f"Filtered to qrels qids: {len(queries)} queries")

    target_q = min(args.queries, len(queries))
    shuffled = queries.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    max_k = args.max_k
    if args.quick:
        target_q = min(800, target_q)
        max_k = min(100, max_k)

    log(f"Building candidate pool: target_q={target_q}, max_k={max_k}")

    rows: List[dict] = []
    kept_qids: List[str] = []
    qid_to_tokens: Dict[str, List[str]] = {}
    qid_to_query_text: Dict[str, str] = {}

    for _, row in shuffled.iterrows():
        if len(kept_qids) >= target_q:
            break
        qid = row["qid"]
        if qid in qid_to_tokens:
            continue
        resp = es.search(
            index=ES_INDEX,
            query={"match": {"passage": row["query"]}},
            size=max_k,
        )
        hits = resp["hits"]["hits"]
        if not hits:
            continue
        if args.filter_qrel:
            if not any((qid, str(h["_source"].get("pid", h["_id"]))) in rel_pairs for h in hits):
                continue
        qid_to_tokens[qid] = tokenize(row["query"])
        qid_to_query_text[qid] = row["query"]
        kept_qids.append(qid)
        for h in hits:
            pid = str(h["_source"].get("pid", h["_id"]))
            passage = h["_source"].get("passage", "")
            rows.append(
                {
                    "qid": qid,
                    "pid": pid,
                    "passage": passage,
                    "bm25": float(h["_score"]),
                }
            )

    df_base = pd.DataFrame(rows)
    if df_base.empty:
        log("ERROR: empty dataframe — check ES / queries file")
        return

    df_base["label"] = [
        1 if (q, p) in rel_pairs else 0
        for q, p in zip(df_base["qid"], df_base["pid"])
    ]

    if args.filter_qrel:
        pos = df_base.groupby("qid")["label"].sum()
        df_base = df_base[df_base["qid"].isin(pos[pos > 0].index)]
        log(f"After qrel filter: {df_base['qid'].nunique()} queries")

    passages_pool = df_base["passage"].tolist()
    idf_map = build_idf_from_passages(passages_pool)
    log(f"IDF vocab size: {len(idf_map)}")

    log("Computing dense embeddings (queries + ES vectors)...")
    emb_arr, emb_names = compute_embedding_block(
        df_base,
        qid_to_query_text,
        args.embedding_model,
        args.embedding_batch,
        seed,
        es,
        ES_INDEX,
        ES_EMB_FIELD,
        500,
    )
    for i, name in enumerate(emb_names):
        df_base[name] = emb_arr[:, i]

    all_qids = df_base["qid"].unique().tolist()
    random.shuffle(all_qids)
    split_at = int(0.8 * len(all_qids))
    train_set = set(all_qids[:split_at])
    test_set = set(all_qids[split_at:])
    df_base["_split"] = df_base["qid"].apply(lambda q: "train" if q in train_set else "test")

    # K values to try (subset of max_k)
    if args.quick:
        k_list = [100, min(500, max_k)]
    else:
        k_list = sorted({100, 500, min(1000, max_k), max_k})

    log(f"Train queries: {len(train_set)} | Test queries: {len(test_set)} | K list: {k_list}")

    xgb_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    }

    specs: List[Tuple[str, FeatureSpec]] = [
        ("01_baseline", FeatureSpec("baseline")),
        ("02_bm25_variants", FeatureSpec("b", bm25_variants=True)),
        ("03_plus_idf_tf", FeatureSpec("c", bm25_variants=True, idf_tf=True)),
        ("04_plus_proximity", FeatureSpec("d", bm25_variants=True, idf_tf=True, proximity=True)),
        ("05_plus_coverage_tiers", FeatureSpec("e", bm25_variants=True, idf_tf=True, proximity=True, coverage_tiers=True)),
        ("06_plus_dense", FeatureSpec("f", bm25_variants=True, idf_tf=True, proximity=True, coverage_tiers=True, dense=True)),
        ("07_plus_dense_extras", FeatureSpec("g", bm25_variants=True, idf_tf=True, proximity=True, coverage_tiers=True, dense=True, dense_extras=True)),
        ("08_plus_query_norm_flags", FeatureSpec("h", bm25_variants=True, idf_tf=True, proximity=True, coverage_tiers=True, dense=True, dense_extras=True)),
        ("09_plus_interactions", FeatureSpec("i", bm25_variants=True, idf_tf=True, proximity=True, coverage_tiers=True, dense=True, dense_extras=True, interactions=True)),
    ]

    results_rows: List[Dict] = []

    for k_eval in k_list:
        log("")
        log(f"========== K_eval = {k_eval} (pool built with max_k={max_k}) ==========")
        for spec_name, fspec in specs:
            run = RunConfig(
                name=f"{spec_name}_k{k_eval}",
                k_eval=k_eval,
                feature_spec=fspec,
                eval_metric=args.eval_metric,
                eval_cutoff=args.eval_cutoff,
                neg_random=0,
                neg_hard=0,
                per_query_norm_bm25_cos=(spec_name >= "08_"),
            )
            log(f"-- Run: {run.name} | objective={xgb_objective_for_metric(args.eval_metric)}")
            try:
                met = run_single_experiment(
                    run,
                    df_base,
                    qid_to_tokens,
                    rel_pairs,
                    idf_map,
                    seed,
                    xgb_params,
                    log,
                )
            except Exception as e:
                log(f"  ERROR {run.name}: {e}")
                continue
            if not met:
                continue
            log(
                f"  → BM25={met['bm25']:.4f} | L2R={met['l2r']:.4f} (Δ {met['l2r_minus_bm25']:+.4f})"
                + (f" | cos={met['cosine']:.4f}" if "cosine" in met else "")
            )
            results_rows.append({"k": k_eval, "spec": spec_name, **met})

        # Neg sampling + full feature stack (09)
        for neg_label, n_rand, n_hard in [("neg_rand5", 5, 0), ("neg_hard3", 0, 3)]:
            run = RunConfig(
                name=f"10_{neg_label}_k{k_eval}",
                k_eval=k_eval,
                feature_spec=FeatureSpec(
                    "i2",
                    bm25_variants=True,
                    idf_tf=True,
                    proximity=True,
                    coverage_tiers=True,
                    dense=True,
                    dense_extras=True,
                    interactions=True,
                ),
                eval_metric=args.eval_metric,
                eval_cutoff=args.eval_cutoff,
                neg_random=n_rand,
                neg_hard=n_hard,
                per_query_norm_bm25_cos=True,
            )
            log(f"-- Run: {run.name} (neg_random={n_rand}, neg_hard={n_hard})")
            try:
                met = run_single_experiment(
                    run,
                    df_base,
                    qid_to_tokens,
                    rel_pairs,
                    idf_map,
                    seed,
                    xgb_params,
                    log,
                )
            except Exception as e:
                log(f"  ERROR {run.name}: {e}")
                continue
            if not met:
                continue
            log(
                f"  → BM25={met['bm25']:.4f} | L2R={met['l2r']:.4f} (Δ {met['l2r_minus_bm25']:+.4f})"
            )
            results_rows.append({"k": k_eval, "spec": run.name, **met})

    res_df = pd.DataFrame(results_rows)
    log("")
    log("================ FINAL TABLE ================")
    if not res_df.empty:
        log(res_df.sort_values(["k", "spec"]).to_string(index=False))
    else:
        log("(no results)")
    log("Done.")
    _log_fp.close()


if __name__ == "__main__":
    main()
