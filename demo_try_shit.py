# ================================
# FULL EXPERIMENT SCRIPT (ES + L2R)
# ================================
import gc
import math
import os
import re
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker


# ================================
# CONFIG
# ================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# How many distinct qids to sample (shuffled, fixed seed) from train / dev (after qrels filter).
TRAIN_QUERY_LIMIT = 10000
EVAL_QUERY_LIMIT = 5000

CANDIDATE_KS = [100,500,1000]
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

# Evaluation: MRR@K on dev (same BM25 candidate pool for BM25 ordering and L2R re-ranking).
EVAL_METRIC = "mrr"
EVAL_CUTOFF = 10
if EVAL_METRIC not in ("ndcg", "mrr"):
    raise ValueError("EVAL_METRIC must be 'ndcg' or 'mrr'")

# Option C: optional dense embedding features (CPU-friendly small model).
# Requires: pip install sentence-transformers torch
# (CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu
#  then pip install sentence-transformers)
USE_EMBEDDING_FEATURES = True
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 128  # query encode batch size
ES_MGET_BATCH_SIZE = 500  # passage vectors via ES mget

# Extra proximity: min window (unordered), optimal ordered window, pair avg distance, cluster density.
USE_PROX_FEATURES = True

# Optional eval-only rerank: BM25 → L2R → take top min(rerank_k, K) by L2R → CrossEncoder → final list.
# Requires: sentence-transformers (same stack as bi-encoder features).
USE_CROSS_ENCODER_RERANK = True
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_RERANK_K = int(os.environ.get("CROSS_ENCODER_RERANK_K", "100"))
CROSS_ENCODER_BATCH_SIZE = int(os.environ.get("CROSS_ENCODER_BATCH_SIZE", "64"))

# Memory: BM25 + featurize in query-sized chunks (not all passages at once).
QUERY_BUILD_BATCH = int(os.environ.get("QUERY_BUILD_BATCH", "200"))
# Running mean MRR print interval during eval.
MRR_PROGRESS_EVERY = int(os.environ.get("MRR_PROGRESS_EVERY", "100"))

# ================================
# ENV LOADER
# ================================
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


ENV = load_dotenv_like(".env.local")

ES_URL = ENV["ES_LOCAL_URL"]
ES_API_KEY = ENV["ES_LOCAL_API_KEY"]
# ES_INDEX = ENV.get("ES_INDEX", "msmarco")
ES_INDEX = "msmarco"

es = Elasticsearch(ES_URL, api_key=ES_API_KEY)

ES_PASSAGE_EMB_INDEX = ENV.get("ES_PASSAGE_EMB_INDEX", ES_INDEX)
ES_PASSAGE_EMB_FIELD = ENV.get("ES_PASSAGE_EMB_FIELD", "embedding")

if USE_EMBEDDING_FEATURES:
    print(
        "Embedding features: ON | queries:",
        EMBEDDING_MODEL_NAME,
        "| passages: ES",
        ES_PASSAGE_EMB_INDEX,
        f"field={ES_PASSAGE_EMB_FIELD!r} mget_batch={ES_MGET_BATCH_SIZE}",
    )
else:
    print("Embedding features: OFF (set USE_EMBEDDING_FEATURES = True for Option C)")

if USE_PROX_FEATURES:
    print("Proximity extras: ON (min_window, ordered_window, pair_avg, cluster_density)")
else:
    print("Proximity extras: OFF (set USE_PROX_FEATURES = True)")

if USE_CROSS_ENCODER_RERANK:
    print(
        "Cross-encoder rerank: ON | model:",
        CROSS_ENCODER_MODEL_NAME,
        f"| top-{CROSS_ENCODER_RERANK_K} after L2R | batch={CROSS_ENCODER_BATCH_SIZE}",
    )
else:
    print(
        "Cross-encoder rerank: OFF (set USE_CROSS_ENCODER_RERANK = True; "
        "eval: L2R top-K → CE scores → final order)",
    )

print(f"Eval metric: {EVAL_METRIC.upper()}@{EVAL_CUTOFF}")

print("Ping:", es.ping())
print("Index exists:", es.indices.exists(index=ES_INDEX))

# ================================
# HELPERS
# ================================
def tokenize(x):
    return str(x).lower().split()


def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    if n < 1 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _distinct_query_terms(q_tokens: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for t in q_tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def prox_min_window_unordered(q_tokens: List[str], d_tokens: List[str], max_doc: int = 400) -> float:
    """
    Smallest doc span covering all distinct query terms (any order).
    Returns 1 / (1 + min_window) with min_window = pos_max - pos_min, or 0 if impossible.
    """
    d = d_tokens[:max_doc]
    U = _distinct_query_terms(q_tokens)
    if not U:
        return 0.0
    term_to_id = {t: i for i, t in enumerate(U)}
    k = len(U)
    events: List[Tuple[int, int]] = []
    for ti, t in enumerate(U):
        for pos in range(len(d)):
            if d[pos] == t:
                events.append((pos, ti))
    events.sort(key=lambda x: x[0])
    if len(events) < k:
        return 0.0
    cnt: Dict[int, int] = defaultdict(int)
    covered = 0
    l = 0
    best: Optional[int] = None
    for r in range(len(events)):
        tid = events[r][1]
        if cnt[tid] == 0:
            covered += 1
        cnt[tid] += 1
        while covered == k:
            pos_l = events[l][0]
            pos_r = events[r][0]
            span = pos_r - pos_l
            best = span if best is None else min(best, span)
            tid_l = events[l][1]
            cnt[tid_l] -= 1
            if cnt[tid_l] == 0:
                covered -= 1
            l += 1
    if best is None:
        return 0.0
    return 1.0 / (1.0 + float(best))


def prox_ordered_window_optimal(q_tokens: List[str], d_tokens: List[str], max_doc: int = 400) -> float:
    """
    Minimum span over embeddings of q_tokens as an ordered subsequence (not greedy per step).
    Returns 1 / (1 + span) or 0 if no match.
    """
    d = d_tokens[:max_doc]
    k = len(q_tokens)
    if k == 0:
        return 0.0
    n = len(d)
    INF = n + k + 10
    # g[t][j] = min start index for matching q[0..t] with q[t] at j
    g: List[List[int]] = [[INF] * n for _ in range(k)]
    for j in range(n):
        if d[j] == q_tokens[0]:
            g[0][j] = j
    for t in range(1, k):
        for j in range(n):
            if d[j] != q_tokens[t]:
                continue
            best = INF
            for jprev in range(j):
                if g[t - 1][jprev] < INF:
                    best = min(best, g[t - 1][jprev])
            g[t][j] = best
    best_span = INF
    for j in range(n):
        if g[k - 1][j] < INF:
            best_span = min(best_span, j - g[k - 1][j])
    if best_span >= INF:
        return 0.0
    return 1.0 / (1.0 + float(best_span))


def prox_pair_avg_distance(q_tokens: List[str], d_tokens: List[str], max_doc: int = 400) -> float:
    """
    Mean |pos_i - pos_j| over unordered pairs of distinct query terms (first occurrence each).
    Returns 1 / (1 + avg_pair_dist), or 1.0 if <2 terms (no pairs), 0 if a term is missing.
    """
    d = d_tokens[:max_doc]
    U = _distinct_query_terms(q_tokens)
    if len(U) < 2:
        return 1.0
    pos_map: Dict[str, int] = {}
    for t in U:
        try:
            pos_map[t] = d.index(t)
        except ValueError:
            return 0.0
    dists: List[float] = []
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            dists.append(abs(float(pos_map[U[i]] - pos_map[U[j]])))
    avg = sum(dists) / len(dists)
    return 1.0 / (1.0 + avg)


def prox_cluster_density(q_tokens: List[str], d_tokens: List[str], max_doc: int = 400) -> float:
    """
    First occurrence per distinct query term; density = (# matched) / (max_pos - min_pos),
    or #matched when span is 0. Returns 0 if any term missing.
    """
    d = d_tokens[:max_doc]
    U = _distinct_query_terms(q_tokens)
    if not U:
        return 0.0
    positions: List[int] = []
    for t in U:
        try:
            positions.append(d.index(t))
        except ValueError:
            return 0.0
    span = max(positions) - min(positions)
    cnt = len(positions)
    if span == 0:
        return float(cnt)
    return float(cnt) / float(span)


def proximity_greedy_sequential(q_tokens: List[str], d_tokens: List[str], max_doc: int = 400) -> float:
    """
    Greedy sequential match: walk query tokens left-to-right, first match in doc after previous.
    Returns score in (0,1], higher when query terms appear close in order.
    """
    if not q_tokens:
        return 0.0
    d = d_tokens[:max_doc]
    pos = 0
    total_gap = 0
    for t in q_tokens:
        try:
            j = d.index(t, pos)
        except ValueError:
            return 0.0
        total_gap += j - pos
        pos = j + 1
    return 1.0 / (1.0 + total_gap / max(1, len(q_tokens)))


def lexical_features_row(
    q_tokens: List[str],
    passage: str,
    d_tokens: List[str],
) -> List[float]:
    d_set = set(d_tokens)
    bi_q, bi_d = ngrams(q_tokens, 2), ngrams(d_tokens, 2)
    tri_q, tri_d = ngrams(q_tokens, 3), ngrams(d_tokens, 3)

    overlap = float(sum(1 for t in q_tokens if t in d_set))
    coverage = overlap / max(1, len(q_tokens))
    doc_len_log = math.log1p(len(d_tokens))
    q_joined = " ".join(q_tokens)
    d_joined = " ".join(d_tokens)
    phrase_exact = float(q_joined in d_joined)

    bi_inter = bi_q & bi_d
    tri_overlap = float(len(tri_q & tri_d))
    bi_union = len(bi_q | bi_d)
    bi_jacc = float(len(bi_inter) / bi_union) if bi_union else 0.0

    prox_score = proximity_greedy_sequential(q_tokens, d_tokens)

    out = [
        coverage,
        doc_len_log,
        phrase_exact,
        tri_overlap,
        bi_jacc,
        prox_score,
    ]
    if USE_PROX_FEATURES:
        out.extend([
            prox_min_window_unordered(q_tokens, d_tokens),
            prox_ordered_window_optimal(q_tokens, d_tokens),
            prox_pair_avg_distance(q_tokens, d_tokens),
            prox_cluster_density(q_tokens, d_tokens),
        ])
    return out


LEX_COLS_BASE = [
    "coverage",
    "doc_len_log",
    "phrase_exact",
    "tri_overlap",
    "bi_jacc",
    "prox_score",
]
PROX_COLS = [
    "prox_min_window",
    "prox_ordered_window",
    "prox_pair_avg",
    "prox_cluster_density",
]
LEX_COLS = LEX_COLS_BASE + (PROX_COLS if USE_PROX_FEATURES else [])

PER_QUERY_COLS = ["bm25_norm", "inv_rank", "rank_pct"]


def add_per_query_bm25_rank_features(df: pd.DataFrame) -> None:
    """In-place: min-max BM25 within each qid + rank / inverse-rank (by raw BM25 desc)."""
    g = df.groupby("qid", sort=False)["bm25"]

    def _minmax(s: pd.Series) -> pd.Series:
        lo = float(s.min())
        hi = float(s.max())
        den = hi - lo
        if den < 1e-12:
            return pd.Series(0.5, index=s.index, dtype=float)
        return (s - lo) / den

    df["bm25_norm"] = g.transform(_minmax)
    rk = df.groupby("qid", sort=False)["bm25"].rank(ascending=False, method="first")
    ns = df.groupby("qid", sort=False)["bm25"].transform("size")
    df["inv_rank"] = 1.0 / rk
    df["rank_pct"] = (rk - 1.0) / (ns - 1.0).replace(0, 1)

def bm25_search(query, k):
    resp = es.search(
        index="msmarco",
        query={"match": {"passage": query}},
        size=k
    )
    return [
        (
            str(h["_source"].get("pid", h["_id"])),
            h["_source"].get("passage", ""),
            float(h["_score"])
        )
        for h in resp["hits"]["hits"]
    ]


def ndcg(rels, k=10):
    rels = rels[:k]
    dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(rels, k=10):
    """Reciprocal rank of the first relevant document (rel > 0) within the top-k list; 0 if none."""
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def eval_query_metric(rels: List[int], metric: str, k: int) -> float:
    if metric == "ndcg":
        return ndcg(rels, k=k)
    if metric == "mrr":
        return mrr_at_k(rels, k=k)
    raise ValueError(f"Unknown EVAL_METRIC: {metric!r} (use 'ndcg' or 'mrr')")


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
    """Batch mget passage vectors from ES (same pattern as demo_l2r_experiment_suite)."""
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


def compute_embedding_features(
    df: pd.DataFrame,
    qid_to_query_text: Dict[str, str],
    model_name: str,
    batch_size: int,
    seed: int,
    es_client: Elasticsearch,
    passage_index: str,
    passage_field: str,
    mget_batch_size: int,
    passage_vec_cache: Optional[Dict[str, np.ndarray]] = None,
    st_model: Optional[Any] = None,
    show_progress_bar: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode unique queries with SentenceTransformer; passage side from ES mget only.
    Reuses vectors in passage_vec_cache when provided (across K / train+eval).
    Pass st_model to avoid reloading the transformer on every batch.
    """
    from sentence_transformers import SentenceTransformer

    np.random.seed(seed)
    if st_model is None:
        device = ENV.get("EMBEDDING_DEVICE", "cuda")
        st_model = SentenceTransformer(model_name, device=device)

    uqids = df["qid"].unique().tolist()
    q_texts = [qid_to_query_text[q] for q in uqids]
    qid_to_row = {q: i for i, q in enumerate(uqids)}

    pid_first = df.drop_duplicates(subset=["pid"], keep="first")
    upids = pid_first["pid"].tolist()
    pid_to_row = {p: i for i, p in enumerate(upids)}

    cache = passage_vec_cache if passage_vec_cache is not None else {}
    need = [p for p in upids if p not in cache]
    if need:
        fetched, n_miss = fetch_passage_embeddings_from_es(
            es_client,
            passage_index,
            need,
            passage_field,
            mget_batch_size,
        )
        cache.update(fetched)
        if n_miss:
            print(
                f"Embedding: ES mget missing {n_miss}/{len(need)} passage vectors "
                f"(index={passage_index!r} field={passage_field!r})"
            )

    dim = None
    for p in upids:
        v = cache.get(p)
        if v is not None:
            dim = int(v.shape[0])
            break
    if dim is None:
        raise RuntimeError(
            f"No passage embeddings in index {passage_index!r} field {passage_field!r} "
            f"for {len(upids)} unique pids (check ES mapping / indexing)."
        )

    p_emb = np.zeros((len(upids), dim), dtype=np.float64)
    for j, p in enumerate(upids):
        v = cache.get(p)
        if v is not None and v.shape[0] == dim:
            p_emb[j] = v
        elif v is not None:
            raise ValueError(f"Inconsistent embedding dim for pid={p!r}")

    p_emb = _l2_normalize_rows(p_emb)

    if show_progress_bar:
        print(
            f"Embedding: model={model_name} | unique queries={len(uqids)} | "
            f"unique passages={len(upids)} | ES fetch this call: {len(need)} pids | "
            f"cache size={len(cache)}",
            flush=True,
        )

    q_emb = st_model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    qi = np.array([qid_to_row[q] for q in df["qid"].to_numpy()], dtype=np.int64)
    pi = np.array([pid_to_row[p] for p in df["pid"].to_numpy()], dtype=np.int64)
    qv = q_emb[qi]
    pv = p_emb[pi]
    emb_cos = np.sum(qv * pv, axis=1)
    emb_l2 = np.linalg.norm(qv - pv, axis=1)
    emb_dot = np.sum(qv * pv, axis=1)
    emb_stack = np.column_stack([emb_cos, emb_l2, emb_dot]).astype(np.float64)
    return emb_stack, ["emb_cos", "emb_l2", "emb_dot"]


# ================================
# LOAD DATA
# ================================
queries_train_all = pd.read_csv(
    "queries.train.tsv",
    sep="\t",
    names=["qid", "query"],
    dtype=str,
)
queries_dev_all = pd.read_csv(
    "queries.dev.tsv",
    sep="\t",
    names=["qid", "query"],
    dtype=str,
)

qrels_train = pd.read_csv(
    "qrels.train.tsv",
    sep="\t",
    names=["qid", "unused", "pid", "rel"],
    dtype=str,
).drop(columns=["unused"])
qrels_dev = pd.read_csv(
    "qrels.dev.tsv",
    sep="\t",
    names=["qid", "unused", "pid", "rel"],
    dtype=str,
).drop(columns=["unused"])

qrels_train["rel"] = qrels_train["rel"].astype(int)
qrels_dev["rel"] = qrels_dev["rel"].astype(int)

train_qrel_qids = set(qrels_train["qid"].unique())
dev_qrel_qids = set(qrels_dev["qid"].unique())
queries_train = queries_train_all[queries_train_all["qid"].isin(train_qrel_qids)].copy()
queries_dev = queries_dev_all[queries_dev_all["qid"].isin(dev_qrel_qids)].copy()

rel_pairs_train = set(zip(qrels_train["qid"], qrels_train["pid"]))
rel_pairs_dev = set(zip(qrels_dev["qid"], qrels_dev["pid"]))

print(
    "Train queries (in qrels.train):",
    len(queries_train),
    "| Dev queries (in qrels.dev):",
    len(queries_dev),
)


def sample_unique_qid_queries(
    queries_df: pd.DataFrame, max_qids: int, seed: int
) -> List[Tuple[str, str]]:
    """Up to max_qids distinct (qid, query) rows in shuffled table order."""
    shuffled = queries_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out: List[Tuple[str, str]] = []
    seen = set()
    for _, row in shuffled.iterrows():
        qid = row["qid"]
        if qid in seen:
            continue
        seen.add(qid)
        out.append((qid, str(row["query"])))
        if len(out) >= max_qids:
            break
    return out


RESULTS = []

# ================================
# EXPERIMENT LOOP
# ================================
train_cap = min(TRAIN_QUERY_LIMIT, queries_train["qid"].nunique())
eval_cap = min(EVAL_QUERY_LIMIT, queries_dev["qid"].nunique())
train_qid_queries = sample_unique_qid_queries(queries_train, train_cap, SEED)
eval_qid_queries = sample_unique_qid_queries(queries_dev, eval_cap, SEED + 1)

# Reuse passage vectors from ES across K values (mget only for unseen pids).
passage_emb_cache: Dict[str, np.ndarray] = {}

print(
    f"Batched pipeline: QUERY_BUILD_BATCH={QUERY_BUILD_BATCH} | "
    f"MRR_PROGRESS_EVERY={MRR_PROGRESS_EVERY}",
    flush=True,
)

for K in CANDIDATE_KS:
    print(
        f"\nRunning: train_qids={len(train_qid_queries)} (cap {train_cap}), "
        f"eval_qids={len(eval_qid_queries)} (cap {eval_cap}), K={K}",
        flush=True,
    )

    def build_lexical_matrix(
        df: pd.DataFrame, tok_map: Dict[str, List[str]]
    ) -> np.ndarray:
        return np.vstack([
            lexical_features_row(
                tok_map[row["qid"]],
                row["passage"],
                tokenize(row["passage"]),
            )
            for _, row in df.iterrows()
        ])

    st_model: Optional[Any] = None
    if USE_EMBEDDING_FEATURES:
        from sentence_transformers import SentenceTransformer

        st_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, device=ENV.get("EMBEDDING_DEVICE", "cuda")
        )
        print("  Loaded SentenceTransformer once for this K (query encode per batch).", flush=True)

    # ------------------------
    # TRAIN: BM25 + features in query batches; drop passage text after featurizing
    # ------------------------
    qid_to_tokens_tr: Dict[str, List[str]] = {}
    qid_to_query_text_tr: Dict[str, str] = {}
    train_with_hits = 0
    train_empty_hits = 0
    train_chunks: List[pd.DataFrame] = []
    emb_names: List[str] = []
    n_tr = len(train_qid_queries)
    n_tr_batches = (n_tr + QUERY_BUILD_BATCH - 1) // QUERY_BUILD_BATCH

    print("  [1/2] Train pools + lexical + optional embeddings (batched)...", flush=True)
    for bi, start in enumerate(range(0, n_tr, QUERY_BUILD_BATCH)):
        batch_pairs = train_qid_queries[start : start + QUERY_BUILD_BATCH]
        rows: List[dict] = []
        for qid, query_text in batch_pairs:
            hits = bm25_search(query_text, k=K)
            if not hits:
                train_empty_hits += 1
                continue
            train_with_hits += 1
            qid_to_tokens_tr[qid] = tokenize(query_text)
            qid_to_query_text_tr[qid] = query_text
            for pid, passage, score in hits:
                rows.append({
                    "qid": qid,
                    "pid": pid,
                    "passage": passage,
                    "bm25": score,
                })
        if not rows:
            print(
                f"    train batch {bi + 1}/{n_tr_batches}: no rows",
                flush=True,
            )
            continue
        bdf = pd.DataFrame(rows)
        bdf["label"] = [
            1 if (qid, pid) in rel_pairs_train else 0
            for qid, pid in zip(bdf["qid"], bdf["pid"])
        ]
        bdf = bdf[bdf.groupby("qid")["label"].transform("sum") > 0]
        if bdf.empty:
            print(
                f"    train batch {bi + 1}/{n_tr_batches}: no positives after filter",
                flush=True,
            )
            del bdf, rows
            gc.collect()
            continue
        bdf[LEX_COLS] = build_lexical_matrix(bdf, qid_to_tokens_tr)
        add_per_query_bm25_rank_features(bdf)
        if USE_EMBEDDING_FEATURES:
            emb_arr, emb_names = compute_embedding_features(
                bdf,
                qid_to_query_text_tr,
                EMBEDDING_MODEL_NAME,
                EMBEDDING_BATCH_SIZE,
                SEED,
                es,
                ES_PASSAGE_EMB_INDEX,
                ES_PASSAGE_EMB_FIELD,
                ES_MGET_BATCH_SIZE,
                passage_emb_cache,
                st_model=st_model,
                show_progress_bar=(bi == 0),
            )
            bdf[emb_names] = emb_arr
        bdf = bdf.drop(columns=["passage"], errors="ignore")
        train_chunks.append(bdf)
        total_rows = sum(len(x) for x in train_chunks)
        print(
            f"    train batch {bi + 1}/{n_tr_batches}: +{len(bdf)} rows | cumulative_rows={total_rows}",
            flush=True,
        )
        del rows
        gc.collect()

    if not train_chunks:
        print("Skipped (no train rows — every sampled train query had empty BM25)", flush=True)
        del st_model
        gc.collect()
        continue

    feat_cols: List[str] = ["bm25"] + PER_QUERY_COLS + LEX_COLS + emb_names
    scale_cols = [c for c in feat_cols if c != "bm25"]

    scaler = StandardScaler()
    for bdf in train_chunks:
        scaler.partial_fit(bdf[scale_cols].to_numpy(dtype=np.float64))

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    group_train: List[int] = []
    for bdf in train_chunks:
        bdf = bdf.sort_values(["qid", "bm25"], ascending=[True, False])
        bdf = bdf.copy()
        bdf[scale_cols] = scaler.transform(
            bdf[scale_cols].to_numpy(dtype=np.float64)
        )
        X_parts.append(bdf[feat_cols].to_numpy(np.float32))
        y_parts.append(bdf["label"].to_numpy(np.float32))
        group_train.extend(bdf.groupby("qid").size().to_list())

    X_train = np.vstack(X_parts)
    y_train = np.concatenate(y_parts)
    del train_chunks, X_parts, y_parts
    gc.collect()

    print(
        f"Train rows: {len(y_train)} | train qids w/ ≥1 hit: {train_with_hits} / "
        f"{len(train_qid_queries)} (empty BM25: {train_empty_hits})",
        flush=True,
    )

    model = XGBRanker(
        objective="rank:pairwise",
        random_state=SEED,
        n_jobs=4,
        **XGB_PARAMS,
    )
    model.fit(X_train, y_train, group=group_train)
    del X_train, y_train
    gc.collect()

    ce_model: Optional[Any] = None
    if USE_CROSS_ENCODER_RERANK:
        from sentence_transformers import CrossEncoder

        ce_model = CrossEncoder(
            CROSS_ENCODER_MODEL_NAME,
            device=ENV.get("EMBEDDING_DEVICE", "cuda"),
        )
        print(
            "  Loaded CrossEncoder for eval rerank (L2R top-"
            f"{CROSS_ENCODER_RERANK_K} → CE → final).",
            flush=True,
        )

    # ------------------------
    # EVAL: BM25 + features in batches; MRR with running means (no full eval_df)
    # ------------------------
    n_eval_total = len(eval_qid_queries)
    eval_empty_hits = 0

    print(
        f"  [2/2] Eval MRR — {n_eval_total} qids in slices of {QUERY_BUILD_BATCH} "
        f"(running {EVAL_METRIC.upper()}@{EVAL_CUTOFF} every {MRR_PROGRESS_EVERY})...",
        flush=True,
    )
    bm25_sum = 0.0
    l2r_sum = 0.0
    ce_sum = 0.0
    n_done = 0
    n_ev_batches = (n_eval_total + QUERY_BUILD_BATCH - 1) // QUERY_BUILD_BATCH

    for ei, estart in enumerate(range(0, n_eval_total, QUERY_BUILD_BATCH)):
        batch_pairs = eval_qid_queries[estart : estart + QUERY_BUILD_BATCH]
        rows = []
        qid_to_tokens_ev: Dict[str, List[str]] = {}
        qid_to_query_text_ev: Dict[str, str] = {}
        for qid, query_text in batch_pairs:
            hits = bm25_search(query_text, k=K)
            if not hits:
                continue
            qid_to_tokens_ev[qid] = tokenize(query_text)
            qid_to_query_text_ev[qid] = query_text
            for pid, passage, score in hits:
                rows.append({
                    "qid": qid,
                    "pid": pid,
                    "passage": passage,
                    "bm25": score,
                })

        if not rows:
            eval_empty_hits += len(batch_pairs)
            for _ in batch_pairs:
                n_done += 1
                if MRR_PROGRESS_EVERY and (
                    n_done % MRR_PROGRESS_EVERY == 0 or n_done == n_eval_total
                ):
                    if USE_CROSS_ENCODER_RERANK:
                        print(
                            f"  [MRR] {n_done}/{n_eval_total}  running {EVAL_METRIC.upper()}@{EVAL_CUTOFF}  "
                            f"BM25={bm25_sum / n_done:.4f}  L2R={l2r_sum / n_done:.4f}  "
                            f"CE={ce_sum / n_done:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  [MRR] {n_done}/{n_eval_total}  running {EVAL_METRIC.upper()}@{EVAL_CUTOFF}  "
                            f"BM25={bm25_sum / n_done:.4f}  L2R={l2r_sum / n_done:.4f}",
                            flush=True,
                        )
            print(
                f"    eval batch {ei + 1}/{n_ev_batches}: no ES hits (all zeros)",
                flush=True,
            )
            continue

        edf = pd.DataFrame(rows)
        edf["label"] = [
            1 if (qid, pid) in rel_pairs_dev else 0
            for qid, pid in zip(edf["qid"], edf["pid"])
        ]
        edf[LEX_COLS] = build_lexical_matrix(edf, qid_to_tokens_ev)
        add_per_query_bm25_rank_features(edf)
        if USE_EMBEDDING_FEATURES:
            emb_ev, _ = compute_embedding_features(
                edf,
                qid_to_query_text_ev,
                EMBEDDING_MODEL_NAME,
                EMBEDDING_BATCH_SIZE,
                SEED + 2,
                es,
                ES_PASSAGE_EMB_INDEX,
                ES_PASSAGE_EMB_FIELD,
                ES_MGET_BATCH_SIZE,
                passage_emb_cache,
                st_model=st_model,
                show_progress_bar=False,
            )
            edf[emb_names] = emb_ev
        if not USE_CROSS_ENCODER_RERANK:
            edf = edf.drop(columns=["passage"], errors="ignore")
        edf[scale_cols] = scaler.transform(
            edf[scale_cols].to_numpy(dtype=np.float64)
        )

        for qid, _ in batch_pairs:
            g = edf[edf["qid"] == qid]
            if g.empty:
                eval_empty_hits += 1
                m_b = m_l = m_ce = 0.0
            else:
                pool = g.sort_values("bm25", ascending=False).head(K)
                rels_bm25 = pool["label"].head(EVAL_CUTOFF).tolist()
                m_b = eval_query_metric(rels_bm25, EVAL_METRIC, EVAL_CUTOFF)
                Xq = pool[feat_cols].to_numpy(np.float32)
                preds = model.predict(Xq)
                l2r_ranked = pool.assign(pred=preds).sort_values("pred", ascending=False)
                rels_l2r = l2r_ranked["label"].head(EVAL_CUTOFF).tolist()
                m_l = eval_query_metric(rels_l2r, EVAL_METRIC, EVAL_CUTOFF)
                if USE_CROSS_ENCODER_RERANK and ce_model is not None:
                    rk = min(CROSS_ENCODER_RERANK_K, len(l2r_ranked))
                    top = l2r_ranked.head(rk)
                    rest = l2r_ranked.iloc[rk:]
                    qtext = qid_to_query_text_ev[qid]
                    pairs = [[qtext, str(row["passage"])] for _, row in top.iterrows()]
                    ce_scores = ce_model.predict(
                        pairs,
                        batch_size=CROSS_ENCODER_BATCH_SIZE,
                        show_progress_bar=False,
                    )
                    top_ce = top.assign(ce_score=ce_scores).sort_values(
                        "ce_score", ascending=False
                    )
                    final_ranked = pd.concat([top_ce, rest], ignore_index=True)
                    rels_ce = final_ranked["label"].head(EVAL_CUTOFF).tolist()
                    m_ce = eval_query_metric(rels_ce, EVAL_METRIC, EVAL_CUTOFF)
                else:
                    m_ce = m_l
            bm25_sum += m_b
            l2r_sum += m_l
            ce_sum += m_ce
            n_done += 1
            if MRR_PROGRESS_EVERY and (
                n_done % MRR_PROGRESS_EVERY == 0 or n_done == n_eval_total
            ):
                if USE_CROSS_ENCODER_RERANK:
                    print(
                        f"  [MRR] {n_done}/{n_eval_total}  running {EVAL_METRIC.upper()}@{EVAL_CUTOFF}  "
                        f"BM25={bm25_sum / n_done:.4f}  L2R={l2r_sum / n_done:.4f}  "
                        f"CE={ce_sum / n_done:.4f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [MRR] {n_done}/{n_eval_total}  running {EVAL_METRIC.upper()}@{EVAL_CUTOFF}  "
                        f"BM25={bm25_sum / n_done:.4f}  L2R={l2r_sum / n_done:.4f}",
                        flush=True,
                    )

        del edf, rows, qid_to_tokens_ev, qid_to_query_text_ev
        gc.collect()
        print(
            f"    eval batch {ei + 1}/{n_ev_batches} done | queries_scored_so_far={n_done}",
            flush=True,
        )

    del st_model
    if USE_CROSS_ENCODER_RERANK:
        del ce_model
    gc.collect()

    print(
        f"Eval qids: {n_eval_total} | w/ ≥1 hit: {n_eval_total - eval_empty_hits} | "
        f"empty BM25: {eval_empty_hits}",
        flush=True,
    )

    bm25_mean = bm25_sum / n_done if n_done else 0.0
    l2r_mean = l2r_sum / n_done if n_done else 0.0
    ce_mean = ce_sum / n_done if n_done else 0.0
    gain_l2r = l2r_mean - bm25_mean
    if USE_CROSS_ENCODER_RERANK:
        gain_ce = ce_mean - bm25_mean
        print(
            f"BM25 ({EVAL_METRIC}@{EVAL_CUTOFF})={bm25_mean:.4f} | "
            f"L2R ({EVAL_METRIC}@{EVAL_CUTOFF})={l2r_mean:.4f} | "
            f"CE ({EVAL_METRIC}@{EVAL_CUTOFF})={ce_mean:.4f} | "
            f"Δ(L2R−BM25)={gain_l2r:+.4f} | Δ(CE−BM25)={gain_ce:+.4f}",
        )
    else:
        print(
            f"BM25 ({EVAL_METRIC}@{EVAL_CUTOFF})={bm25_mean:.4f} | "
            f"L2R ({EVAL_METRIC}@{EVAL_CUTOFF})={l2r_mean:.4f} | gain={gain_l2r:+.4f}",
        )

    row: Dict[str, Any] = {
        "train_limit": TRAIN_QUERY_LIMIT,
        "eval_limit": EVAL_QUERY_LIMIT,
        "train_qids_sampled": len(train_qid_queries),
        "eval_qids_sampled": n_eval_total,
        "K": K,
        "metric": EVAL_METRIC,
        "cutoff": EVAL_CUTOFF,
        "embedding": USE_EMBEDDING_FEATURES,
        "cross_encoder_rerank": USE_CROSS_ENCODER_RERANK,
        "ce_rerank_k": CROSS_ENCODER_RERANK_K if USE_CROSS_ENCODER_RERANK else None,
        "xgb_params": str(XGB_PARAMS),
        "bm25": bm25_mean,
        "l2r": l2r_mean,
        "gain": gain_l2r,
    }
    if USE_CROSS_ENCODER_RERANK:
        row["ce_rerank"] = ce_mean
        row["gain_ce"] = gain_ce
    RESULTS.append(row)

# ================================
# RESULTS
# ================================
results_df = pd.DataFrame(RESULTS)

print("\nFINAL RESULTS:")
print(
    results_df.sort_values(
        ["train_limit", "eval_limit", "K", "l2r"],
        ascending=[True, True, True, False],
    )
)