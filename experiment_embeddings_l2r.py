
# ================================
# FULL EXPERIMENT SCRIPT (ES + L2R)
# ================================
import os
import re
import random
import time
from typing import Dict, List, Tuple

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

QUERY_SIZES = [10000]
CANDIDATE_KS = [100]

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
}

# Evaluation: "ndcg" (NDCG@EVAL_CUTOFF) or "mrr" (MRR@EVAL_CUTOFF, first relevant in top-k).
EVAL_METRIC = "mrr"
EVAL_CUTOFF = 10
if EVAL_METRIC not in ("ndcg", "mrr"):
    raise ValueError("EVAL_METRIC must be 'ndcg' or 'mrr'")

# If True: only use queries that have at least one relevant doc in qrels, backfill until
# quota with queries that have >=1 relevant in top-K, drop qids with no positive labels, etc.
# If False (default): use queries as sampled without those relevance-based filters.
FILTER_QUERIES_WITH_RELEVANT_DOCS = False

# Option C: optional dense embedding features.
# Passage vectors: read from Elasticsearch (dense_vector / numeric array in _source).
# Query vectors: encoded locally — use the same model that produced the index embeddings.
# Requires: pip install sentence-transformers torch (queries only)
USE_EMBEDDING_FEATURES = True
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64  # query encode batch size
# Passage vectors: same index as BM25 unless you override (e.g. split indices).
# Your msmarco mapping: dense_vector "embedding", dims=384, cosine — matches all-MiniLM-L6-v2.
ES_EMBEDDING_INDEX = None  # None → ES_INDEX
ES_PASSAGE_EMBEDDING_FIELD = "embedding"
ES_MGET_BATCH_SIZE = 5000

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

if USE_EMBEDDING_FEATURES:
    _emb_idx = ES_EMBEDDING_INDEX or ES_INDEX
    print(
        "Embedding features: ON | queries:",
        EMBEDDING_MODEL_NAME,
        "| passages from ES index",
        repr(_emb_idx),
        "field",
        repr(ES_PASSAGE_EMBEDDING_FIELD),
    )
else:
    print("Embedding features: OFF (set USE_EMBEDDING_FEATURES = True for Option C)")

print(f"Eval metric: {EVAL_METRIC.upper()}@{EVAL_CUTOFF}")

print("Ping:", es.ping())
print("Index exists:", es.indices.exists(index=ES_INDEX))

# ================================
# HELPERS
# ================================
def tokenize(x):
    return str(x).lower().split()

def bm25_search(query, k):
    resp = es.search(
        index=ES_INDEX,
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


def _log(msg: str) -> None:
    print(msg, flush=True)


def fetch_passage_embeddings_from_es(
    client: Elasticsearch,
    index: str,
    pids: List[str],
    field: str,
    batch_size: int,
) -> Tuple[np.ndarray, int]:
    """mget passage vectors by document id (pid must match ES _id).

    Already batched: each loop iteration is **one** Elasticsearch ``mget`` request for up to
    ``batch_size`` ids (not one HTTP call per passage).

    Returns a dense matrix (len(pids), dim) with zeros for missing docs, and a missing count.
    """
    n = len(pids)
    if n == 0:
        raise ValueError("fetch_passage_embeddings_from_es: empty pids")
    n_batches = (n + batch_size - 1) // batch_size
    t0 = time.perf_counter()
    _log(
        f"  [ES] mget batching: {n_batches} request(s), up to {batch_size} passage ids per request "
        f"({n} unique passages total) → index {index!r} field {field!r}"
    )
    p_emb: np.ndarray | None = None
    dim: int | None = None
    missing = 0
    for b_idx, i in enumerate(range(0, n, batch_size), start=1):
        batch = pids[i : i + batch_size]
        _log(f"  [ES] mget batch {b_idx}/{n_batches} (rows {i}..{i + len(batch) - 1}, {len(batch)} ids)...")
        t_batch = time.perf_counter()
        resp = client.mget(
            index=index,
            ids=batch,
            source_includes=[field],
        )
        for k, doc in enumerate(resp["docs"]):
            j = i + k
            if not doc.get("found"):
                missing += 1
                continue
            src = doc.get("_source") or {}
            raw = src.get(field)
            if raw is None:
                missing += 1
                continue
            v = np.asarray(raw, dtype=np.float64).ravel()
            if dim is None:
                dim = int(v.shape[0])
                p_emb = np.zeros((n, dim), dtype=np.float64)
            elif v.shape[0] != dim:
                raise ValueError(
                    f"Inconsistent embedding dim at row j={j}: got {v.shape[0]}, expected {dim}"
                )
            p_emb[j] = v
        _log(f"  [ES]   batch {b_idx}/{n_batches} done in {time.perf_counter() - t_batch:.2f}s (missing so far: {missing})")
    if dim is None or p_emb is None:
        raise RuntimeError(
            f"No passage embeddings found in {index!r} for field {field!r}; "
            "check ES_PASSAGE_EMBEDDING_FIELD and that _id matches pid from BM25 hits."
        )
    _log(f"  [ES] all mget batches finished in {time.perf_counter() - t0:.2f}s (missing={missing}/{n})")
    return p_emb, missing


def compute_embedding_features(
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
    """
    Encode each unique query once; load each unique passage vector from ES; then per-row
    cosine similarity and L2 distance on L2-normalized vectors.
    """
    from sentence_transformers import SentenceTransformer

    np.random.seed(seed)
    n_rows = len(df)
    _log("--- Embedding features: start ---")
    _log(f"  [embed] dataframe rows (q–passage pairs): {n_rows}")

    _log(f"  [embed] loading SentenceTransformer {model_name!r} on cuda (can take a bit the first time)...")
    t_load = time.perf_counter()
    model = SentenceTransformer(model_name, device="cuda")
    _log(f"  [embed] model loaded in {time.perf_counter() - t_load:.2f}s")

    uqids = df["qid"].unique().tolist()
    q_texts = [qid_to_query_text[q] for q in uqids]

    upids = pd.unique(df["pid"]).tolist()

    _log(
        f"  [embed] unique queries={len(uqids)} | unique passages={len(upids)} | "
        f"passage vectors via ES {passage_index!r}.{passage_embedding_field!r}"
    )

    _log(f"  [embed] encoding {len(q_texts)} unique queries (batch_size={batch_size})...")
    t_enc = time.perf_counter()
    q_emb = model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    _log(f"  [embed] query encode done in {time.perf_counter() - t_enc:.2f}s → shape {q_emb.shape}")

    _log("  [embed] fetching passage vectors from Elasticsearch (mget batches, see [ES] lines)...")
    p_emb, n_missing = fetch_passage_embeddings_from_es(
        es_client,
        passage_index,
        upids,
        passage_embedding_field,
        mget_batch_size,
    )
    if n_missing:
        _log(
            f"  [embed] Warning: {n_missing}/{len(upids)} passage ids missing or without field "
            f"{passage_embedding_field!r} in index {passage_index!r} (rows use zero vector)"
        )

    _log(f"  [embed] L2-normalizing passage matrix {p_emb.shape}...")
    t_norm = time.perf_counter()
    p_emb = _l2_normalize_rows(p_emb)
    _log(f"  [embed] L2 normalize done in {time.perf_counter() - t_norm:.2f}s")

    # factorize aligns with .unique() order for qid/pid (first-appearance order).
    _log(f"  [embed] aligning rows ({n_rows}) to query/passage indices + cosine / L2...")
    t_align = time.perf_counter()
    qi = pd.factorize(df["qid"], sort=False)[0].astype(np.int64)
    pi = pd.factorize(df["pid"], sort=False)[0].astype(np.int64)
    qv = q_emb[qi]
    pv = p_emb[pi]
    emb_cos = np.sum(qv * pv, axis=1)
    emb_l2 = np.linalg.norm(qv - pv, axis=1)
    emb_stack = np.column_stack([emb_cos, emb_l2]).astype(np.float64)
    _log(f"  [embed] pairwise features done in {time.perf_counter() - t_align:.2f}s → {emb_stack.shape}")
    _log("--- Embedding features: done ---")
    return emb_stack, ["emb_cos", "emb_l2"]


# ================================
# LOAD DATA
# ================================
queries = pd.read_csv(
    "queries.train.tsv",
    sep="\t",
    names=["qid", "query"],
    dtype=str,
)

qrels = pd.read_csv(
    "qrels.train.tsv",
    sep="\t",
    names=["qid", "unused", "pid", "rel"],
    dtype=str,
).drop(columns=["unused"])

qrels["rel"] = qrels["rel"].astype(int)

# ================================
# Optional: restrict to qids that appear in qrels (only when FILTER_QUERIES_WITH_RELEVANT_DOCS)
# ================================
if FILTER_QUERIES_WITH_RELEVANT_DOCS:
    valid_qids_global = set(qrels["qid"].unique())
    queries = queries[queries["qid"].isin(valid_qids_global)].copy()
    print("Filtered queries (in qrels):", len(queries))
else:
    print("Queries (no qrel filter):", len(queries))

# Precompute relevance set
rel_pairs = set(zip(qrels["qid"], qrels["pid"]))

RESULTS = []

# ================================
# EXPERIMENT LOOP
# ================================
for qsize in QUERY_SIZES:
    target_q = min(qsize, len(queries))
    shuffled_queries = queries.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    for K in CANDIDATE_KS:
        print(f"\nRunning: queries={qsize}, K={K}")

        rows = []
        kept_qids = []
        qid_to_tokens = {}
        qid_to_query_text: Dict[str, str] = {}

        # ------------------------
        # BUILD CANDIDATES (backfill until quota)
        # ------------------------
        _log(f"  [candidates] target {target_q} queries, top-{K} BM25 hits each (scanning shuffled query list)...")
        t_cand = time.perf_counter()
        for _, row in shuffled_queries.iterrows():
            if len(kept_qids) >= target_q:
                break

            qid = row["qid"]
            query_text = row["query"]
            if qid in qid_to_tokens:
                continue

            hits = bm25_search(row["query"], k=K)
            if not hits:
                continue

            if FILTER_QUERIES_WITH_RELEVANT_DOCS:
                has_relevant = any((qid, pid) in rel_pairs for pid, _, _ in hits)
                if not has_relevant:
                    continue

            qid_to_tokens[qid] = tokenize(query_text)
            qid_to_query_text[qid] = query_text
            kept_qids.append(qid)

            for pid, passage, score in hits:
                rows.append({
                    "qid": qid,
                    "pid": pid,
                    "passage": passage,
                    "bm25": score,
                })

            nk = len(kept_qids)
            if nk <= 3 or nk % 25 == 0 or nk == target_q:
                _log(
                    f"  [candidates] accepted query {nk}/{target_q} qid={qid!r} "
                    f"→ {len(rows)} candidate rows so far"
                )

        _log(
            f"  [candidates] finished in {time.perf_counter() - t_cand:.2f}s → "
            f"{len(kept_qids)} queries, {len(rows)} candidate rows"
        )

        if FILTER_QUERIES_WITH_RELEVANT_DOCS and len(kept_qids) < target_q:
            print(f"Skipped (could only find {len(kept_qids)}/{target_q} queries with >=1 relevant doc in top-{K})")
            continue

        df = pd.DataFrame(rows)

        # ------------------------
        # LABEL
        # ------------------------
        df["label"] = [
            1 if (qid, pid) in rel_pairs else 0
            for qid, pid in zip(df["qid"], df["pid"])
        ]

        # ------------------------
        # Optional: drop qids with no positive label in candidates
        # ------------------------
        if FILTER_QUERIES_WITH_RELEVANT_DOCS:
            pos_per_q = df.groupby("qid")["label"].sum()
            valid_qids = pos_per_q[pos_per_q > 0].index
            df = df[df["qid"].isin(valid_qids)]

            if df.empty:
                print("Skipped (no positives)")
                continue

            print(f"Queries kept (>=1 relevant in top-{K}): {df['qid'].nunique()} / target {target_q}")
        else:
            print(f"Queries in df: {df['qid'].nunique()} / sampled {target_q}")

        # ------------------------
        # FEATURES
        # ------------------------
        def featurize(qid, passage, bm25):
            q_tokens = qid_to_tokens[qid]
            d_tokens = tokenize(passage)
            d_set = set(d_tokens)

            overlap = sum(1 for t in q_tokens if t in d_set)
            coverage = overlap / max(1, len(q_tokens))
            doc_len = len(d_tokens)
            q_phrase = " ".join(q_tokens)
            d_text = " ".join(d_tokens)
            phrase = int(q_phrase in d_text)

            return [bm25, overlap, coverage, doc_len, phrase]

        feat_cols = ["bm25", "overlap", "coverage", "doc_len", "phrase"]

        _log(f"  [lexical] featurizing {len(df)} rows (overlap, coverage, len, phrase)...")
        t_lex = time.perf_counter()
        feats = np.vstack([
            featurize(qid, txt, bm)
            for qid, txt, bm in zip(df["qid"], df["passage"], df["bm25"])
        ])
        _log(f"  [lexical] done in {time.perf_counter() - t_lex:.2f}s")

        df[feat_cols] = feats

        if USE_EMBEDDING_FEATURES:
            emb_arr, emb_names = compute_embedding_features(
                df,
                qid_to_query_text,
                EMBEDDING_MODEL_NAME,
                EMBEDDING_BATCH_SIZE,
                SEED,
                es,
                ES_EMBEDDING_INDEX or ES_INDEX,
                ES_PASSAGE_EMBEDDING_FIELD,
                ES_MGET_BATCH_SIZE,
            )
            df[emb_names] = emb_arr
            feat_cols = feat_cols + emb_names
            # Raw cosine for "cosine-only" rerank (not scaled with L2R features).
            df["emb_cos_rerank"] = df["emb_cos"].astype(np.float64)

        # ------------------------
        # SORT
        # ------------------------
        df = df.sort_values(["qid", "bm25"], ascending=[True, False])

        # ------------------------
        # TRAIN / TEST SPLIT (by qid)
        # ------------------------
        all_qids = list(df["qid"].unique())
        random.shuffle(all_qids)

        split = int(0.8 * len(all_qids))
        train_qids = set(all_qids[:split])
        test_qids = set(all_qids[split:])

        train_df = df[df["qid"].isin(train_qids)]
        test_df = df[df["qid"].isin(test_qids)]

        if train_df.empty or test_df.empty:
            print("Skipped (bad split)")
            continue

        if FILTER_QUERIES_WITH_RELEVANT_DOCS:
            train_pos = train_df.groupby("qid")["label"].sum()
            test_pos = test_df.groupby("qid")["label"].sum()
            train_df = train_df[train_df["qid"].isin(train_pos[train_pos > 0].index)]
            test_df = test_df[test_df["qid"].isin(test_pos[test_pos > 0].index)]

            if train_df.empty or test_df.empty:
                print("Skipped (train/test lost positives after safety filter)")
                continue

        # Fit scaler on train only, then transform train/test separately (no leakage).
        _log(f"  [train] StandardScaler on {len(train_df)} train rows, {len(feat_cols)} features...")
        t_sc = time.perf_counter()
        scaler = StandardScaler()
        train_df = train_df.copy()
        test_df = test_df.copy()
        scaler.fit(train_df[feat_cols])
        train_df[feat_cols] = scaler.transform(train_df[feat_cols])
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])
        _log(f"  [train] scaler fit+transform in {time.perf_counter() - t_sc:.2f}s")

        # ------------------------
        # PREP TRAIN
        # ------------------------
        X_train = train_df[feat_cols].to_numpy(np.float32)
        y_train = train_df["label"].to_numpy(np.float32)
        group_train = train_df.groupby("qid").size().to_list()
        _log(
            f"  [train] X_train {X_train.shape} | {len(group_train)} groups (queries) | "
            f"y positives {int(y_train.sum())}/{len(y_train)}"
        )

        # ------------------------
        # EVAL: BM25
        # ------------------------
        _log(f"  [eval] baseline BM25 ranking on {test_df['qid'].nunique()} test queries...")
        bm25_scores = []
        for _, group_df in test_df.groupby("qid"):
            bm25_ranked = group_df.sort_values("bm25", ascending=False).head(K)
            rels_bm25 = bm25_ranked["label"].head(EVAL_CUTOFF).tolist()
            bm25_scores.append(eval_query_metric(rels_bm25, EVAL_METRIC, EVAL_CUTOFF))
        bm25_mean = float(np.mean(bm25_scores))

        # ------------------------
        # EVAL: cosine-only rerank (query–passage cosine, no XGB)
        # ------------------------
        cosine_mean = None
        if USE_EMBEDDING_FEATURES:
            _log("  [eval] cosine-only rerank (emb_cos_rerank) on test queries...")
            cosine_scores = []
            for _, group_df in test_df.groupby("qid"):
                cos_ranked = group_df.sort_values("emb_cos_rerank", ascending=False).head(K)
                rels_cos = cos_ranked["label"].head(EVAL_CUTOFF).tolist()
                cosine_scores.append(eval_query_metric(rels_cos, EVAL_METRIC, EVAL_CUTOFF))
            cosine_mean = float(np.mean(cosine_scores))

        # ------------------------
        # TRAIN + EVAL: L2R (XGB on full feature vector)
        # ------------------------
        _log(f"  [xgb] fitting XGBRanker ({XGB_PARAMS})...")
        t_xgb = time.perf_counter()
        model = XGBRanker(
            objective="rank:ndcg",
            random_state=SEED,
            n_jobs=24,
            **XGB_PARAMS,
        )
        model.fit(X_train, y_train, group=group_train)
        _log(f"  [xgb] fit done in {time.perf_counter() - t_xgb:.2f}s")

        _log("  [eval] L2R (XGB scores) reranking test queries...")
        l2r_scores = []
        for _, group_df in test_df.groupby("qid"):
            bm25_ranked = group_df.sort_values("bm25", ascending=False).head(K)
            Xq = bm25_ranked[feat_cols].to_numpy(np.float32)
            preds = model.predict(Xq)
            l2r_ranked = bm25_ranked.assign(pred=preds).sort_values("pred", ascending=False)
            rels_l2r = l2r_ranked["label"].head(EVAL_CUTOFF).tolist()
            l2r_scores.append(eval_query_metric(rels_l2r, EVAL_METRIC, EVAL_CUTOFF))
        l2r_mean = float(np.mean(l2r_scores))

        print(
            f"BM25 ({EVAL_METRIC}@{EVAL_CUTOFF})={bm25_mean:.4f}\n"
            + (
                f"Cosine-only rerank ({EVAL_METRIC}@{EVAL_CUTOFF})={cosine_mean:.4f} "
                f"| vs BM25 {cosine_mean - bm25_mean:+.4f}\n"
                if cosine_mean is not None
                else "Cosine-only rerank: N/A (USE_EMBEDDING_FEATURES=False)\n"
            )
            + f"L2R XGB ({EVAL_METRIC}@{EVAL_CUTOFF})={l2r_mean:.4f} | vs BM25 {l2r_mean - bm25_mean:+.4f}"
        )

        RESULTS.append(
            {
                "queries": qsize,
                "K": K,
                "metric": EVAL_METRIC,
                "cutoff": EVAL_CUTOFF,
                "embedding": USE_EMBEDDING_FEATURES,
                "bm25": bm25_mean,
                "cosine_rerank": cosine_mean,
                "l2r_xgb": l2r_mean,
                "l2r_vs_bm25": l2r_mean - bm25_mean,
            }
        )

# ================================
# RESULTS
# ================================
results_df = pd.DataFrame(RESULTS)

print("\nFINAL RESULTS:")
print(results_df.sort_values(["queries", "K"]))