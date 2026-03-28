# ================================
# FULL EXPERIMENT SCRIPT (ES + L2R)
# ================================
import os
import re
import random
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

QUERY_SIZES = [5000]
CANDIDATE_KS = [100]
XGB_PARAM_GRID = [
    {
        "name": "baseline",
        "params": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
    },
]

# Evaluation: "ndcg" (NDCG@EVAL_CUTOFF) or "mrr" (MRR@EVAL_CUTOFF, first relevant in top-k).
EVAL_METRIC = "mrr"
EVAL_CUTOFF = 10
if EVAL_METRIC not in ("ndcg", "mrr"):
    raise ValueError("EVAL_METRIC must be 'ndcg' or 'mrr'")

# If True: only use queries that have at least one relevant doc in qrels, backfill until
# quota with queries that have >=1 relevant in top-K, drop qids with no positive labels, etc.
# If False (default): use queries as sampled without those relevance-based filters.
FILTER_QUERIES_WITH_RELEVANT_DOCS = False

# Option C: optional dense embedding features (CPU-friendly small model).
# Requires: pip install sentence-transformers torch
# (CPU-only PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cpu
#  then pip install sentence-transformers)
USE_EMBEDDING_FEATURES = False
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 64  # raise if you have RAM headroom

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
ES_INDEX = "msmarco_v2"

es = Elasticsearch(ES_URL, api_key=ES_API_KEY)

if USE_EMBEDDING_FEATURES:
    print("Embedding features: ON |", EMBEDDING_MODEL_NAME, "| batch_size=", EMBEDDING_BATCH_SIZE)
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


def compute_embedding_features(
    df: pd.DataFrame,
    qid_to_query_text: Dict[str, str],
    model_name: str,
    batch_size: int,
    seed: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Encode each unique query and passage once, then add per-row cosine similarity
    and L2 distance between L2-normalized embeddings (CPU-friendly; no per-row model call).
    """
    from sentence_transformers import SentenceTransformer

    np.random.seed(seed)
    model = SentenceTransformer(model_name, device="cpu")

    uqids = df["qid"].unique().tolist()
    q_texts = [qid_to_query_text[q] for q in uqids]
    qid_to_row = {q: i for i, q in enumerate(uqids)}

    pid_first = df.drop_duplicates(subset=["pid"], keep="first")
    upids = pid_first["pid"].tolist()
    p_texts = pid_first["passage"].tolist()
    pid_to_row = {p: i for i, p in enumerate(upids)}

    print(
        f"Embedding: model={model_name} | unique queries={len(uqids)} | unique passages={len(upids)}"
    )

    q_emb = model.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    p_emb = model.encode(
        p_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    qi = np.array([qid_to_row[q] for q in df["qid"].to_numpy()], dtype=np.int64)
    pi = np.array([pid_to_row[p] for p in df["pid"].to_numpy()], dtype=np.int64)
    qv = q_emb[qi]
    pv = p_emb[pi]
    emb_cos = np.sum(qv * pv, axis=1)
    emb_l2 = np.linalg.norm(qv - pv, axis=1)
    emb_stack = np.column_stack([emb_cos, emb_l2]).astype(np.float64)
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

        feats = np.vstack([
            featurize(qid, txt, bm)
            for qid, txt, bm in zip(df["qid"], df["passage"], df["bm25"])
        ])

        df[feat_cols] = feats

        if USE_EMBEDDING_FEATURES:
            emb_arr, emb_names = compute_embedding_features(
                df,
                qid_to_query_text,
                EMBEDDING_MODEL_NAME,
                EMBEDDING_BATCH_SIZE,
                SEED,
            )
            df[emb_names] = emb_arr
            feat_cols = feat_cols + emb_names

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
        scaler = StandardScaler()
        train_df = train_df.copy()
        test_df = test_df.copy()
        scaler.fit(train_df[feat_cols])
        train_df[feat_cols] = scaler.transform(train_df[feat_cols])
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])

        # ------------------------
        # PREP TRAIN
        # ------------------------
        X_train = train_df[feat_cols].to_numpy(np.float32)
        y_train = train_df["label"].to_numpy(np.float32)
        group_train = train_df.groupby("qid").size().to_list()

        # ------------------------
        # EVALUATE BASELINE BM25 ONCE
        # ------------------------
        bm25_scores = []
        for _, group_df in test_df.groupby("qid"):
            bm25_ranked = group_df.sort_values("bm25", ascending=False).head(K)
            rels_bm25 = bm25_ranked["label"].head(EVAL_CUTOFF).tolist()
            bm25_scores.append(eval_query_metric(rels_bm25, EVAL_METRIC, EVAL_CUTOFF))
        bm25_mean = float(np.mean(bm25_scores))

        best_result = None
        print(
            f"BM25 ({EVAL_METRIC}@{EVAL_CUTOFF})={bm25_mean:.4f} | "
            f"testing {len(XGB_PARAM_GRID)} XGB configs..."
        )

        for config in XGB_PARAM_GRID:
            # ------------------------
            # TRAIN
            # ------------------------
            model = XGBRanker(
                objective="rank:ndcg",
                random_state=SEED,
                n_jobs=4,
                **config["params"],
            )
            model.fit(X_train, y_train, group=group_train)

            # ------------------------
            # EVALUATE
            # ------------------------
            l2r_scores = []
            for _, group_df in test_df.groupby("qid"):
                bm25_ranked = group_df.sort_values("bm25", ascending=False).head(K)
                Xq = bm25_ranked[feat_cols].to_numpy(np.float32)
                preds = model.predict(Xq)
                l2r_ranked = bm25_ranked.assign(pred=preds).sort_values("pred", ascending=False)
                rels_l2r = l2r_ranked["label"].head(EVAL_CUTOFF).tolist()
                l2r_scores.append(eval_query_metric(rels_l2r, EVAL_METRIC, EVAL_CUTOFF))

            l2r_mean = float(np.mean(l2r_scores))
            gain = l2r_mean - bm25_mean
            print(
                f"  [{config['name']}] L2R ({EVAL_METRIC}@{EVAL_CUTOFF})={l2r_mean:.4f} | "
                f"gain={gain:+.4f}"
            )

            run_result = {
                "queries": qsize,
                "K": K,
                "metric": EVAL_METRIC,
                "cutoff": EVAL_CUTOFF,
                "embedding": USE_EMBEDDING_FEATURES,
                "config": config["name"],
                "xgb_params": str(config["params"]),
                "bm25": bm25_mean,
                "l2r": l2r_mean,
                "gain": gain,
            }
            RESULTS.append(run_result)

            if best_result is None or run_result["l2r"] > best_result["l2r"]:
                best_result = run_result

        if best_result is not None:
            print(
                f"Best config: {best_result['config']} | "
                f"L2R ({EVAL_METRIC}@{EVAL_CUTOFF})={best_result['l2r']:.4f} | "
                f"gain={best_result['gain']:+.4f}"
            )

# ================================
# RESULTS
# ================================
results_df = pd.DataFrame(RESULTS)

print("\nFINAL RESULTS:")
print(results_df.sort_values(["queries", "K", "l2r"], ascending=[True, True, False]))