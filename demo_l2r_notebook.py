# %% [markdown]
# ## Minimal MS MARCO Learning-to-Rank (L2R) with Elasticsearch + XGBoost
# 
# This notebook is a minimal working prototype:
# 
# - BM25 retrieval via Elasticsearch (candidates)
# - Basic feature extraction
# - Train a Learning-to-Rank model (`xgboost.XGBRanker`)
# - Rerank the top results
# - Evaluate with NDCG@10
# 
# Assumptions:
# - Elasticsearch index: `msmarco` with fields `pid`, `text`
# - Local files: `queries.tsv`, `qrels.train.tsv`
# - ES config in `.env.local`: `ES_LOCAL_URL`, `ES_LOCAL_API_KEY`

# %% [markdown]
# ## Step 1: Setup and Imports
# 
# Import libraries, load `.env.local`, initialize Elasticsearch client, and define helper functions.

# %%
import os
import re
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split  # imported per requirements (not used)
from xgboost import XGBRanker


def load_dotenv_like(path: str) -> Dict[str, str]:
    """Minimal .env loader (supports ${VAR} expansion)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing env file: {path}")

    env = dict(os.environ)
    var_ref = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def expand(value: str) -> str:
        def repl(m):
            k = m.group(1)
            return env.get(k, os.environ.get(k, ""))
        return var_ref.sub(repl, value)

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            env[k] = expand(v)
    return env


ENV = load_dotenv_like(".env.local")

ES_URL = ENV.get("ES_LOCAL_URL")
ES_API_KEY = ENV.get("ES_LOCAL_API_KEY")
ES_INDEX = ENV.get("ES_INDEX", "msmarco")

if not ES_URL:
    raise ValueError("ES_LOCAL_URL not found in .env.local")
if not ES_API_KEY:
    raise ValueError("ES_LOCAL_API_KEY not found in .env.local")

es = Elasticsearch(ES_URL, api_key=ES_API_KEY)


def tokenize(text: str) -> List[str]:
    return str(text).lower().split()


def snippet(text: str, n: int = 180) -> str:
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s[:n] + ("…" if len(s) > n else "")


def bm25_search(query: str, k: int = 10, index: str = None) -> List[Dict]:
    index = index or ES_INDEX
    q = {"match": {"passage": query}}
    try:
        resp = es.search(index=index, query=q, size=k)
    except TypeError:
        # fallback for older client versions
        resp = es.search(index=index, body={"query": q}, size=k)

    hits = resp.get("hits", {}).get("hits", [])
    out = []
    for h in hits:
        src = h.get("_source", {})
        pid = src.get("pid", h.get("_id"))
        out.append(
            {
                "pid": str(pid),
                "passage": src.get("passage", ""),
                "bm25_score": float(h.get("_score", 0.0)),
            }
        )
    return out


print({"ES_URL": ES_URL, "ES_INDEX": ES_INDEX, "api_key_loaded": bool(ES_API_KEY)})
# 1) Ping + basic info
print("Ping:", es.ping())
print("Info:", es.info()["version"]["number"])

# 2) Index exists?
exists = es.indices.exists(index=ES_INDEX)
print("Index exists:", exists)

# 3) Count docs (quick check that data is there)
if exists:
    try:
        cnt = es.count(index=ES_INDEX)["count"]
        print("Doc count:", cnt)
    except Exception as e:
        print("Count failed:", e)


# %% [markdown]
# ## Step 2: Load Data
# 
# Load `queries.tsv` and `qrels.train.tsv` and show basic stats.

# %%
queries = pd.read_csv(
    "queries.train.tsv",
    sep="\t",
    names=["qid", "query"],
    dtype={"qid": str, "query": str},
)

qrels = pd.read_csv(
    "qrels.train.tsv",
    sep="\t",
    names=["qid", "unused", "pid", "relevance"],
    dtype={"qid": str, "pid": str, "relevance": int},
)
qrels = qrels.drop(columns=["unused"])

display(queries.head())
print("#queries rows:", len(queries), "unique qids:", queries["qid"].nunique())
print("#qrels rows:", len(qrels), "unique qids:", qrels["qid"].nunique())
print("qrels relevance distribution:")
print(qrels["relevance"].value_counts().sort_index())

# %% [markdown]
# ## Step 3: Sample Queries
# 
# Sample ~50 queries that have labels (appear in qrels).

# %%
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

qids_with_labels = set(qrels["qid"].unique())
queries_labeled = queries[queries["qid"].isin(qids_with_labels)].copy()

sample_size = min(500, len(queries_labeled))
sampled_queries = queries_labeled.sample(n=sample_size, random_state=SEED).reset_index(drop=True)

display(sampled_queries.head(10))
print("sampled qids:", sampled_queries["qid"].nunique())

# %% [markdown]
# ## Step 4: BM25 Retrieval Demo (visual)
# 
# For 3–5 queries, show BM25 top-10 (pid + snippet + score).

# %%
demo_n = min(5, len(sampled_queries))
demo_rows = sampled_queries.sample(n=demo_n, random_state=SEED)

for _, row in demo_rows.iterrows():
    qid, qtext = row["qid"], row["query"]
    print("\n" + "=" * 80)
    print(f"QID {qid}: {qtext}")
    hits = bm25_search(qtext, k=10)
    for rank, h in enumerate(hits, start=1):
        print(f"  {rank:2d}. pid={h['pid']}  bm25={h['bm25_score']:.4f}  text=" + snippet(h["passage"]))

# %% [markdown]
# ## Step 5: Build Candidate Set
# 
# For each sampled query, retrieve top-100 BM25 docs and store `(qid, pid, bm25_score, text)`.

# %%
candidate_k = 75
candidate_rows = []

for _, row in sampled_queries.iterrows():
    qid, qtext = row["qid"], row["query"]
    hits = bm25_search(qtext, k=candidate_k)
    for h in hits:
        candidate_rows.append(
            {
                "qid": qid,
                "query": qtext,
                "pid": str(h["pid"]),
                "bm25_score": float(h["bm25_score"]),
                "passage": h["passage"],
            }
        )

candidates = pd.DataFrame(candidate_rows)
print(
    "candidates rows:",
    len(candidates),
    "unique qids:",
    candidates["qid"].nunique(),
    "unique pids:",
    candidates["pid"].nunique(),
)
display(candidates.head())

# %% [markdown]
# ## Step 6: Label Construction
# 
# Label = 1 if `(qid, pid)` is relevant (`relevance > 0`), else 0. Show class balance.

# %%
qrels_sampled = qrels[qrels["qid"].isin(sampled_queries["qid"])].copy()

relevant_pairs = set(
    zip(
        qrels_sampled.loc[qrels_sampled["relevance"] > 0, "qid"],
        qrels_sampled.loc[qrels_sampled["relevance"] > 0, "pid"],
    )
)

candidates["label"] = [
    1 if (qid, pid) in relevant_pairs else 0
    for qid, pid in zip(candidates["qid"], candidates["pid"])
]

print("label balance:")
print(candidates["label"].value_counts())

# %% [markdown]
# ## Step 7: Feature Engineering (basic)
# 
# Features per (query, doc):
# - BM25 score
# - Term overlap count
# - Query coverage (fraction of query terms present)
# - Document length

# %%
# --- Precompute query tokens ---
qid_to_qtokens = {
    qid: tokenize(qtext)
    for qid, qtext in sampled_queries[["qid", "query"]].itertuples(index=False, name=None)
}

# --- Helpers ---
def bigrams(tokens):
    return list(zip(tokens, tokens[1:]))

def min_pairwise_distance(pos_a, pos_b):
    i = j = 0
    best = None
    while i < len(pos_a) and j < len(pos_b):
        a, b = pos_a[i], pos_b[j]
        d = abs(a - b)
        best = d if best is None else min(best, d)
        if a < b:
            i += 1
        else:
            j += 1
    return best

# --- Feature function ---
def featurize_row(qid: str, doc_text: str, bm25_score: float):
    q_tokens = qid_to_qtokens.get(qid, [])
    d_tokens = tokenize(doc_text)
    d_set = set(d_tokens)
    d_text = " ".join(d_tokens)

    # --- Basic features ---
    overlap = sum(1 for t in q_tokens if t in d_set)
    coverage = overlap / max(1, len(q_tokens))
    doc_len = len(d_tokens)

    # --- Phrase match ---
    q_phrase = " ".join(q_tokens)
    phrase_match = int(q_phrase in d_text) if q_phrase else 0

    # --- Proximity (FIXED: use inverse, not raw distance) ---
    q_unique = list(dict.fromkeys(q_tokens))
    if len(q_unique) < 2:
        prox_inv = 0.0
    else:
        positions = {}
        for i, tok in enumerate(d_tokens):
            if tok in q_unique:
                positions.setdefault(tok, []).append(i)

        if any(t not in positions for t in q_unique):
            prox_inv = 0.0
        else:
            best = None
            for i in range(len(q_unique)):
                for j in range(i + 1, len(q_unique)):
                    d = min_pairwise_distance(
                        positions[q_unique[i]],
                        positions[q_unique[j]],
                    )
                    best = d if best is None else min(best, d)

            prox_inv = 1.0 / (1.0 + best) if best is not None else 0.0

    # --- Bigram match (FIXED: normalized) ---
    q_bigrams = bigrams(q_tokens)
    if q_bigrams:
        d_bigrams_set = set(bigrams(d_tokens))
        matches = sum(1 for bg in q_bigrams if bg in d_bigrams_set)
        bigram_score = matches / len(q_bigrams)
    else:
        bigram_score = 0.0

    return (
        float(bm25_score),
        float(overlap),
        float(coverage),
        float(doc_len),
        float(phrase_match),
        float(prox_inv),
        float(bigram_score),
    )

# --- Feature columns ---
feat_cols = [
    "bm25_score",
    "term_overlap_count",
    "query_coverage",
    "doc_length",
    "phrase_match",
    "proximity_inv",
    "bigram_score",
]

# --- Compute features ---
features = np.vstack(
    [
        featurize_row(qid, txt, score)
        for qid, txt, score in zip(
            candidates["qid"],
            candidates["passage"],
            candidates["bm25_score"],
        )
    ]
)

for i, c in enumerate(feat_cols):
    candidates[c] = features[:, i]

# --- Preview ---
display(candidates[["qid", "pid"] + feat_cols + ["label"]].head())

# %% [markdown]
# ## Step 8: Prepare Training Data
# 
# Sort by `qid` and build `X`, `y`, and `group` (docs per query).

# %%
candidates_sorted = candidates.sort_values(["qid", "bm25_score"], ascending=[True, False]).reset_index(drop=True)

X = candidates_sorted[feat_cols].to_numpy(dtype=np.float32)
y = candidates_sorted["label"].to_numpy(dtype=np.float32)

group = candidates_sorted.groupby("qid").size().to_list()

print(
    "X shape:", X.shape,
    "y shape:", y.shape,
    "#groups:", len(group),
    "min/max group size:", min(group), max(group),
)

# %% [markdown]
# ## Step 9: Train L2R Model
# 
# Train an `XGBRanker` with a fast configuration.

# %%
ranker = XGBRanker(
    objective="rank:ndcg",
    n_estimators=60,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=SEED,
    n_jobs=4,
)

ranker.fit(X, y, group=group)

print("Trained XGBRanker on", len(group), "queries and", X.shape[0], "(query,doc) pairs")

try:
    fi = ranker.feature_importances_
    for name, val in sorted(zip(feat_cols, fi), key=lambda x: -x[1]):
        print(f"  {name}: {val:.4f}")
except Exception as e:
    print("(feature importance unavailable)", e)

# %% [markdown]
# ## Step 10: Reranking Demo
# 
# For a few queries, compare BM25 top-10 vs L2R reranked top-10.

# %%
rerank_demo_n = min(3, sampled_queries["qid"].nunique())
demo_qids = sampled_queries["qid"].drop_duplicates().sample(n=rerank_demo_n, random_state=SEED).to_list()

for qid in demo_qids:
    qtext = sampled_queries.loc[sampled_queries["qid"] == qid, "query"].iloc[0]
    dfq = candidates_sorted[candidates_sorted["qid"] == qid].head(10).copy()

    Xq = dfq[feat_cols].to_numpy(dtype=np.float32)
    dfq["pred_score"] = ranker.predict(Xq)

    bm25_view = dfq[["pid", "bm25_score", "pred_score", "label", "passage"]].copy()
    bm25_view["snippet"] = bm25_view["passage"].map(snippet)
    bm25_view = bm25_view.drop(columns=["passage"]).reset_index(drop=True)

    reranked_view = bm25_view.sort_values("pred_score", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"QID {qid}: {qtext}")
    print("\nBM25 top-10:")
    display(bm25_view[["pid", "bm25_score", "pred_score", "label", "snippet"]])
    print("\nL2R reranked top-10:")
    display(reranked_view[["pid", "bm25_score", "pred_score", "label", "snippet"]])

# %% [markdown]
# ## Step 11: Basic Evaluation (NDCG@10)
# 
# Compute mean NDCG@10 for BM25 vs L2R rerank over the sampled queries.

# %%
def dcg_at_k(rels: List[float], k: int = 10) -> float:
    rels = rels[:k]
    return float(sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(rels))) if rels else 0.0

def ndcg_at_k(rels: List[float], k: int = 10) -> float:
    dcg = dcg_at_k(rels, k=k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k=k)
    return 0.0 if ideal == 0 else float(dcg / ideal)

bm25_scores = []
l2r_scores = []

for qid, dfq in candidates_sorted.groupby("qid"):
    top = dfq.head(10).copy()

    rels_bm25 = top["label"].astype(float).to_list()

    Xq = top[feat_cols].to_numpy(dtype=np.float32)
    preds = ranker.predict(Xq)
    rels_l2r = top.assign(pred=preds).sort_values("pred", ascending=False)["label"].astype(float).to_list()

    bm25_scores.append(ndcg_at_k(rels_bm25, k=10))
    l2r_scores.append(ndcg_at_k(rels_l2r, k=10))

print(f"Mean NDCG@10 (BM25): {np.mean(bm25_scores):.4f}")
print(f"Mean NDCG@10 (L2R rerank): {np.mean(l2r_scores):.4f}")

# %%


# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


