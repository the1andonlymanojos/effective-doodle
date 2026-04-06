"""
Shared utilities for L2R experiments.

Provides:
- Data loading (queries, qrels)
- BM25 search via Elasticsearch
- Evaluation metrics (MRR, NDCG)
- Feature engineering utilities
- Embedding helpers
"""

import math
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch


def load_env() -> Tuple[str, str]:
    """Load environment and return (ES_URL, ES_API_KEY)."""
    load_dotenv()
    es_url = os.environ.get("ES_LOCAL_URL")
    es_api_key = os.environ.get("ES_LOCAL_API_KEY")
    if not es_url or not es_api_key:
        raise RuntimeError("Missing ES_LOCAL_URL or ES_LOCAL_API_KEY in environment")
    return es_url, es_api_key


def get_elasticsearch_client() -> Elasticsearch:
    """Create Elasticsearch client from environment."""
    es_url, es_api_key = load_env()
    return Elasticsearch(es_url, api_key=es_api_key)


def load_queries(path: str) -> pd.DataFrame:
    """Load queries TSV (qid, query)."""
    return pd.read_csv(path, sep="\t", names=["qid", "query"], dtype=str)


def load_qrels(path: str) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
    """Load qrels TSV (qid, unused, pid, rel).

    Returns (qrels_df, qid_to_relevant_pids).
    """
    df = pd.read_csv(path, sep="\t", names=["qid", "unused", "pid", "rel"], dtype=str)
    df = df.drop(columns=["unused"])
    df["rel"] = df["rel"].astype(int)

    qid_to_pids: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df[df["rel"] > 0].iterrows():
        qid_to_pids[str(row["qid"])].add(str(row["pid"]))

    return df, dict(qid_to_pids)


def bm25_search(
    client: Elasticsearch,
    index: str,
    query: str,
    k: int,
    passage_field: str = "passage",
) -> List[Tuple[str, str, float]]:
    """Search Elasticsearch with BM25.

    Returns list of (pid, passage_text, score).
    """
    resp = client.search(
        index=index,
        query={"match": {passage_field: query}},
        size=k,
    )
    results = []
    for hit in resp["hits"]["hits"]:
        src = hit.get("_source") or {}
        pid = str(src.get("pid", hit["_id"]))
        passage = str(src.get(passage_field, ""))
        score = float(hit["_score"])
        results.append((pid, passage, score))
    return results


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer."""
    return str(text).lower().split()


def ndcg(rels: List[int], k: int = 10) -> float:
    """Compute NDCG@k."""
    rels = rels[:k]
    dcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sorted(rels, reverse=True)
    idcg = sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return 0.0 if idcg == 0 else dcg / idcg


def mrr_at_k(rels: List[int], k: int = 10) -> float:
    """Compute MRR@k (reciprocal rank of first relevant doc)."""
    for i, r in enumerate(rels[:k]):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


def eval_metric(rels: List[int], metric: str, k: int) -> float:
    """Evaluate using specified metric ('mrr' or 'ndcg')."""
    if metric == "ndcg":
        return ndcg(rels, k=k)
    if metric == "mrr":
        return mrr_at_k(rels, k=k)
    raise ValueError(f"Unknown metric: {metric!r}")


def ngrams(tokens: List[str], n: int) -> Set[Tuple[str, ...]]:
    """Extract n-grams from token list."""
    if n < 1 or len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def lexical_features(
    q_tokens: List[str],
    d_tokens: List[str],
    use_prox: bool = False,
) -> List[float]:
    """Compute lexical features for a query-document pair."""
    q_set = set(q_tokens)
    d_set = set(d_tokens)

    overlap = sum(1 for t in q_tokens if t in d_set)
    coverage = overlap / max(1, len(q_tokens))
    doc_len_log = math.log1p(len(d_tokens))

    q_joined = " ".join(q_tokens)
    d_joined = " ".join(d_tokens)
    phrase_exact = float(q_joined in d_joined)

    bi_q = ngrams(q_tokens, 2)
    bi_d = ngrams(d_tokens, 2)
    bi_union = len(bi_q | bi_d)
    bi_jacc = len(bi_q & bi_d) / bi_union if bi_union else 0.0

    tri_q = ngrams(q_tokens, 3)
    tri_d = ngrams(d_tokens, 3)
    tri_overlap = float(len(tri_q & tri_d))

    features = [coverage, doc_len_log, phrase_exact, tri_overlap, bi_jacc]

    if use_prox:
        prox = proximity_greedy(q_tokens, d_tokens)
        features.append(prox)

    return features


def add_per_query_bm25_features(df: pd.DataFrame) -> None:
    """Add per-query BM25 features (in-place): bm25_norm, inv_rank, rank_pct."""
    g = df.groupby("qid", sort=False)["bm25"]

    def _minmax(s):
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


def proximity_greedy(
    q_tokens: List[str], d_tokens: List[str], max_doc: int = 400
) -> float:
    """Greedy sequential proximity match."""
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


def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """L2 normalize rows of a matrix."""
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    return x / norm


def fetch_passage_embeddings(
    client: Elasticsearch,
    index: str,
    pids: List[str],
    field: str,
    batch_size: int = 500,
) -> Tuple[Dict[str, np.ndarray], int]:
    """Fetch passage embeddings from Elasticsearch via mget.

    Returns (pid_to_vec dict, missing_count).
    """
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
    query_texts: List[str],
    pids: List[str],
    pid_to_vec: Dict[str, np.ndarray],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[str]]:
    """Compute embedding features (cosine, l2, dot).

    Returns (features_array, feature_names).
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)

    q_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    dim = None
    for p in pids:
        if p in pid_to_vec:
            dim = pid_to_vec[p].shape[0]
            break

    if dim is None:
        raise RuntimeError("No passage embeddings found")

    p_emb = np.zeros((len(pids), dim), dtype=np.float64)
    for i, pid in enumerate(pids):
        if pid in pid_to_vec:
            v = pid_to_vec[pid]
            if v.shape[0] == dim:
                p_emb[i] = v

    p_emb = l2_normalize_rows(p_emb)

    cos_sim = np.sum(q_emb * p_emb, axis=1)
    l2_dist = np.linalg.norm(q_emb - p_emb, axis=1)
    dot_prod = np.sum(q_emb * p_emb, axis=1)

    features = np.column_stack([cos_sim, l2_dist, dot_prod])
    return features, ["emb_cos", "emb_l2", "emb_dot"]
