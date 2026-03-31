import math
import os
import pickle
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRanker


def load_dotenv_like(path: str) -> Dict[str, str]:
    env = dict(os.environ)
    var_ref = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

    def expand(value: str) -> str:
        return var_ref.sub(lambda m: env.get(m.group(1), ""), value)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" not in line or line.startswith("#"):
                continue
            k, v = line.strip().split("=", 1)
            env[k.strip()] = expand(v.strip().strip('"'))
    return env


def tokenize(x: str) -> List[str]:
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
    d = d_tokens[:max_doc]
    U = _distinct_query_terms(q_tokens)
    if not U:
        return 0.0
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
    d = d_tokens[:max_doc]
    k = len(q_tokens)
    if k == 0:
        return 0.0
    n = len(d)
    INF = n + k + 10
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
    d_tokens: List[str],
    use_prox_features: bool,
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

    tri_overlap = float(len(tri_q & tri_d))
    bi_union = len(bi_q | bi_d)
    bi_jacc = float(len((bi_q & bi_d)) / bi_union) if bi_union else 0.0

    prox_score = proximity_greedy_sequential(q_tokens, d_tokens)

    out = [
        coverage,
        doc_len_log,
        phrase_exact,
        tri_overlap,
        bi_jacc,
        prox_score,
    ]
    if use_prox_features:
        out.extend(
            [
                prox_min_window_unordered(q_tokens, d_tokens),
                prox_ordered_window_optimal(q_tokens, d_tokens),
                prox_pair_avg_distance(q_tokens, d_tokens),
                prox_cluster_density(q_tokens, d_tokens),
            ]
        )
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

PER_QUERY_COLS = ["bm25_norm", "inv_rank", "rank_pct"]


def add_per_query_bm25_rank_features(df: pd.DataFrame) -> None:
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


def bm25_search(
    client: Elasticsearch,
    index: str,
    query: str,
    k: int,
    passage_field: str = "passage",
) -> List[Tuple[str, str, float]]:
    resp = client.search(
        index=index,
        query={"match": {passage_field: query}},
        size=k,
    )
    out: List[Tuple[str, str, float]] = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source") or {}
        out.append(
            (
                str(src.get("pid", h.get("_id"))),
                str(src.get(passage_field, "")),
                float(h.get("_score", 0.0)),
            )
        )
    return out


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
    show_progress_bar: bool = False,
) -> Tuple[np.ndarray, List[str], Any]:
    from sentence_transformers import SentenceTransformer

    np.random.seed(seed)
    if st_model is None:
        st_model = SentenceTransformer(model_name, device=os.environ.get("EMBEDDING_DEVICE", "cuda"))

    uqids = df["qid"].unique().tolist()
    q_texts = [qid_to_query_text[q] for q in uqids]
    qid_to_row = {q: i for i, q in enumerate(uqids)}

    pid_first = df.drop_duplicates(subset=["pid"], keep="first")
    upids = pid_first["pid"].tolist()
    pid_to_row = {p: i for i, p in enumerate(upids)}

    cache = passage_vec_cache if passage_vec_cache is not None else {}
    need = [p for p in upids if p not in cache]
    if need:
        fetched, _ = fetch_passage_embeddings_from_es(
            es_client, passage_index, need, passage_field, mget_batch_size
        )
        cache.update(fetched)

    dim = None
    for p in upids:
        v = cache.get(p)
        if v is not None:
            dim = int(v.shape[0])
            break
    if dim is None:
        raise RuntimeError(
            f"No passage embeddings in index {passage_index!r} field {passage_field!r} "
            f"for {len(upids)} unique pids."
        )

    p_emb = np.zeros((len(upids), dim), dtype=np.float64)
    for j, p in enumerate(upids):
        v = cache.get(p)
        if v is not None and v.shape[0] == dim:
            p_emb[j] = v
        elif v is not None:
            raise ValueError(f"Inconsistent embedding dim for pid={p!r}")

    p_emb = _l2_normalize_rows(p_emb)

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
    return emb_stack, ["emb_cos", "emb_l2", "emb_dot"], st_model


@dataclass(frozen=True)
class L2RConfig:
    seed: int = 42
    es_index: str = "msmarco"
    es_passage_field: str = "passage"
    use_embedding_features: bool = True
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 128
    es_mget_batch_size: int = 500
    es_passage_emb_index: Optional[str] = None
    es_passage_emb_field: str = "embedding"
    use_prox_features: bool = True
    xgb_params: Dict[str, Any] = None  # set in __post_init__ style below

    def with_defaults(self) -> "L2RConfig":
        if self.xgb_params is not None:
            return self
        return L2RConfig(
            seed=self.seed,
            es_index=self.es_index,
            es_passage_field=self.es_passage_field,
            use_embedding_features=self.use_embedding_features,
            embedding_model_name=self.embedding_model_name,
            embedding_batch_size=self.embedding_batch_size,
            es_mget_batch_size=self.es_mget_batch_size,
            es_passage_emb_index=self.es_passage_emb_index,
            es_passage_emb_field=self.es_passage_emb_field,
            use_prox_features=self.use_prox_features,
            xgb_params={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
        )


def build_feature_columns(use_prox_features: bool, emb_names: Sequence[str]) -> Tuple[List[str], List[str], List[str]]:
    lex_cols = LEX_COLS_BASE + (PROX_COLS if use_prox_features else [])
    feat_cols = ["bm25"] + PER_QUERY_COLS + lex_cols + list(emb_names)
    scale_cols = [c for c in feat_cols if c != "bm25"]
    return lex_cols, feat_cols, scale_cols


def featurize_candidates(
    df: pd.DataFrame,
    qid_to_query_text: Dict[str, str],
    qid_to_tokens: Dict[str, List[str]],
    cfg: L2RConfig,
    es: Elasticsearch,
    passage_emb_cache: Optional[Dict[str, np.ndarray]] = None,
    st_model: Optional[Any] = None,
    emb_names: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str], Any]:
    cfg = cfg.with_defaults()

    def build_lexical_matrix(frame: pd.DataFrame) -> np.ndarray:
        return np.vstack(
            [
                lexical_features_row(
                    qid_to_tokens[str(row["qid"])],
                    tokenize(str(row[cfg.es_passage_field])),
                    cfg.use_prox_features,
                )
                for _, row in frame.iterrows()
            ]
        )

    lex_cols = LEX_COLS_BASE + (PROX_COLS if cfg.use_prox_features else [])
    df = df.copy()
    df[lex_cols] = build_lexical_matrix(df)
    add_per_query_bm25_rank_features(df)

    out_emb_names: List[str] = list(emb_names) if emb_names is not None else []
    if cfg.use_embedding_features:
        emb_arr, out_emb_names, st_model = compute_embedding_features(
            df,
            qid_to_query_text,
            cfg.embedding_model_name,
            cfg.embedding_batch_size,
            cfg.seed,
            es,
            cfg.es_passage_emb_index or cfg.es_index,
            cfg.es_passage_emb_field,
            cfg.es_mget_batch_size,
            passage_vec_cache=passage_emb_cache,
            st_model=st_model,
            show_progress_bar=False,
        )
        df[out_emb_names] = emb_arr

    return df, out_emb_names, st_model


def train_ranker_from_bm25_pools(
    es: Elasticsearch,
    cfg: L2RConfig,
    queries_train_path: str,
    qrels_train_path: str,
    candidate_k: int,
    train_query_limit: int,
    passage_emb_cache: Optional[Dict[str, np.ndarray]] = None,
    st_model: Optional[Any] = None,
) -> Tuple[XGBRanker, StandardScaler, List[str], List[str], Any]:
    cfg = cfg.with_defaults()

    queries_train_all = pd.read_csv(
        queries_train_path, sep="\t", names=["qid", "query"], dtype=str
    )
    qrels_train = pd.read_csv(
        qrels_train_path, sep="\t", names=["qid", "unused", "pid", "rel"], dtype=str
    ).drop(columns=["unused"])
    qrels_train["rel"] = qrels_train["rel"].astype(int)
    rel_pairs_train = set(zip(qrels_train["qid"], qrels_train["pid"]))

    train_qrel_qids = set(qrels_train[qrels_train["rel"] > 0]["qid"].unique())
    queries_train = queries_train_all[queries_train_all["qid"].isin(train_qrel_qids)].copy()

    rng = random.Random(cfg.seed)
    qrows = list(queries_train[["qid", "query"]].itertuples(index=False, name=None))
    rng.shuffle(qrows)
    qrows = qrows[: min(train_query_limit, len(qrows))]

    rows: List[dict] = []
    qid_to_tokens: Dict[str, List[str]] = {}
    qid_to_query_text: Dict[str, str] = {}
    for qid, qtext in qrows:
        hits = bm25_search(es, cfg.es_index, str(qtext), candidate_k, passage_field=cfg.es_passage_field)
        if not hits:
            continue
        qid_to_tokens[str(qid)] = tokenize(str(qtext))
        qid_to_query_text[str(qid)] = str(qtext)
        for pid, passage, score in hits:
            rows.append({"qid": str(qid), "pid": str(pid), cfg.es_passage_field: passage, "bm25": float(score)})

    if not rows:
        raise RuntimeError("No BM25 hits for any sampled training queries; cannot train ranker.")

    df = pd.DataFrame(rows)
    df["label"] = [
        1 if (qid, pid) in rel_pairs_train else 0 for qid, pid in zip(df["qid"], df["pid"])
    ]
    df = df[df.groupby("qid")["label"].transform("sum") > 0].copy()
    if df.empty:
        raise RuntimeError("Training pool has no positives after filtering; try larger candidate_k or limit.")

    df, emb_names, st_model = featurize_candidates(
        df,
        qid_to_query_text,
        qid_to_tokens,
        cfg,
        es,
        passage_emb_cache=passage_emb_cache,
        st_model=st_model,
    )

    _, feat_cols, scale_cols = build_feature_columns(cfg.use_prox_features, emb_names)
    scaler = StandardScaler()
    scaler.fit(df[scale_cols].to_numpy(dtype=np.float64))

    df = df.sort_values(["qid", "bm25"], ascending=[True, False]).copy()
    df[scale_cols] = scaler.transform(df[scale_cols].to_numpy(dtype=np.float64))
    X = df[feat_cols].to_numpy(np.float32)
    y = df["label"].to_numpy(np.float32)
    group = df.groupby("qid").size().to_list()

    model = XGBRanker(
        objective="rank:pairwise",
        random_state=cfg.seed,
        n_jobs=int(os.environ.get("XGB_N_JOBS", "4")),
        **cfg.xgb_params,
    )
    model.fit(X, y, group=group)

    return model, scaler, list(feat_cols), list(scale_cols), st_model


def train_ranker_from_bm25_pools_batched(
    es: Elasticsearch,
    cfg: L2RConfig,
    queries_train_path: str,
    qrels_train_path: str,
    candidate_k: int,
    train_query_limit: int,
    query_batch_size: int = 200,
    passage_emb_cache: Optional[Dict[str, np.ndarray]] = None,
    st_model: Optional[Any] = None,
    progress_every_batches: int = 1,
) -> Tuple[XGBRanker, StandardScaler, List[str], List[str], Any]:
    """
    Batchwise trainer to cap peak RAM:
    - fetch BM25 pools per query in batches
    - featurize each batch, store unscaled feature matrices
    - fit scaler at end (via partial_fit per batch)
    - transform per batch and then fit XGBRanker
    """
    cfg = cfg.with_defaults()
    if query_batch_size < 1:
        raise ValueError("query_batch_size must be >= 1")

    queries_train_all = pd.read_csv(
        queries_train_path, sep="\t", names=["qid", "query"], dtype=str
    )
    qrels_train = pd.read_csv(
        qrels_train_path, sep="\t", names=["qid", "unused", "pid", "rel"], dtype=str
    ).drop(columns=["unused"])
    qrels_train["rel"] = qrels_train["rel"].astype(int)

    rel_pairs_train = set(zip(qrels_train["qid"], qrels_train["pid"]))
    train_qrel_qids = set(qrels_train[qrels_train["rel"] > 0]["qid"].unique())
    queries_train = queries_train_all[queries_train_all["qid"].isin(train_qrel_qids)].copy()

    rng = random.Random(cfg.seed)
    qrows = list(queries_train[["qid", "query"]].itertuples(index=False, name=None))
    rng.shuffle(qrows)
    qrows = qrows[: min(train_query_limit, len(qrows))]

    n_q = len(qrows)
    n_batches = (n_q + query_batch_size - 1) // query_batch_size

    scaler = StandardScaler()
    emb_names: List[str] = []
    feat_cols: List[str] = []
    scale_cols: List[str] = []

    X_parts_raw: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    group_parts: List[List[int]] = []

    total_rows = 0
    total_qids_kept = 0

    for bi, start in enumerate(range(0, n_q, query_batch_size)):
        batch = qrows[start : start + query_batch_size]
        rows: List[dict] = []
        qid_to_tokens: Dict[str, List[str]] = {}
        qid_to_query_text: Dict[str, str] = {}

        for qid, qtext in batch:
            hits = bm25_search(
                es, cfg.es_index, str(qtext), candidate_k, passage_field=cfg.es_passage_field
            )
            if not hits:
                continue
            qid_s = str(qid)
            qtext_s = str(qtext)
            qid_to_tokens[qid_s] = tokenize(qtext_s)
            qid_to_query_text[qid_s] = qtext_s
            for pid, passage, score in hits:
                rows.append(
                    {
                        "qid": qid_s,
                        "pid": str(pid),
                        cfg.es_passage_field: passage,
                        "bm25": float(score),
                    }
                )

        if not rows:
            if progress_every_batches and ((bi + 1) % progress_every_batches == 0):
                print(f"[train] batch {bi+1}/{n_batches}: no BM25 rows")
            continue

        df = pd.DataFrame(rows)
        df["label"] = [
            1 if (qid, pid) in rel_pairs_train else 0
            for qid, pid in zip(df["qid"], df["pid"])
        ]
        df = df[df.groupby("qid")["label"].transform("sum") > 0].copy()
        if df.empty:
            if progress_every_batches and ((bi + 1) % progress_every_batches == 0):
                print(f"[train] batch {bi+1}/{n_batches}: 0 rows after positive filter")
            continue

        df, emb_names, st_model = featurize_candidates(
            df,
            qid_to_query_text=qid_to_query_text,
            qid_to_tokens=qid_to_tokens,
            cfg=cfg,
            es=es,
            passage_emb_cache=passage_emb_cache,
            st_model=st_model,
            emb_names=emb_names if emb_names else None,
        )

        if not feat_cols:
            _, feat_cols, scale_cols = build_feature_columns(cfg.use_prox_features, emb_names)

        df = df.sort_values(["qid", "bm25"], ascending=[True, False]).copy()
        group = df.groupby("qid").size().to_list()

        X_raw = df[feat_cols].to_numpy(dtype=np.float64, copy=True)
        y = df["label"].to_numpy(dtype=np.float32, copy=True)

        # Fit scaler incrementally (on scale cols only).
        scale_idx = [feat_cols.index(c) for c in scale_cols]
        scaler.partial_fit(X_raw[:, scale_idx])

        X_parts_raw.append(X_raw)
        y_parts.append(y)
        group_parts.append(group)

        total_rows += int(X_raw.shape[0])
        total_qids_kept += len(group)

        if progress_every_batches and ((bi + 1) % progress_every_batches == 0):
            print(
                f"[train] batch {bi+1}/{n_batches}: +{X_raw.shape[0]} rows "
                f"(cum_rows={total_rows}, cum_qids={total_qids_kept})"
            )

    if not X_parts_raw:
        raise RuntimeError("No training rows after BM25 + positive filter; cannot train ranker.")

    # Transform all batches and stack for XGBoost.
    X_parts_scaled: List[np.ndarray] = []
    group_train: List[int] = []
    scale_idx = [feat_cols.index(c) for c in scale_cols]
    for X_raw, y, group in zip(X_parts_raw, y_parts, group_parts):
        X_scaled = X_raw.astype(np.float64, copy=True)
        X_scaled[:, scale_idx] = scaler.transform(X_scaled[:, scale_idx])
        X_parts_scaled.append(X_scaled.astype(np.float32, copy=False))
        group_train.extend(group)

    X = np.vstack(X_parts_scaled)
    y_all = np.concatenate(y_parts)

    model = XGBRanker(
        objective="rank:pairwise",
        random_state=cfg.seed,
        n_jobs=int(os.environ.get("XGB_N_JOBS", "4")),
        **cfg.xgb_params,
    )
    print(f"[train] fitting XGBRanker on rows={X.shape[0]} qids={len(group_train)} …")
    model.fit(X, y_all, group=group_train)

    return model, scaler, list(feat_cols), list(scale_cols), st_model


def save_ranker_artifacts(
    out_dir: str,
    model: XGBRanker,
    scaler: StandardScaler,
    feat_cols: Sequence[str],
    scale_cols: Sequence[str],
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, "xgb_ranker.json"))
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(out_dir, "features.pkl"), "wb") as f:
        pickle.dump({"feat_cols": list(feat_cols), "scale_cols": list(scale_cols)}, f)


def load_ranker_artifacts(out_dir: str) -> Tuple[XGBRanker, StandardScaler, List[str], List[str]]:
    model_path = os.path.join(out_dir, "xgb_ranker.json")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    feat_path = os.path.join(out_dir, "features.pkl")
    model = XGBRanker()
    model.load_model(model_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(feat_path, "rb") as f:
        meta = pickle.load(f)
    return model, scaler, list(meta["feat_cols"]), list(meta["scale_cols"])

