import os
import random
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from elasticsearch import Elasticsearch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from l2r_features import (
    L2RConfig,
    bm25_search,
    featurize_candidates,
    load_dotenv_like,
    load_ranker_artifacts,
    tokenize,
)


def _snippet(text: str, n: int = 220) -> str:
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s[:n] + ("…" if len(s) > n else "")


class AppState:
    def __init__(self) -> None:
        self.es: Optional[Elasticsearch] = None
        self.cfg: Optional[L2RConfig] = None
        self.rank_model: Any = None
        self.scaler: Any = None
        self.feat_cols: List[str] = []
        self.scale_cols: List[str] = []
        self.passage_emb_cache: Dict[str, Any] = {}
        self.st_model: Any = None
        self.ce_model: Any = None

        self.qid_to_query: Dict[str, str] = {}
        self.qid_to_rels: Dict[str, Dict[str, int]] = {}
        self.eligible_qids: List[str] = []


STATE = AppState()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _load_dev_qrels_and_queries(queries_dev_path: str, qrels_dev_path: str) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]], List[str]]:
    queries_dev = pd.read_csv(
        queries_dev_path,
        sep="\t",
        names=["qid", "query"],
        dtype=str,
    )
    qrels_dev = pd.read_csv(
        qrels_dev_path,
        sep="\t",
        names=["qid", "unused", "pid", "rel"],
        dtype=str,
    ).drop(columns=["unused"])
    qrels_dev["rel"] = qrels_dev["rel"].astype(int)
    qrels_dev = qrels_dev[qrels_dev["rel"] > 0].copy()

    qid_to_query = {str(q): str(t) for q, t in zip(queries_dev["qid"], queries_dev["query"])}
    qid_to_rels: Dict[str, Dict[str, int]] = {}
    for qid, pid, rel in zip(qrels_dev["qid"], qrels_dev["pid"], qrels_dev["rel"]):
        qid = str(qid)
        pid = str(pid)
        qid_to_rels.setdefault(qid, {})[pid] = int(rel)

    eligible = [qid for qid in qid_to_rels.keys() if qid in qid_to_query]
    return qid_to_query, qid_to_rels, eligible


def _init_models_and_state() -> None:
    env = load_dotenv_like(os.environ.get("DOTENV_PATH", ".env.local"))

    es_url = env.get("ES_LOCAL_URL")
    es_api_key = env.get("ES_LOCAL_API_KEY")
    if not es_url or not es_api_key:
        raise RuntimeError("Missing ES_LOCAL_URL / ES_LOCAL_API_KEY (expected in .env.local).")

    cfg = L2RConfig(
        seed=_env_int("SEED", 42),
        es_index=os.environ.get("ES_INDEX", "msmarco"),
        es_passage_field=os.environ.get("ES_PASSAGE_FIELD", "passage"),
        use_embedding_features=_env_bool("USE_EMBEDDING_FEATURES", True),
        embedding_model_name=os.environ.get("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        embedding_batch_size=_env_int("EMBEDDING_BATCH_SIZE", 128),
        es_mget_batch_size=_env_int("ES_MGET_BATCH_SIZE", 500),
        es_passage_emb_index=os.environ.get("ES_PASSAGE_EMB_INDEX") or None,
        es_passage_emb_field=os.environ.get("ES_PASSAGE_EMB_FIELD", "embedding"),
        use_prox_features=_env_bool("USE_PROX_FEATURES", True),
    ).with_defaults()

    es = Elasticsearch(es_url, api_key=es_api_key)

    artifact_dir = os.environ.get("L2R_ARTIFACT_DIR", "runs/l2r_api_artifacts")
    if not os.path.exists(os.path.join(artifact_dir, "xgb_ranker.json")):
        raise RuntimeError(
            f"Missing ranker artifacts under {artifact_dir!r}. "
            "Expected xgb_ranker.json, scaler.pkl, features.pkl. "
            "Run: python3 train_l2r_artifacts.py --out-dir runs/l2r_api_artifacts"
        )

    model, scaler, feat_cols, scale_cols = load_ranker_artifacts(artifact_dir)
    st_model = None

    if _env_bool("USE_CROSS_ENCODER_RERANK", True):
        from sentence_transformers import CrossEncoder

        ce_model = CrossEncoder(
            os.environ.get("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            device=os.environ.get("EMBEDDING_DEVICE", "cuda"),
        )
    else:
        ce_model = None

    qid_to_query, qid_to_rels, eligible_qids = _load_dev_qrels_and_queries(
        os.environ.get("QUERIES_DEV_PATH", "queries.dev.tsv"),
        os.environ.get("QRELS_DEV_PATH", "qrels.dev.tsv"),
    )
    if not eligible_qids:
        raise RuntimeError("No eligible dev qids found (need qrels.dev rel>0 intersect queries.dev).")

    STATE.es = es
    STATE.cfg = cfg
    STATE.rank_model = model
    STATE.scaler = scaler
    STATE.feat_cols = feat_cols
    STATE.scale_cols = scale_cols
    STATE.st_model = st_model
    STATE.ce_model = ce_model
    STATE.qid_to_query = qid_to_query
    STATE.qid_to_rels = qid_to_rels
    STATE.eligible_qids = eligible_qids


@asynccontextmanager
async def lifespan(_: FastAPI):
    _init_models_and_state()
    yield


app = FastAPI(title="L2R + BM25 + Cross-encoder demo API", lifespan=lifespan)
_cors_origins_raw = os.environ.get("CORS_ALLOW_ORIGINS", "http://localhost,http://localhost:3040,http://127.0.0.1:3000")
_cors_allow_origins = [o.strip() for o in _cors_origins_raw.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    if STATE.es is None or STATE.cfg is None:
        raise HTTPException(status_code=503, detail="Not initialized")
    return {
        "ok": True,
        "es_ping": bool(STATE.es.ping()),
        "es_index": STATE.cfg.es_index,
        "embedding_features": bool(STATE.cfg.use_embedding_features),
        "cross_encoder": STATE.ce_model is not None,
        "eligible_dev_qids": len(STATE.eligible_qids),
        "artifact_dir": os.environ.get("L2R_ARTIFACT_DIR"),
    }


@app.get("/sample")
def sample(
    l2r_candidate_k: int = Query(1000, ge=1, le=5000),
    bm25_return_k: int = Query(1000, ge=1, le=1000),
    l2r_return_k: int = Query(1000, ge=1, le=5000),
    seed: Optional[int] = None,
    ce_rerank_k: int = Query(100, ge=1, le=1000),
    ce_return_k: int = Query(100, ge=1, le=1000),
) -> Dict[str, Any]:
    if STATE.es is None or STATE.cfg is None:
        raise HTTPException(status_code=503, detail="Not initialized")

    hard_return_cap = _env_int("API_MAX_RETURN_K", 1000)
    bm25_return_k = min(int(bm25_return_k), hard_return_cap)
    l2r_return_k = min(int(l2r_return_k), hard_return_cap)
    ce_return_k = min(int(ce_return_k), hard_return_cap)

    rng = random.Random(seed) if seed is not None else random
    qid = rng.choice(STATE.eligible_qids)
    qtext = STATE.qid_to_query[qid]

    hits = bm25_search(
        STATE.es,
        STATE.cfg.es_index,
        qtext,
        l2r_candidate_k,
        passage_field=STATE.cfg.es_passage_field,
    )
    if not hits:
        raise HTTPException(status_code=503, detail="BM25 returned no hits for sampled qid; try again.")

    rows = [
        {"qid": qid, "pid": pid, STATE.cfg.es_passage_field: passage, "bm25": float(score)}
        for pid, passage, score in hits
    ]
    df = pd.DataFrame(rows)
    qid_to_tokens = {qid: tokenize(qtext)}
    qid_to_query_text = {qid: qtext}

    df, emb_names, st_model = featurize_candidates(
        df,
        qid_to_query_text=qid_to_query_text,
        qid_to_tokens=qid_to_tokens,
        cfg=STATE.cfg,
        es=STATE.es,
        passage_emb_cache=STATE.passage_emb_cache,
        st_model=STATE.st_model,
        emb_names=None,
    )
    STATE.st_model = st_model

    df[STATE.scale_cols] = STATE.scaler.transform(df[STATE.scale_cols].to_numpy(dtype="float64"))
    Xq = df[STATE.feat_cols].to_numpy("float32")
    preds = STATE.rank_model.predict(Xq)

    df_bm25 = df.sort_values("bm25", ascending=False).reset_index(drop=True)
    df_l2r = df.assign(l2r_score=preds).sort_values("l2r_score", ascending=False).reset_index(drop=True)

    bm25_list = [
        {
            "rank": i + 1,
            "pid": str(r["pid"]),
            "bm25": float(r["bm25"]),
            "snippet": _snippet(r[STATE.cfg.es_passage_field]),
        }
        for i, r in df_bm25.head(bm25_return_k).iterrows()
    ]
    l2r_list = [
        {
            "rank": i + 1,
            "pid": str(r["pid"]),
            "l2r_score": float(r["l2r_score"]),
            "snippet": _snippet(r[STATE.cfg.es_passage_field]),
        }
        for i, r in df_l2r.head(l2r_return_k).iterrows()
    ]

    ce_final: List[Dict[str, Any]] = []
    if STATE.ce_model is not None:
        rk = min(int(ce_rerank_k), len(df_l2r))
        top = df_l2r.head(rk).copy()
        pairs = [[qtext, str(p)] for p in top[STATE.cfg.es_passage_field].tolist()]
        ce_scores = STATE.ce_model.predict(
            pairs,
            batch_size=_env_int("CROSS_ENCODER_BATCH_SIZE", 64),
            show_progress_bar=False,
        )
        top["ce_score"] = ce_scores
        final = top.sort_values("ce_score", ascending=False).head(ce_return_k).reset_index(drop=True)
        for i, r in final.iterrows():
            ce_final.append(
                {
                    "rank": i + 1,
                    "pid": str(r["pid"]),
                    "ce_score": float(r["ce_score"]),
                    "snippet": _snippet(r[STATE.cfg.es_passage_field]),
                }
            )
    else:
        ce_final = []

    rels_map = STATE.qid_to_rels.get(qid, {})
    relevant_pids = sorted(rels_map.keys())

    in_pool = [pid for pid in relevant_pids if pid in set(df["pid"].astype(str).tolist())]

    return {
        "qid": qid,
        "query": qtext,
        "config": {
            "l2r_candidate_k": int(l2r_candidate_k),
            "bm25_return_k": int(bm25_return_k),
            "l2r_return_k": int(l2r_return_k),
            "embedding_features": bool(STATE.cfg.use_embedding_features),
            "cross_encoder_rerank": STATE.ce_model is not None,
            "ce_rerank_k": int(ce_rerank_k),
            "ce_return_k": int(ce_return_k),
            "feature_cols": list(STATE.feat_cols),
            "emb_names": list(emb_names),
        },
        "relevant_pids": relevant_pids,
        "relevance": rels_map,
        "relevant_in_pool": in_pool,
        "bm25": bm25_list,
        "l2r": l2r_list,
        "cross_encoder": ce_final,
    }

