import argparse
import os

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from l2r.features import (
    L2RConfig,
    save_ranker_artifacts,
    train_ranker_from_bm25_pools_batched,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train XGBRanker + scaler artifacts for the demo API."
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Where to write xgb_ranker.json, scaler.pkl, features.pkl",
    )
    p.add_argument(
        "--candidate-k",
        type=int,
        default=int(os.environ.get("TRAIN_CANDIDATE_K", "100")),
    )
    p.add_argument(
        "--train-query-limit",
        type=int,
        default=int(os.environ.get("TRAIN_QUERY_LIMIT", "10000")),
    )
    p.add_argument(
        "--query-batch-size",
        type=int,
        default=int(os.environ.get("TRAIN_QUERY_BATCH_SIZE", "200")),
    )
    p.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "data"))
    args = p.parse_args()

    dotenv_path = os.environ.get("DOTENV_PATH", ".env.local")
    load_dotenv(dotenv_path)

    es_url = os.environ.get("ES_LOCAL_URL")
    es_api_key = os.environ.get("ES_LOCAL_API_KEY")
    if not es_url or not es_api_key:
        raise RuntimeError(
            "Missing ES_LOCAL_URL / ES_LOCAL_API_KEY (expected in .env.local)."
        )

    cfg = L2RConfig(
        seed=int(os.environ.get("SEED", "42")),
        es_index=os.environ.get("ES_INDEX", "msmarco"),
        es_passage_field=os.environ.get("ES_PASSAGE_FIELD", "passage"),
        use_embedding_features=os.environ.get("USE_EMBEDDING_FEATURES", "true")
        .strip()
        .lower()
        in ("1", "true", "yes", "y", "on"),
        embedding_model_name=os.environ.get(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        embedding_batch_size=int(os.environ.get("EMBEDDING_BATCH_SIZE", "128")),
        es_mget_batch_size=int(os.environ.get("ES_MGET_BATCH_SIZE", "500")),
        es_passage_emb_index=os.environ.get("ES_PASSAGE_EMB_INDEX") or None,
        es_passage_emb_field=os.environ.get("ES_PASSAGE_EMB_FIELD", "embedding"),
        use_prox_features=os.environ.get("USE_PROX_FEATURES", "true").strip().lower()
        in ("1", "true", "yes", "y", "on"),
    ).with_defaults()

    es = Elasticsearch(es_url, api_key=es_api_key)

    queries_train_path = os.path.join(args.data_dir, "queries.train.tsv")
    qrels_train_path = os.path.join(args.data_dir, "qrels.train.tsv")

    print(
        "[train] starting:",
        f"train_query_limit={args.train_query_limit}",
        f"candidate_k={args.candidate_k}",
        f"query_batch_size={args.query_batch_size}",
        f"embeddings={'on' if cfg.use_embedding_features else 'off'}",
        f"prox={'on' if cfg.use_prox_features else 'off'}",
        flush=True,
    )

    model, scaler, feat_cols, scale_cols, _ = train_ranker_from_bm25_pools_batched(
        es=es,
        cfg=cfg,
        queries_train_path=queries_train_path,
        qrels_train_path=qrels_train_path,
        candidate_k=args.candidate_k,
        train_query_limit=args.train_query_limit,
        query_batch_size=args.query_batch_size,
        passage_emb_cache={},
        st_model=None,
        progress_every_batches=1,
    )

    save_ranker_artifacts(args.out_dir, model, scaler, feat_cols, scale_cols)
    print(f"Wrote artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()
