#!/usr/bin/env python3
"""
Elasticsearch BM25 baseline for MS MARCO-style passage ranking.

Mirrors the Anserini guide flow (batch retrieval + MRR@10), using the same
Elasticsearch setup: ``match`` on the ``passage`` field,``pid`` from ``_source`` (fallback ``_id``).

Anserini dev baseline uses queries filtered to qids that appear in qrels
(queries.dev.small.tsv). Set FILTER_QUERIES_TO_QRELS_ONLY = True to do the same
with full queries.dev.tsv + qrels.dev.tsv (intersection of qids).

Note: Lucene/Anserini tuned BM25 (k1=0.82, b=0.68) must be configured on the
Elasticsearch index mapping; this script does not change index settings. Default
ES BM25 will differ slightly from the Anserini leaderboard number (~0.187 MRR@10).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


ES_INDEX = "msmarco"
ES_PASSAGE_FIELD = "passage"
RETRIEVE_K = 1000
MRR_CUTOFF = 10
MIN_REL = 1
ES_SEARCH_WORKERS = 24


def bm25_search(
    client: Elasticsearch,
    index: str,
    field: str,
    query: str,
    k: int,
) -> List[Tuple[str, float]]:
    resp = client.search(
        index=index,
        query={"match": {field: query}},
        size=k,
    )
    out: List[Tuple[str, float]] = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source") or {}
        pid = str(src.get("pid", h["_id"]))
        out.append((pid, float(h["_score"])))
    return out


def mrr_at_cutoff(ranked_pids: List[str], relevant: Set[str], cutoff: int) -> float:
    for i, pid in enumerate(ranked_pids[:cutoff]):
        if pid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_pids: List[str], relevant: Set[str], k: int) -> float:
    top = set(ranked_pids[:k])
    return 1.0 if (top & relevant) else 0.0


def load_qrels(path: str, min_rel: int) -> Dict[str, Set[str]]:
    df = pd.read_csv(
        path,
        sep="\t",
        names=["qid", "unused", "pid", "rel"],
        dtype=str,
    ).drop(columns=["unused"])
    df["rel"] = df["rel"].astype(int)
    qid_to_rel: Dict[str, Set[str]] = defaultdict(set)
    for _, r in df[df["rel"] >= min_rel].iterrows():
        qid_to_rel[str(r["qid"])].add(str(r["pid"]))
    return dict(qid_to_rel)


def main() -> None:
    p = argparse.ArgumentParser(description="ES BM25 MS MARCO-style baseline (MRR@10).")
    p.add_argument(
        "--dotenv", default=".env.local", help="Path to .env.local-style file"
    )
    p.add_argument("--index", default=ES_INDEX, help="Elasticsearch index name")
    p.add_argument("--passage-field", default=ES_PASSAGE_FIELD, dest="passage_field")
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing queries.*.tsv and qrels.*.tsv",
    )
    p.add_argument(
        "--queries", default="queries.dev.tsv", help="TSV filename: qid, query"
    )
    p.add_argument(
        "--qrels", default="qrels.dev.tsv", help="MS MARCO qrels TSV filename"
    )
    p.add_argument(
        "--no-filter-qrels",
        action="store_true",
        help="Use all queries from the queries file (do not restrict to qids in qrels)",
    )
    p.add_argument("--retrieve-k", type=int, default=RETRIEVE_K)
    p.add_argument("--mrr-cutoff", type=int, default=MRR_CUTOFF)
    p.add_argument("--min-rel", type=int, default=MIN_REL)
    p.add_argument("--workers", type=int, default=ES_SEARCH_WORKERS)
    p.add_argument(
        "--out-run",
        default=None,
        help="Write msmarco run (qid \\t pid \\t rank). Default: no file",
    )
    args = p.parse_args()

    if args.mrr_cutoff > args.retrieve_k:
        raise SystemExit(
            f"--mrr-cutoff ({args.mrr_cutoff}) cannot exceed --retrieve-k ({args.retrieve_k})"
        )

    load_dotenv(args.dotenv)
    es_url = os.environ.get("ES_LOCAL_URL")
    es_api_key = os.environ.get("ES_LOCAL_API_KEY")
    if not es_url or not es_api_key:
        raise SystemExit("Missing ES_LOCAL_URL / ES_LOCAL_API_KEY in environment.")

    es = Elasticsearch(es_url, api_key=es_api_key)
    if not es.ping():
        raise SystemExit("Elasticsearch ping failed; check .env.local and cluster.")

    queries_path = os.path.join(args.data_dir, args.queries)
    qrels_path = os.path.join(args.data_dir, args.qrels)

    qid_to_rel = load_qrels(qrels_path, args.min_rel)
    queries_df = pd.read_csv(
        queries_path,
        sep="\t",
        names=["qid", "query"],
        dtype=str,
    ).drop_duplicates(subset=["qid"], keep="first")

    if not args.no_filter_qrels:
        before = len(queries_df)
        queries_df = queries_df[queries_df["qid"].isin(qid_to_rel.keys())].copy()
        print(
            f"Filtered queries to qids in qrels: {len(queries_df)} / {before} rows "
            f"(unique qids with qrels in file: {len(qid_to_rel)})"
        )
    else:
        print(f"Queries (no qrel filter): {len(queries_df)} unique qids")

    qrows = list(queries_df.itertuples(index=False))
    n_q = len(qrows)
    print(
        f"Index: {args.index!r} | field: {args.passage_field!r} | "
        f"retrieve_k={args.retrieve_k} | MRR@{args.mrr_cutoff} | workers={args.workers} | queries={n_q}"
    )

    results: List[Tuple[str, List[str], Set[str]]] = []

    t0 = time.perf_counter()

    def work(row: Tuple[str, str]) -> Tuple[str, List[str], Set[str]]:
        qid, qtext = row
        hits = bm25_search(es, args.index, args.passage_field, qtext, args.retrieve_k)
        pids = [pid for pid, _ in hits]
        return qid, pids, qid_to_rel.get(qid, set())

    if args.workers <= 1:
        for row in tqdm(qrows, desc="BM25 search"):
            results.append(work(tuple(row)))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, tuple(r)): r[0] for r in qrows}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="BM25 search"):
                results.append(fut.result())

    qid_order = {str(r[0]): i for i, r in enumerate(qrows)}
    results.sort(key=lambda t: qid_order[str(t[0])])

    run_lines: List[str] = []
    for qid, pids, _rel in results:
        for rank, pid in enumerate(pids, start=1):
            run_lines.append(f"{qid}\t{pid}\t{rank}\n")

    elapsed = time.perf_counter() - t0

    mrr_scores: List[float] = []
    recall_scores: List[float] = []
    n_with_qrels = 0
    for qid, pids, rel in results:
        if not rel:
            continue
        n_with_qrels += 1
        mrr_scores.append(mrr_at_cutoff(pids, rel, args.mrr_cutoff))
        recall_scores.append(recall_at_k(pids, rel, min(1000, args.retrieve_k)))

    mrr_mean = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    recall_mean = float(np.mean(recall_scores)) if recall_scores else 0.0

    print()
    print("#####################")
    print(f"MRR @{args.mrr_cutoff}: {mrr_mean}")
    print(f"QueriesRanked: {n_q}")
    print(f"Queries with >=1 qrel (evaluated for MRR): {n_with_qrels}")
    if recall_scores:
        print(
            f"Recall @ min(1000, retrieve_k) (any relevant in top-k): {recall_mean:.6f}"
        )
    print(f"Elapsed: {elapsed:.1f}s ({n_q / max(elapsed, 1e-9):.2f} q/s)")
    print("#####################")

    if args.out_run:
        os.makedirs(os.path.dirname(args.out_run) or ".", exist_ok=True)
        with open(args.out_run, "w") as f:
            f.writelines(run_lines)
        print(f"Wrote run file: {args.out_run} ({len(run_lines)} lines)")


if __name__ == "__main__":
    main()
