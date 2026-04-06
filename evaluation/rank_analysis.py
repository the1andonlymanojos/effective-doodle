#!/usr/bin/env python3
"""
Debug: sample queries, BM25 top-K from Elasticsearch, measure where qrel passages rank.

Outputs under OUT_DIR (default runs/bm25_rank_debug):
  - per_query.csv          one row per sampled query
  - summary.txt            human-readable stats
  - hit_rates.csv          recall@k table
  - fig_hit_rates.png      bar chartof hit@k
  - fig_rank_cdf.png       CDF of best relevant rank (among hits in top-K)
  - fig_rank_hist.png      histogram of best rank (capped bucket for misses)
  - fig_reciprocal_rank.png histogram of 1/rank for found queries
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


def bm25_search(
    client: Elasticsearch,
    index: str,
    query: str,
    k: int,
    passage_field: str = "passage",
) -> List[Tuple[str, float]]:
    resp = client.search(
        index=index,
        query={"match": {passage_field: query}},
        size=k,
    )
    out: List[Tuple[str, float]] = []
    for h in resp["hits"]["hits"]:
        pid = str(h["_source"].get("pid", h["_id"]))
        out.append((pid, float(h["_score"])))
    return out


def best_relevant_rank(
    hits: Sequence[Tuple[str, float]],
    relevant_pids: Set[str],
) -> Tuple[Optional[int], int, int]:
    n_total = len(relevant_pids)
    best: Optional[int] = None
    n_in = 0
    for rank, (pid, _) in enumerate(hits, start=1):
        if pid in relevant_pids:
            n_in += 1
            if best is None or rank < best:
                best = rank
    return best, n_in, n_total


def main() -> None:
    p = argparse.ArgumentParser(
        description="Analyze BM25 ranking positions for qrel documents."
    )
    p.add_argument("--dotenv", default=".env.local", help="Path to .env file")
    p.add_argument("--index", default="msmarco", help="Elasticsearch index name")
    p.add_argument("--passage-field", default="passage", dest="passage_field")
    p.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing queries.*.tsv and qrels.*.tsv",
    )
    p.add_argument(
        "--queries", default="queries.train.tsv", help="Queries TSV filename"
    )
    p.add_argument("--qrels", default="qrels.train.tsv", help="Qrels TSV filename")
    p.add_argument(
        "--n-queries", type=int, default=10000, help="Number of queries to sample"
    )
    p.add_argument("--retrieve-k", type=int, default=1000, help="BM25 top-k retrieval")
    p.add_argument("--workers", type=int, default=12, help="Parallel ES search workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: runs/bm25_rank_debug/timestamp)",
    )
    p.add_argument(
        "--sample-qrels-only",
        action="store_true",
        help="Sample only queries with qrels",
    )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    hit_cutoffs = (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
    for c in hit_cutoffs:
        if c > args.retrieve_k:
            raise ValueError(f"hit cutoff {c} > retrieve_k={args.retrieve_k}")

    load_dotenv(args.dotenv)
    es_url = os.environ.get("ES_LOCAL_URL")
    es_api_key = os.environ.get("ES_LOCAL_API_KEY")
    if not es_url or not es_api_key:
        raise SystemExit("Missing ES_LOCAL_URL / ES_LOCAL_API_KEY in environment.")

    es = Elasticsearch(es_url, api_key=es_api_key)
    if not es.ping():
        raise RuntimeError("Elasticsearch ping failed; check .env.local and cluster.")

    queries_path = os.path.join(args.data_dir, args.queries)
    qrels_path = os.path.join(args.data_dir, args.qrels)

    print(
        "Ping OK | index:",
        args.index,
        "| retrieve_k:",
        args.retrieve_k,
        "| n_queries target:",
        args.n_queries,
    )

    queries_df = pd.read_csv(
        queries_path,
        sep="\t",
        names=["qid", "query"],
        dtype=str,
    )
    qrels_df = pd.read_csv(
        qrels_path,
        sep="\t",
        names=["qid", "unused", "pid", "rel"],
        dtype=str,
    ).drop(columns=["unused"])
    qrels_df["rel"] = qrels_df["rel"].astype(int)

    qid_to_rel: Dict[str, Set[str]] = defaultdict(set)
    for _, r in qrels_df[qrels_df["rel"] >= 1].iterrows():
        qid_to_rel[str(r["qid"])].add(str(r["pid"]))

    if args.sample_qrels_only:
        pool = queries_df[queries_df["qid"].isin(qid_to_rel.keys())].copy()
        print(
            "Sampling from queries with >=1 qrel:",
            len(pool),
            "rows (unique qids in pool:",
            pool["qid"].nunique(),
            ") | total qids with qrels:",
            len(qid_to_rel),
        )
    else:
        pool = queries_df.copy()
        print("Sampling from all queries:", len(pool))

    pool = pool.drop_duplicates(subset=["qid"], keep="first")
    n_take = min(args.n_queries, len(pool))
    sampled = pool.sample(n=n_take, random_state=args.seed).reset_index(drop=True)

    qid_list = sampled["qid"].tolist()
    qid_to_query = dict(zip(sampled["qid"], sampled["query"]))

    rows: List[dict] = []

    def work(qid: str) -> dict:
        qtext = qid_to_query[qid]
        hits = bm25_search(es, args.index, qtext, args.retrieve_k, args.passage_field)
        rel = qid_to_rel.get(qid, set())
        best, n_in_hits, n_rel_known = best_relevant_rank(hits, rel)
        return {
            "qid": qid,
            "n_hits": len(hits),
            "n_relevant_qrel": n_rel_known,
            "n_relevant_in_hits": n_in_hits,
            "best_rank": best,
            "found_in_top_k": best is not None,
        }

    if args.workers <= 1:
        for qid in tqdm(qid_list, desc="BM25 search"):
            rows.append(work(qid))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, qid): qid for qid in qid_list}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="BM25 search"):
                rows.append(fut.result())

    per_query = pd.DataFrame(rows).sort_values("qid").reset_index(drop=True)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", "bm25_rank_debug", stamp)
    os.makedirs(out_dir, exist_ok=True)

    with_qrels = per_query[per_query["n_relevant_qrel"] > 0].copy()
    n_with = len(with_qrels)
    n_found = int(with_qrels["found_in_top_k"].sum())
    n_not_found = n_with - n_found

    ranks = with_qrels.loc[with_qrels["best_rank"].notna(), "best_rank"].astype(int)

    def _hit_at_k(best, k: int) -> bool:
        if best is None or (isinstance(best, float) and np.isnan(best)):
            return False
        return int(best) <= k

    lines: List[str] = []
    lines.append("=== BM25 vs qrels (debug) ===")
    lines.append(f"out_dir: {out_dir}")
    lines.append(f"queries file: {queries_path} | qrels: {qrels_path}")
    lines.append(f"sampled unique queries: {len(per_query)}")
    lines.append(f"queries with >=1 qrel in train qrels: {n_with}")
    lines.append(
        f"relevant doc appeared somewhere in top-{args.retrieve_k}: {n_found} ({100 * n_found / max(1, n_with):.2f}%)"
    )
    lines.append(
        f"no relevant pid in top-{args.retrieve_k}: {n_not_found} ({100 * n_not_found / max(1, n_with):.2f}%)"
    )
    lines.append("")

    if len(ranks):
        lines.append(f"best_rank among FOUND (n={len(ranks)}):")
        lines.append(f"  mean: {ranks.mean():.2f}")
        lines.append(f"  median: {ranks.median():.2f}")
        lines.append(f"  p90: {ranks.quantile(0.90):.2f}")
        lines.append(f"  p95: {ranks.quantile(0.95):.2f}")
        lines.append(f"  p99: {ranks.quantile(0.99):.2f}")
        lines.append("")

    rr = np.zeros(n_with, dtype=np.float64)
    for i, r in enumerate(with_qrels["best_rank"].tolist()):
        if _hit_at_k(r, args.retrieve_k):
            rr[i] = 1.0 / int(r)
    mrr = float(rr.mean()) if n_with else 0.0
    lines.append(f"MRR (relevant in top-{args.retrieve_k}, else 0): {mrr:.6f}")
    lines.append("")

    hit_table = []
    lines.append(
        f"Hit rate (at least one relevant in top-k) over {n_with} qrels-queries:"
    )
    for k in hit_cutoffs:
        ok = with_qrels["best_rank"].apply(lambda x: _hit_at_k(x, k))
        rate = float(ok.mean()) if n_with else 0.0
        hit_table.append(
            {"k": k, "hit_rate": rate, "n_hit": int(ok.sum()), "n_queries": n_with}
        )
        lines.append(f"  @{k:4d}: {rate * 100:6.2f}%  ({int(ok.sum())}/{n_with})")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    per_query.to_csv(os.path.join(out_dir, "per_query.csv"), index=False)
    pd.DataFrame(hit_table).to_csv(os.path.join(out_dir, "hit_rates.csv"), index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ht = pd.DataFrame(hit_table)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        [str(k) for k in ht["k"]],
        ht["hit_rate"] * 100.0,
        color="steelblue",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_ylabel("Hit rate (%)")
    ax.set_xlabel("k (BM25 top-k)")
    ax.set_title(
        f"Fraction of qrel-queries with >=1 relevant in BM25 top-k\n(n={n_with}, retrieve={args.retrieve_k})"
    )
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_hit_rates.png"), dpi=150)
    plt.close(fig)

    if len(ranks):
        sorted_r = np.sort(ranks.to_numpy())
        y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(sorted_r, y, where="post", color="darkgreen")
        ax.set_xlabel("Best relevant rank (min over qrel pids)")
        ax.set_ylabel("CDF")
        ax.set_title("CDF of best relevant rank (queries with a hit in top-K)")
        ax.set_xlim(1, args.retrieve_k)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_rank_cdf.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, args.retrieve_k + 1]
        bins = sorted(set(b for b in bins if b <= args.retrieve_k + 1))
        ax.hist(sorted_r, bins=bins, color="coral", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Best relevant rank")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of best relevant rank (found queries)")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_rank_hist.png"), dpi=150)
        plt.close(fig)

        recip = 1.0 / sorted_r
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(recip, bins=50, color="mediumpurple", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("1 / best_rank")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of reciprocal rank (found only)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_reciprocal_rank.png"), dpi=150)
        plt.close(fig)

    effective = []
    for _, r in with_qrels.iterrows():
        br = r["best_rank"]
        if br is None or (isinstance(br, float) and np.isnan(br)):
            effective.append(args.retrieve_k + 1)
        else:
            effective.append(int(br))
    if effective:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            effective,
            bins=[1, 5, 10, 20, 50, 100, 200, 500, 1000, args.retrieve_k + 2],
            color="gray",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xlabel(
            f"Best rank (bucket {args.retrieve_k + 1} = not in top-{args.retrieve_k})"
        )
        ax.set_ylabel("Count")
        ax.set_title("Best relevant rank including misses as rightmost bucket")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_rank_with_misses.png"), dpi=150)
        plt.close(fig)

    print(f"\nWrote artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
