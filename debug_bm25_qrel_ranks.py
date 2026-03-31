#!/usr/bin/env python3
"""
Debug: sample queries, BM25 top-K from Elasticsearch, measure where qrel passages rank.

Outputs under OUT_DIR (default runs/bm25_rank_debug):
  - per_query.csv          one row per sampled query
  - summary.txt            human-readable stats
  - hit_rates.csv          recall@k table
  - fig_hit_rates.png      bar chart of hit@k
  - fig_rank_cdf.png       CDF of best relevant rank (among hits in top-K)
  - fig_rank_hist.png      histogram of best rank (capped bucket for misses)
  - fig_reciprocal_rank.png histogram of 1/rank for found queries
"""
from __future__ import annotations

import os
import re
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch

# Optional progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


# ================================
# CONFIG
# ================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

N_QUERIES = 10_000
RETRIEVE_K = 1000
# Hit rates reported at these cutoffs (must be <= RETRIEVE_K)
HIT_CUTOFFS: Sequence[int] = (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000)

# Sample only qids that have >=1 qrel (recommended for "answer in top-k" stats)
SAMPLE_QIDS_WITH_QRELS_ONLY = False

QUERIES_PATH = "queries.train.tsv"
QRELS_PATH = "qrels.train.tsv"
ES_INDEX = "msmarco"
DOTENV_PATH = ".env.local"

# Parallel ES searches (set 1 to disable)
ES_SEARCH_WORKERS = 12

OUT_DIR = os.path.join("runs", "bm25_rank_debug")

# qrel relevance: MS MARCO passage uses 1 for relevant
MIN_REL = 1


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


def bm25_search(client: Elasticsearch, index: str, query: str, k: int) -> List[Tuple[str, float]]:
    resp = client.search(
        index=index,
        query={"match": {"passage": query}},
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
    """
    Returns (best_rank_1_indexed, n_relevant_in_list, n_relevant_total_known).
    best_rank is None if no relevant pid appears in hits.
    """
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
    for c in HIT_CUTOFFS:
        if c > RETRIEVE_K:
            raise ValueError(f"HIT_CUTOFFS contains {c} > RETRIEVE_K={RETRIEVE_K}")

    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, stamp)
    os.makedirs(run_dir, exist_ok=True)

    env = load_dotenv_like(DOTENV_PATH)
    es = Elasticsearch(env["ES_LOCAL_URL"], api_key=env["ES_LOCAL_API_KEY"])
    if not es.ping():
        raise RuntimeError("Elasticsearch ping failed; check .env.local and cluster.")

    print("Ping OK | index:", ES_INDEX, "| retrieve_k:", RETRIEVE_K, "| n_queries target:", N_QUERIES)

    queries_df = pd.read_csv(
        QUERIES_PATH,
        sep="\t",
        names=["qid", "query"],
        dtype=str,
    )
    qrels_df = pd.read_csv(
        QRELS_PATH,
        sep="\t",
        names=["qid", "unused", "pid", "rel"],
        dtype=str,
    ).drop(columns=["unused"])
    qrels_df["rel"] = qrels_df["rel"].astype(int)

    # qid -> set of relevant pids (rel >= MIN_REL)
    qid_to_rel: Dict[str, Set[str]] = defaultdict(set)
    for _, r in qrels_df[qrels_df["rel"] >= MIN_REL].iterrows():
        qid_to_rel[str(r["qid"])].add(str(r["pid"]))

    if SAMPLE_QIDS_WITH_QRELS_ONLY:
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
    n_take = min(N_QUERIES, len(pool))
    sampled = pool.sample(n=n_take, random_state=SEED).reset_index(drop=True)

    qid_list = sampled["qid"].tolist()
    qid_to_query = dict(zip(sampled["qid"], sampled["query"]))

    rows: List[dict] = []

    def work(qid: str) -> dict:
        qtext = qid_to_query[qid]
        hits = bm25_search(es, ES_INDEX, qtext, RETRIEVE_K)
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

    if ES_SEARCH_WORKERS <= 1:
        for qid in tqdm(qid_list, desc="BM25 search"):
            rows.append(work(qid))
    else:
        with ThreadPoolExecutor(max_workers=ES_SEARCH_WORKERS) as ex:
            futs = {ex.submit(work, qid): qid for qid in qid_list}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="BM25 search"):
                rows.append(fut.result())

    per_query = pd.DataFrame(rows).sort_values("qid").reset_index(drop=True)

    # --- Stats: only interpret best_rank where we had qrels
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
    lines.append(f"run_dir: {run_dir}")
    lines.append(f"queries file: {QUERIES_PATH} | qrels: {QRELS_PATH}")
    lines.append(f"sampled unique queries: {len(per_query)}")
    lines.append(f"queries with >=1 qrel in train qrels: {n_with}")
    lines.append(f"relevant doc appeared somewhere in top-{RETRIEVE_K}: {n_found} ({100 * n_found / max(1, n_with):.2f}%)")
    lines.append(f"no relevant pid in top-{RETRIEVE_K}: {n_not_found} ({100 * n_not_found / max(1, n_with):.2f}%)")
    lines.append("")

    if len(ranks):
        lines.append(f"best_rank among FOUND (n={len(ranks)}):")
        lines.append(f"  mean: {ranks.mean():.2f}")
        lines.append(f"  median: {ranks.median():.2f}")
        lines.append(f"  p90: {ranks.quantile(0.90):.2f}")
        lines.append(f"  p95: {ranks.quantile(0.95):.2f}")
        lines.append(f"  p99: {ranks.quantile(0.99):.2f}")
        lines.append("")

    # MRR: 0 if not in top-K
    rr = np.zeros(n_with, dtype=np.float64)
    for i, r in enumerate(with_qrels["best_rank"].tolist()):
        if _hit_at_k(r, RETRIEVE_K):
            rr[i] = 1.0 / int(r)
    mrr = float(rr.mean()) if n_with else 0.0
    lines.append(f"MRR (relevant in top-{RETRIEVE_K}, else 0): {mrr:.6f}")
    lines.append("")

    hit_table = []
    lines.append(f"Hit rate (at least one relevant in top-k) over {n_with} qrels-queries:")
    for k in HIT_CUTOFFS:
        ok = with_qrels["best_rank"].apply(lambda x: _hit_at_k(x, k))
        rate = float(ok.mean()) if n_with else 0.0
        hit_table.append({"k": k, "hit_rate": rate, "n_hit": int(ok.sum()), "n_queries": n_with})
        lines.append(f"  @{k:4d}: {rate * 100:6.2f}%  ({int(ok.sum())}/{n_with})")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(os.path.join(run_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    per_query.to_csv(os.path.join(run_dir, "per_query.csv"), index=False)
    pd.DataFrame(hit_table).to_csv(os.path.join(run_dir, "hit_rates.csv"), index=False)

    # --- Figures
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ht = pd.DataFrame(hit_table)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(k) for k in ht["k"]], ht["hit_rate"] * 100.0, color="steelblue", edgecolor="black", linewidth=0.4)
    ax.set_ylabel("Hit rate (%)")
    ax.set_xlabel("k (BM25 top-k)")
    ax.set_title(f"Fraction of qrel-queries with ≥1 relevant in BM25 top-k\n(n={n_with}, retrieve={RETRIEVE_K})")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "fig_hit_rates.png"), dpi=150)
    plt.close(fig)

    if len(ranks):
        sorted_r = np.sort(ranks.to_numpy())
        y = np.arange(1, len(sorted_r) + 1) / len(sorted_r)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.step(sorted_r, y, where="post", color="darkgreen")
        ax.set_xlabel("Best relevant rank (min over qrel pids)")
        ax.set_ylabel("CDF")
        ax.set_title("CDF of best relevant rank (queries with a hit in top-K)")
        ax.set_xlim(1, RETRIEVE_K)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_rank_cdf.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        bins = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000, RETRIEVE_K + 1]
        bins = sorted(set(b for b in bins if b <= RETRIEVE_K + 1))
        ax.hist(sorted_r, bins=bins, color="coral", edgecolor="black", linewidth=0.4)
        ax.set_xlabel("Best relevant rank")
        ax.set_ylabel("Count")
        ax.set_title("Histogram of best relevant rank (found queries)")
        ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_rank_hist.png"), dpi=150)
        plt.close(fig)

        recip = 1.0 / sorted_r
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(recip, bins=50, color="mediumpurple", edgecolor="black", linewidth=0.3)
        ax.set_xlabel("1 / best_rank")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of reciprocal rank (found only)")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_reciprocal_rank.png"), dpi=150)
        plt.close(fig)

    # Miss mass: show "effective rank" histogram including misses as RETRIEVE_K+1 bucket
    effective = []
    for _, r in with_qrels.iterrows():
        br = r["best_rank"]
        if br is None or (isinstance(br, float) and np.isnan(br)):
            effective.append(RETRIEVE_K + 1)
        else:
            effective.append(int(br))
    if effective:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            effective,
            bins=[1, 5, 10, 20, 50, 100, 200, 500, 1000, RETRIEVE_K + 2],
            color="gray",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_xlabel(f"Best rank (bucket {RETRIEVE_K + 1} = not in top-{RETRIEVE_K})")
        ax.set_ylabel("Count")
        ax.set_title("Best relevant rank including misses as rightmost bucket")
        fig.tight_layout()
        fig.savefig(os.path.join(run_dir, "fig_rank_with_misses.png"), dpi=150)
        plt.close(fig)

    print(f"\nWrote artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
