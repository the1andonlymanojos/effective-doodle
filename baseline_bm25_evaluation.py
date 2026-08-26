#!/usr/bin/env python3
"""
Elasticsearch BM25 baseline for MS MARCO-style passage ranking.

Mirrors the Anserini guide flow (batch retrieval + MRR@10), using the same
Elasticsearch setup as demo_es_embed.py: ``match`` on the ``passage`` field,
``pid`` from ``_source`` (fallback ``_id``).

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
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple
import collections
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


# --- Defaults (override via CLI) ---
DOTENV_PATH = ".env.local"
ES_INDEX = "msmarco"
ES_PASSAGE_FIELD = "passage"
QUERIES_PATH = "queries.dev.tsv"
QRELS_PATH = "qrels.dev.tsv"
FILTER_QUERIES_TO_QRELS_ONLY = True
RETRIEVE_K = 1000
MRR_CUTOFF = 10
MIN_REL = 1
ES_SEARCH_WORKERS = 24
OUT_RUN_PATH: str | None = None  # e.g. "runs/es_bm25_dev.run.tsv"; None = do not write


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


def bm25_search(
    client: Elasticsearch,
    index: str,
    field: str,
    query: str,
    k: int,
    is_query_string: bool = False
) -> List[Tuple[str, float, str]]:
    q_body = {}
    if is_query_string:
        q_body = {"query_string": {"query": query, "default_field": field}}
    else:
        q_body = {"match": {field: query}}

    resp = client.search(
        index=index,
        query=q_body,
        size=k,
    )
    out: List[Tuple[str, float, str]] = []
    for h in resp["hits"]["hits"]:
        src = h.get("_source") or {}
        pid = str(src.get("pid", h["_id"]))
        out.append((pid, float(h["_score"]), str(src.get(field, ""))))
    return out


def build_rm3_query(
    original_query: str,
    hits: List[Tuple[str, float, str]],
    fb_docs: int,
    fb_terms: int,
    original_weight: float,
) -> str:
    top_hits = hits[:fb_docs]
    if not top_hits:
        return original_query

    scores = np.array([score for _, score, _ in top_hits])
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    p_D = exp_scores / np.sum(exp_scores)

    term_probs = collections.defaultdict(float)
    for i, (_, _, passage) in enumerate(top_hits):
        tokens = str(passage).lower().split()
        clean_tokens = []
        for t in tokens:
            t = "".join(c for c in t if c.isalnum())
            if t and t not in ENGLISH_STOP_WORDS:
                clean_tokens.append(t)

        if not clean_tokens:
            continue

        doc_len = len(clean_tokens)
        counts = collections.Counter(clean_tokens)
        for w, count in counts.items():
            prob_w_given_D = count / doc_len
            term_probs[w] += prob_w_given_D * p_D[i]

    sorted_terms = sorted(term_probs.items(), key=lambda x: x[1], reverse=True)
    top_terms = sorted_terms[:fb_terms]

    q_tokens = original_query.lower().split()
    # Ensure query tokens are clean too, to match terms
    q_tokens_clean = []
    for t in q_tokens:
        t = "".join(c for c in t if c.isalnum())
        if t: q_tokens_clean.append(t)
    
    q_counts = collections.Counter(q_tokens_clean)
    q_len = len(q_tokens_clean)

    final_terms_weight = {}
    for t, p_rm1 in top_terms:
        final_terms_weight[t] = (1 - original_weight) * p_rm1

    for t, count in q_counts.items():
        p_q = count / max(q_len, 1)
        if t in final_terms_weight:
            final_terms_weight[t] += original_weight * p_q
        else:
            final_terms_weight[t] = original_weight * p_q

    max_weight = max(final_terms_weight.values()) if final_terms_weight else 1.0
    query_parts = []
    for t, w in final_terms_weight.items():
        w_norm = w / max_weight
        if w_norm > 0.01:
            query_parts.append(f"{t}^{w_norm:.4f}")

    if not query_parts:
        return original_query

    return " ".join(query_parts)


def mrr_at_cutoff(ranked_pids: List[str], relevant: Set[str], cutoff: int) -> float:
    """MS MARCO MRR@k: reciprocal rank of the first qrel-relevant doc in positions 1..cutoff."""
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
    p.add_argument("--dotenv", default=DOTENV_PATH, help="Path to .env.local-style file")
    p.add_argument("--index", default=ES_INDEX, help="Elasticsearch index name")
    p.add_argument("--passage-field", default=ES_PASSAGE_FIELD, dest="passage_field")
    p.add_argument("--queries", default=QUERIES_PATH, help="TSV: qid, query")
    p.add_argument("--qrels", default=QRELS_PATH, help="MS MARCO qrels TSV")
    p.add_argument(
        "--no-filter-qrels",
        action="store_true",
        help="Use all queries from the queries file (do not restrict to qids in qrels)",
    )
    p.add_argument("--retrieve-k", type=int, default=RETRIEVE_K)
    p.add_argument("--mrr-cutoff", type=int, default=MRR_CUTOFF)
    p.add_argument("--min-rel", type=int, default=MIN_REL)
    p.add_argument("--workers", type=int, default=ES_SEARCH_WORKERS)
    p.add_argument("--rm3", action="store_true", help="Enable TopRM3 pseudo relevance feedback")
    p.add_argument("--rm3-fb-terms", type=int, default=10, help="Number of terms for RM3 expansion")
    p.add_argument("--rm3-fb-docs", type=int, default=10, help="Number of documents for RM3 feedback")
    p.add_argument("--rm3-original-weight", type=float, default=0.5, help="Original query weight for RM3 interpolation")
    p.add_argument(
        "--out-run",
        default=OUT_RUN_PATH,
        help="Write msmarco run (qid \\t pid \\t rank). Default: no file",
    )
    args = p.parse_args()

    if args.mrr_cutoff > args.retrieve_k:
        raise SystemExit(f"--mrr-cutoff ({args.mrr_cutoff}) cannot exceed --retrieve-k ({args.retrieve_k})")

    env = load_dotenv_like(args.dotenv)
    es = Elasticsearch(env["ES_LOCAL_URL"], api_key=env["ES_LOCAL_API_KEY"])
    if not es.ping():
        raise SystemExit("Elasticsearch ping failed; check .env.local and cluster.")

    qid_to_rel = load_qrels(args.qrels, args.min_rel)
    queries_df = pd.read_csv(
        args.queries,
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
        
        if args.rm3:
            expanded_query = build_rm3_query(
                original_query=qtext,
                hits=hits,
                fb_docs=args.rm3_fb_docs,
                fb_terms=args.rm3_fb_terms,
                original_weight=args.rm3_original_weight
            )
            hits = bm25_search(es, args.index, args.passage_field, expanded_query, args.retrieve_k, is_query_string=True)

        pids = [pid for pid, _, _ in hits]
        return qid, pids, qid_to_rel.get(qid, set())

    if args.workers <= 1:
        for row in tqdm(qrows, desc="BM25 search"):
            results.append(work(tuple(row)))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, tuple(r)): r[0] for r in qrows}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="BM25 search"):
                results.append(fut.result())

    # Deterministic run file order by original query list (parallel completion order is random)
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
        print(f"Recall @ min(1000, retrieve_k) (any relevant in top-k): {recall_mean:.6f}")
    print(f"Elapsed: {elapsed:.1f}s ({n_q / max(elapsed, 1e-9):.2f} q/s)")
    print("#####################")

    if args.out_run:
        os.makedirs(os.path.dirname(args.out_run) or ".", exist_ok=True)
        with open(args.out_run, "w") as f:
            f.writelines(run_lines)
        print(f"Wrote run file: {args.out_run} ({len(run_lines)} lines)")


if __name__ == "__main__":
    main()
