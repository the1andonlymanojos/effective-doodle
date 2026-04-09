import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
import numpy as np
import time

# ---------------- LOAD DATA ----------------
print("Loading collection...")
docs = []
pids = []

with open("collection.tsv", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        pid, text = line.strip().split("\t")
        pids.append(pid)
        docs.append(text)

        if i > 100000:  # limit
            break

print(f"Loaded {len(docs)} documents")

# ---------------- TF-IDF ----------------
print("Building TF-IDF matrix...")
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000
)

doc_matrix = vectorizer.fit_transform(docs)

# ---------------- LOAD QUERIES ----------------
queries = pd.read_csv("queries.dev.tsv", sep="\t", names=["qid", "query"], dtype=str)

# ---------------- LOAD QRELS ----------------
qrels_df = pd.read_csv("qrels.dev.tsv", sep="\t", names=["qid", "_", "pid", "rel"], dtype=str)
qrels_df["rel"] = qrels_df["rel"].astype(int)

qrels = {}
for _, row in qrels_df.iterrows():
    if row["rel"] > 0:
        qrels.setdefault(row["qid"], set()).add(row["pid"])

queries = queries[queries["qid"].isin(qrels.keys())]

print(f"Filtered queries: {len(queries)}")

# ---------------- SEARCH ----------------
def search(query, top_k=1000):
    q_vec = vectorizer.transform([query])
    scores = linear_kernel(q_vec, doc_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [pids[i] for i in top_idx]

# ---------------- METRICS ----------------
def mrr_at_k(ranked, relevant, k=10):
    for i, pid in enumerate(ranked[:k]):
        if pid in relevant:
            return 1 / (i + 1)
    return 0

def recall_at_k(ranked, relevant, k=1000):
    retrieved = set(ranked[:k])
    return len(retrieved & relevant) > 0

# ---------------- EVALUATION ----------------
print("\nRunning TF-IDF search...")

start_time = time.time()

mrr_scores = []
recall_hits = 0

for _, row in tqdm(queries.iterrows(), total=len(queries)):
    qid, query = row["qid"], row["query"]

    ranked = search(query, top_k=1000)
    rel = qrels[qid]

    mrr_scores.append(mrr_at_k(ranked, rel, 10))
    recall_hits += recall_at_k(ranked, rel, 1000)

end_time = time.time()

# ---------------- RESULTS ----------------
total_queries = len(queries)
elapsed = end_time - start_time

print("\n############################")
print(f"MRR @10: {np.mean(mrr_scores):.6f}")
print(f"QueriesRanked: {total_queries}")
print(f"Queries with >=1 relevant (for MRR): {total_queries}")
print(f"Recall @1000: {recall_hits / total_queries:.6f}")
print(f"Elapsed: {elapsed:.2f}s ({total_queries/elapsed:.2f} q/s)")
print("############################")