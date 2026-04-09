import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm
import numpy as np

# ---------------- LOAD DATA ----------------
print("Loading collection...")
docs = []
pids = []

with open("collection.tsv", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        pid, text = line.strip().split("\t")
        pids.append(pid)
        docs.append(text)

        # LIMIT for sanity (IMPORTANT)
        if i > 50000:
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

# ---------------- SEARCH ----------------
def search(query, top_k=1000):
    q_vec = vectorizer.transform([query])
    scores = linear_kernel(q_vec, doc_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [pids[i] for i in top_idx]

# ---------------- MRR ----------------
def mrr_at_k(ranked, relevant, k=10):
    for i, pid in enumerate(ranked[:k]):
        if pid in relevant:
            return 1 / (i + 1)
    return 0

# ---------------- EVALUATION ----------------
print("Evaluating...")
mrr_scores = []

for _, row in tqdm(queries.iterrows(), total=len(queries)):
    qid, query = row["qid"], row["query"]

    ranked = search(query)
    rel = qrels[qid]

    mrr_scores.append(mrr_at_k(ranked, rel, 10))

print("\n====================")
print("TF-IDF MRR@10:", np.mean(mrr_scores))
print("====================")