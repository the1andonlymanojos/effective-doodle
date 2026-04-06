# MS MARCO Learning-to-Rank (L2R) with Elasticsearch + XGBoost

A complete L2R pipeline for MS MARCO passage ranking with BM25 retrieval, feature engineering, XGBoost ranking, embedding features, and cross-encoder reranking.

## Project Structure

```
effective-doodle/
├── l2r/                      # Core library
│   ├── __init__.py          # Package exports
│   └── features.py          # Feature engineering & training
├── server/                   # API server
│   └── main.py              # FastAPI server
├── training/                 # Training scripts
│   └── train_artifacts.py   # Train XGBRanker artifacts
├── evaluation/               # Evaluation scripts
│   ├── baseline_bm25.py    # BM25 baseline (MRR@k)
│   └── rank_analysis.py    # BM25 rank analysis
├── experiments/              # Experiment scripts
│   ├── shared.py            # Common utilities
│   ├── bm25_l2r.py          # BM25 + L2R (lexical)
│   ├── l2r_embeddings.py    # L2R + embeddings
│   └── l2r_cross_encoder.py # L2R + cross-encoder
├── notebooks/                # Jupyter notebooks
│   └── demo_l2r.ipynb
├── data/                      # Data files
│   ├── queries.*.tsv
│   ├── qrels.*.tsv
│   └── collection.tsv
├── runs/                      # Outputs
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env.local`:
```bash
ES_LOCAL_URL=http://localhost:9200
ES_LOCAL_API_KEY=your_api_key_here
ES_INDEX=msmarco
```

### 3. Run Experiments

**BM25 Baseline:**
```bash
python evaluation/baseline_bm25.py --data-dir data
```

**BM25 + L2R (lexical features only):**
```bash
python experiments/bm25_l2r.py --queries 5000 --candidates 100
```

**L2R + Embeddings:**
```bash
python experiments/l2r_embeddings.py --queries 5000 --candidates 100
```

**L2R + Cross-Encoder:**
```bash
python experiments/l2r_cross_encoder.py --queries 5000 --candidates 100
```

### 4. Train & Serve

```bash
# Train model
python training/train_artifacts.py --out-dir runs/l2r_artifacts

# Start API
uvicorn server.main:app --reload --port 8000
```

## Experiments

### `bm25_l2r.py`
Pure L2R with lexical features. Good baseline for measuring the value of learned reranking without embedding overhead.

**Features:** BM25, coverage, doc_len, phrase_match, ngram_overlap, proximity

### `l2r_embeddings.py`
L2R with sentence embedding features. Uses pre-indexed passage vectors from Elasticsearch.

**Features:** All lexical + cosine_sim, L2_distance, dot_product

### `l2r_cross_encoder.py`
Three-stage pipeline: BM25 → L2R → Cross-Encoder. Most accurate but slowest.

**Features:** All embedding features + cross-encoder rerank on top-k

## Core Library (`l2r/features.py`)

```python
from l2r import L2RConfig, bm25_search, train_ranker_from_bm25_pools_batched

cfg = L2RConfig(use_embedding_features=True, use_prox_features=True)
model, scaler, feat_cols, scale_cols, _ = train_ranker_from_bm25_pools_batched(
    es, cfg, "data/queries.train.tsv", "data/qrels.train.tsv",
    candidate_k=100, train_query_limit=10000
)
```

## Evaluation Scripts

### `baseline_bm25.py`
Computes BM25 MRR@10 and Recall@k against qrels.

```bash
python evaluation/baseline_bm25.py \
  --data-dir data \
  --queries queries.dev.tsv \
  --qrels qrels.dev.tsv \
  --workers 24
```

### `rank_analysis.py`
Analyzes where relevant documents rank in BM25 results. Outputs histograms and hit-rate tables.

```bash
python evaluation/rank_analysis.py \
  --data-dir data \
  --n-queries 10000 \
  --retrieve-k 1000
```

## Feature Reference

|Feature | Description |
|--------|-------------|
|`bm25` | Elasticsearch BM25 score |
|`coverage` | Fraction of query terms in doc |
|`doc_len_log` | Log document length |
|`phrase_exact` | Exact phrase match |
|`bi_jacc` | Bigram Jaccard |
|`tri_overlap` | Trigram overlap |
|`prox_greedy` | Greedy proximity match |
|`emb_cos` | Query-doc embedding cosine |
|`emb_l2` | Query-doc L2 distance |
|`emb_dot` | Query-doc dot product |

## Architecture

```
Query
  │
  ▼
┌─────────────────┐
│ BM25 Retrieval  │  → top-K candidates
└─────────────────┘
  │
  ▼
┌─────────────────┐
│Feature Extraction│  → lexical + embedding
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ XGBoost Ranker  │  → L2R scores
└─────────────────┘
  │
  ▼
┌─────────────────┐
│ Cross-Encoder   │  (optional) → final ranking
└─────────────────┘
  │
  ▼
Ranked Results
```

## Performance

| Stage | Latency (approx) |
|-------|-----------------|
|BM25 | ~10-50ms/query |
| L2R | ~1-5ms/query |
| Cross-Encoder | ~50-200ms/query |

## License

MIT