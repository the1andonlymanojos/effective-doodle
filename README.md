# MS MARCO Learning-to-Rank (L2R) with Elasticsearch + XGBoost

This repository contains a complete implementation of a Learning-to-Rank pipeline for the MS MARCO passage ranking dataset. It includes BM25 retrieval via Elasticsearch, feature engineering, XGBoost ranking models, neural embedding features, and a FastAPI serving layer.

## Overview

The system implements a two-stage retrieval architecture:
1. **First Stage**: BM25 retrieval from Elasticsearch to get candidate passages
2. **Second Stage**: Learning-to-Rank model (XGBoost) reranks the candidates

Optional enhancements include:
- Dense embedding features (SentenceTransformers)
- Cross-encoder reranking
- Proximity-based features
- Various negative sampling strategies

## Data Files

### `queries.train.tsv`
Training queries from MS MARCO. Format: `qid\tquery_text`
- ~808K queries
- Example: `121352\tdefine extreme`

### `queries.dev.tsv`
Development/validation queries. Same format as training.

### `queries.eval.tsv`
Evaluation queries for final testing.

### `qrels.train.tsv`
Training relevance judgments. Format: `qid\t0\tpid\trelevance`
- Binary relevance (0 or 1)
- ~533K relevance labels
- Example: `1185869\t0\t0\t1`

### `qrels.dev.tsv`
Development relevance judgments. Same format.

### `collection.tsv`
The passage collection. Format: `pid\tpassage_text`
- ~8.8M passages
- Each passage is a short text snippet

### `queries.tar.gz`
Compressed archive containing query files.

## Core Library Files

### `l2r_features.py` (Core Library)
The main library containing all reusable components:

**Key Classes:**
- `L2RConfig`: Configuration dataclass for all L2R experiments
  - Controls embedding features, proximity features, XGBoost params
  - Manages Elasticsearch index settings

**Key Functions:**
- `load_dotenv_like()`: Loads .env files with variable expansion
- `tokenize()`: Simple text tokenization
- `ngrams()`: Extracts n-grams from token lists
- `lexical_features_row()`: Extracts lexical features (overlap, coverage, phrase match, bigrams, trigrams)
- `prox_min_window_unordered()`: Minimum window covering all query terms (any order)
- `prox_ordered_window_optimal()`: Minimum ordered window (query order preserved)
- `prox_pair_avg_distance()`: Average distance between query term pairs
- `prox_cluster_density()`: Density of query term cluster
- `proximity_greedy_sequential()`: Greedy sequential proximity match
- `bm25_search()`: Query Elasticsearch with BM25
- `compute_embedding_features()`: Compute dense embedding similarities
- `featurize_candidates()`: Main feature extraction pipeline
- `train_ranker_from_bm25_pools_batched()`: Batched training to cap memory
- `save_ranker_artifacts()`: Save model, scaler, feature metadata
- `load_ranker_artifacts()`: Load saved artifacts

**Feature Categories:**
- Lexical: coverage, doc_len_log, phrase_exact, tri_overlap, bi_jacc, prox_score
- Proximity: prox_min_window, prox_ordered_window, prox_pair_avg, prox_cluster_density
- Per-query: bm25_norm, inv_rank, rank_pct
- Embeddings: emb_cos, emb_l2, emb_dot

## Main Demo Scripts

### `demo_l2r_notebook.py` (was demo.py)
Interactive learning-to-rank demo script. Steps through:
1. Setup: Load environment, connect to Elasticsearch
2. Load queries and qrels from TSV files
3. Sample queries that have relevance labels
4. Retrieve BM25 candidates for each query
5. Build training labels (1 if in qrels, else 0)
6. Feature engineering: BM25 score, term overlap, query coverage, doc length, phrase match, proximity, bigrams
7. Train XGBRanker with rank:ndcg objective
8. Compare BM25 vs L2R reranking
9. Evaluate with NDCG@10

Usage: `python demo_l2r_notebook.py`

### `demo_l2r_notebook.ipynb` (was demo.ipynb)
Jupyter notebook version with same content as demo.py but interactive.

### `demo_full_experiment_suite.py` (was demo_l2r_experiment_suite.py)
Comprehensive experiment suite testing 12+ feature configurations:

**Experiment Stages:**
1. `01_baseline`: Basic features (overlap, coverage, doc_len, phrase)
2. `02_bm25_variants`: Add BM25 raw/log/normalized variants
3. `03_plus_idf_tf`: Add IDF overlap and TF statistics
4. `04_plus_proximity`: Add proximity features
5. `05_plus_coverage_tiers`: Add coverage threshold features
6. `06_plus_dense`: Add embedding features (cosine, L2, dot)
7. `07_plus_dense_extras`: Add cos² and BM25×cos interactions
8. `08_plus_query_norm_flags`: Per-query normalization
9. `09_plus_interactions`: Explicit feature interactions
10. `10_neg_rand5`: Add 5 random negatives per query
11. `11_curated_train_mix`: Curated mix (pos + hard + lex + rand)
12. `12_curated_plus_rand5`: Curated + extra random negatives

**Features Evaluated:**
- BM25 baseline
- Cosine-only rerank (embedding similarity)
- L2R with feature combinations

Usage: `python demo_full_experiment_suite.py --out runs/my_run.txt --queries 5000`

## Experiment Scripts

### `experiment_embeddings_l2r.py` (was demo_es_embed.py)
Full experiment pipeline with embedding features:

**Configuration:**
- Query sizes: [10000]
- Candidate K values: [100]
- Train/test split: 80/20 by qid
- Evaluation: MRR@10 or NDCG@10

**Pipeline:**
1. Load queries from TSV
2. Sample target number of queries
3. For each query, retrieve top-K BM25 candidates
4. Label candidates using qrels
5. Extract lexical features
6. Optionally extract embedding features:
   - Encode queries with SentenceTransformer
   - Fetch passage vectors from Elasticsearch
   - Compute cosine similarity, L2 distance, dot product
7. Scale features
8. Train XGBRanker
9. Evaluate BM25 vs cosine-only vs L2R

**Embedding Configuration:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Passage vectors stored in Elasticsearch `embedding` field
- Batch processing for memory efficiency

Usage: `python experiment_embeddings_l2r.py`

### `experiment_embeddings_cpu.py` (was demo_embed_try_new_shit.py)
CPU-friendly version of embedding experiment:
- Uses CPU for SentenceTransformer
- Reduced batch sizes
- Same pipeline as experiment_embeddings_l2r.py

Usage: `python experiment_embeddings_cpu.py`

### `experiment_advanced_reranking.py` (was demo_try_shit.py)
Advanced experiment with cross-encoder reranking:

**Additional Features:**
- Proximity features (detailed implementation)
- Embedding features with caching
- Cross-encoder reranking (optional)
- Memory-efficient batched processing

**Cross-Encoder Pipeline:**
1. BM25 retrieval
2. L2R reranking
3. Take top-K from L2R
4. Cross-encoder scoring
5. Final ranking by cross-encoder scores

**Configuration:**
- Cross-encoder model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Train queries: 10000
- Eval queries: 5000
- Candidate K values: [100, 500, 1000]

Usage: `python experiment_advanced_reranking.py`

## Baseline & Evaluation Scripts

### `baseline_bm25_evaluation.py` (was es_bm25_baseline.py)
Elasticsearch BM25 baseline for MS MARCO:

**Features:**
- Mirrors Anserini evaluation flow
- Batch retrieval with ThreadPoolExecutor
- MRR@10 calculation
- Recall@k calculation
- Optional run file output (TREC format)
- Filters queries to those with qrels (like official eval)

**Metrics:**
- MRR@10 (Mean Reciprocal Rank)
- Recall@1000
- Queries ranked per second

Usage:
```bash
python baseline_bm25_evaluation.py \
  --queries queries.dev.tsv \
  --qrels qrels.dev.tsv \
  --out-run runs/es_bm25_dev.run.tsv \
  --workers 24
```

### `debug_bm25_rank_analysis.py` (was debug_bm25_qrel_ranks.py)
Debug tool to analyze where relevant documents rank in BM25 results:

**Analysis:**
- Sample queries (default 10,000)
- Retrieve top-1000 BM25 results per query
- Find best rank of any relevant document
- Calculate hit rates at various cutoffs (1, 3, 5, 10, 20, 50, 100, 200, 500, 1000)
- Compute MRR (0 if not in top-1000)

**Outputs (in `runs/bm25_rank_debug/`):**
- `per_query.csv`: Individual query results
- `summary.txt`: Human-readable statistics
- `hit_rates.csv`: Hit rate table
- `fig_hit_rates.png`: Bar chart of hit rates
- `fig_rank_cdf.png`: CDF of best relevant rank
- `fig_rank_hist.png`: Histogram of ranks
- `fig_reciprocal_rank.png`: Distribution of reciprocal ranks
- `fig_rank_with_misses.png`: Histogram including misses

Usage: `python debug_bm25_rank_analysis.py`

## Training & Serving

### `train_ranker_artifacts.py` (was train_l2r_artifacts.py)
Train and save ranker artifacts for the API:

**Process:**
1. Load environment from .env.local
2. Configure L2R with embeddings/proximity settings
3. Batched training to cap peak RAM
4. Save artifacts: xgb_ranker.json, scaler.pkl, features.pkl

**Arguments:**
- `--out-dir`: Where to save artifacts
- `--candidate-k`: BM25 candidates per query (default: 100)
- `--train-query-limit`: Max queries to use (default: 10000)
- `--query-batch-size`: Batch size for query processing (default: 200)

Usage:
```bash
python train_ranker_artifacts.py \
  --out-dir runs/l2r_api_artifacts \
  --candidate-k 100 \
  --train-query-limit 10000
```

### `api_l2r_server.py` (was api_l2r_demo.py)
FastAPI server for serving the L2R ranker:

**Endpoints:**
- `GET /health`: Health check with status
- `GET /sample`: Run inference on random query
  - Parameters:
    - `l2r_candidate_k`: BM25 candidates (1-5000)
    - `bm25_return_k`: BM25 results to return (1-1000)
    - `l2r_return_k`: L2R results to return (1-5000)
    - `ce_rerank_k`: Cross-encoder rerank cutoff (1-1000)
    - `ce_return_k`: Final results to return (1-1000)
    - `seed`: Random seed for query selection
  - Returns: Query info, BM25 results, L2R results, cross-encoder results, relevant PIDs

**Features:**
- Loads artifacts from `runs/l2r_api_artifacts/`
- Supports embedding features
- Optional cross-encoder reranking
- CORS enabled for frontend integration

Usage:
```bash
# First train artifacts
python train_ranker_artifacts.py --out-dir runs/l2r_api_artifacts

# Then start server
uvicorn api_l2r_server:app --reload --port 8000
```

## Configuration

### `.env.local`
Environment configuration file:

```bash
ES_LOCAL_URL=http://localhost:9200
ES_LOCAL_API_KEY=your_api_key_here
ES_INDEX=msmarco
```

**Required Variables:**
- `ES_LOCAL_URL`: Elasticsearch URL
- `ES_LOCAL_API_KEY`: Elasticsearch API key

**Optional Variables:**
- `ES_INDEX`: Index name (default: msmarco)
- `EMBEDDING_DEVICE`: cuda/cpu (default: cuda)
- `USE_EMBEDDING_FEATURES`: true/false
- `USE_CROSS_ENCODER_RERANK`: true/false

## Dependencies

### `requirements.txt`
```
numpy
pandas
matplotlib
elasticsearch
scikit-learn
xgboost
sentence-transformers
fastapi
uvicorn
```

Install: `pip install -r requirements.txt`

## Quick Start

1. **Setup Elasticsearch:**
   ```bash
   # Start local Elasticsearch with MS MARCO index
   # Ensure passages are indexed with fields: pid, passage, embedding (optional)
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your ES credentials
   ```

3. **Run BM25 baseline:**
   ```bash
   python baseline_bm25_evaluation.py --queries queries.dev.tsv --qrels qrels.dev.tsv
   ```

4. **Run basic L2R demo:**
   ```bash
   python demo_l2r_notebook.py
   ```

5. **Run full experiment suite:**
   ```bash
   python demo_full_experiment_suite.py --out runs/results.txt --queries 5000
   ```

6. **Train and serve model:**
   ```bash
   # Train
   python train_ranker_artifacts.py --out-dir runs/l2r_api_artifacts
   
   # Serve
   uvicorn api_l2r_server:app --reload
   
   # Test
   curl http://localhost:8000/health
   curl http://localhost:8000/sample
   ```

## Feature Engineering Details

### Lexical Features
- `coverage`: Fraction of query terms in document
- `doc_len_log`: Logarithm of document length
- `phrase_exact`: Binary - exact query phrase match
- `tri_overlap`: Trigram overlap count
- `bi_jacc`: Bigram Jaccard similarity
- `prox_score`: Proximity score (greedy sequential)

### Proximity Features
- `prox_min_window`: 1/(1 + min window covering all terms)
- `prox_ordered_window`: 1/(1 + min ordered window)
- `prox_pair_avg`: 1/(1 + avg pair distance)
- `prox_cluster_density`: Terms / span of cluster

### Embedding Features
- `emb_cos`: Cosine similarity (L2-normalized)
- `emb_l2`: L2 distance between vectors
- `emb_dot`: Dot product (same as cosine when normalized)

### Per-Query Features
- `bm25_norm`: Min-max normalized BM25 within query
- `inv_rank`: 1/rank by BM25
- `rank_pct`: Percentile rank (0-1)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Query                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 1: BM25 Retrieval (ES)                   │
│  - Retrieve top-K candidates using BM25 scoring            │
│  - Fast, lexical matching                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 2: Feature Extraction                    │
│  - Lexical features (overlap, coverage, proximity)         │
│  - Embedding features (optional, from ES)                  │
│  - Per-query normalization                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Stage 3: L2R Scoring (XGBoost)                 │
│  - Rank candidates using trained XGBRanker                 │
│  - Optimized for NDCG or pairwise ranking                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         Stage 4: Cross-Encoder Rerank (Optional)            │
│  - Score top-K L2R results with cross-encoder              │
│  - Final reranking for precision                           │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                     Final Results                           │
└─────────────────────────────────────────────────────────────┘
```

## Performance Notes

- **BM25 Retrieval**: ~50-100 queries/sec with 24 workers
- **Feature Extraction**: Depends on embedding model, ~10-50 queries/sec
- **XGBoost Inference**: Very fast, ~1000+ queries/sec
- **Cross-Encoder**: Slower, ~10-50 queries/sec depending on batch size


## License

MIT License - See LICENSE file for details.
