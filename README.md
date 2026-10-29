# Contract Intelligence Assistant

> **Enterprise-grade legal clause retrieval and confident abstention engine built on LegalBench-RAG.**

---

## Overview

The **Contract Intelligence Assistant** is a production-quality Retrieval-Augmented Generation (RAG) system purpose-built for legal contracts. It searches natural-language questions against a corpus of real SEC filing contracts, returns exact character-offset passage references, and — critically — **abstains with an explicit reason** when no relevant clause exists in the indexed corpus.

The system is evaluated on the **LegalBench-RAG mini benchmark** (194 CUAD queries across 29 unique contracts) using strict **character-level Precision, Recall, and F1** scores, with a dual abstention benchmark using 35 expert-authored unanswerable queries.

---

## Directory Structure

```
contract-intelligence-assistant/
├── data/
│   ├── benchmarks/cuad.json         # Full LegalBench-RAG CUAD benchmark
│   ├── corpus/cuad/                 # 29 raw contract .txt files
│   ├── cuad_mini.json               # 194-query mini benchmark with train/test splits
│   └── unanswerable_queries.json    # 35 expert-authored out-of-scope queries
├── src/
│   ├── config.py                    # Pydantic PipelineConfig schema
│   ├── document_loader.py           # UTF-8 document loader with char-level slice API
│   ├── chunker.py                   # Fixed, paragraph, clause-aware chunking + span refinement
│   ├── retriever.py                 # BM25, Dense (SentenceTransformer), Hybrid (RRF/Convex)
│   ├── reranker.py                  # CrossEncoder neural reranker with sigmoid calibration
│   ├── abstention.py                # Threshold / margin / adaptive abstention module
│   ├── evaluator.py                 # LegalBench-RAG char-level P/R/F1@k + abstention metrics
│   ├── failure_analyzer.py          # Automatic error taxonomy (6 failure modes)
│   ├── pipeline.py                  # End-to-end ContractIntelligenceEngine orchestrator
│   └── self_improvement_loop.py     # Automated iterative evaluation and hypothesis engine
├── tests/                           # 14 unit tests (all passing)
├── configs/                         # Auto-generated JSON config per experiment
├── experiments/cache/               # Disk-cached dense embeddings (fast re-run)
├── app.py                           # Streamlit web UI
├── cli.py                           # Command-line interface
├── run_improvement_loop.py          # Single-command self-improvement runner
├── download_and_prep_cuad.py        # Dataset download + offset verification script
├── create_unanswerable_queries.py   # Generates the 35-query abstention benchmark
└── experiment_log.json              # Auto-generated log of all iterations
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install sentence-transformers rank-bm25 scikit-learn streamlit pydantic pytest numpy scipy huggingface-hub torch
```

### 2. Download Dataset

```bash
python download_and_prep_cuad.py
python create_unanswerable_queries.py
```

This downloads 29 CUAD contracts and the benchmark, verifies all 272 character-level snippet offsets (100% exact match), and creates `data/cuad_mini.json` with deterministic train/test splits.

### 3. Run the Self-Improvement Loop (Train Only)

```bash
python run_improvement_loop.py
```

Runs 6+ planned iterations on the **train split only**. Logs accepted/rejected configurations to `experiment_log.json`.

### 4. Launch Web UI

```bash
streamlit run app.py
```

### 5. CLI Usage

```bash
# Single query
python cli.py search "What is the governing law of this agreement?"

# Interactive shell
python cli.py interactive

# Evaluate on test split (run once -- final evaluation only)
python cli.py evaluate --split test --output results/final_test_eval.json

# Run improvement loop
python cli.py loop
```

### 6. Run Tests

```bash
python -m pytest tests -v
# 14 passed, 0 failed
```

---

## System Architecture

```
User Query
    |
    v
+--------------------------------------------------------------------+
|                    ContractIntelligenceEngine                      |
|                                                                    |
|  +-------------+   +-----------------------------------------+   |
|  | BM25        |   |                                         |   |
|  | Retriever   +-->|  Hybrid Fusion (Convex or RRF)          |   |
|  +-------------+   |                                         |   |
|  +-------------+   |  bm25_w * norm(BM25) +                  |   |
|  | Dense       |   |  dense_w * norm(Dense)                  |   |
|  | Retriever   +-->|                                         |   |
|  | (MiniLM-L6) |   +------------------+----------------------+   |
|  +-------------+                      |                          |
|                                       v                          |
|                          +--------------------------+            |
|                          | CrossEncoder Reranker    |            |
|                          | (ms-marco-MiniLM-L-6-v2) |            |
|                          +-------------+------------+            |
|                                        |                         |
|                                        v                         |
|                          +--------------------------+            |
|                          |   Abstention Module      |            |
|                          | (threshold/margin/adapt) |            |
|                          +--------+--------+--------+            |
|                                   |        |                     |
|                    +--------------+        +--------------+      |
|                    v                                      v      |
|        +--------------------+              +--------------------+ |
|        | Ranked Passages    |              | Abstention Response | |
|        | with char offsets  |              | with reason string  | |
|        +--------------------+              +--------------------+ |
+--------------------------------------------------------------------+
```

---

## Key Design Decisions

### 1. Character-Level Indexing
Every `Chunk` carries exact `start_char` and `end_char` offsets, verified against original document bytes at ingest time. `document[start:end] == snippet['answer']` for all 272 ground-truth spans.

### 2. Three Chunking Strategies

| Strategy | Description | When Best |
|---|---|---|
| `fixed` | Sliding character window with whitespace snapping | Dense uniform documents |
| `paragraph` | Split on double-newlines, merge short paragraphs | Well-formatted contracts |
| `clause_aware` | Regex-based legal section/article header detection | Structured SEC agreements |

### 3. Hybrid Retrieval Fusion
BM25 captures exact keyword and section-number matches; Dense embeddings handle paraphrase and semantic variation. The convex combination is min-max normalized independently before fusion. RRF is available as an alternative.

### 4. Neural Reranking
A cross-encoder (`ms-marco-MiniLM-L-6-v2`) re-scores query-chunk pairs with full cross-attention. Raw logits are sigmoid-calibrated to [0,1] for threshold comparison.

### 5. Honest Abstention
The system returns an explicit abstention with a human-readable reason when confidence is insufficient. The threshold is calibrated on train to satisfy both guardrails:
- **False Abstention Rate <= 10%** (answerable queries must not be wrongly rejected)
- **Correct Abstention Rate >= 65%** (unanswerable queries must be caught)

### 6. Span Refinement
After retrieval, `refine_span` trims chunk boundaries to the 1-4 line window with highest token-match density, boosting Character Precision without sacrificing Recall.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `char_precision@k` | Fraction of retrieved chars that are ground-truth chars |
| `char_recall@k` | Fraction of ground-truth chars covered by retrieved passages |
| `char_f1@k` | Harmonic mean of char precision and recall |
| `doc_hit@k` | Whether any retrieved passage overlaps with the ground truth |
| `correct_abstention_rate` | Fraction of unanswerable queries correctly abstained |
| `false_abstention_rate` | Fraction of answerable queries wrongly abstained |
| `abstention_accuracy` | Balanced accuracy across both abstention directions |

---

## Failure Taxonomy

The `FailureAnalyzer` automatically categorises each query into one of:

| Code | Description | Fix Applied |
|---|---|---|
| `INCORRECT_ABSTENTION` | Answerable query wrongly abstained | Lower threshold |
| `FAILED_TO_ABSTAIN` | Unanswerable query returned passages | Raise threshold / enable reranker |
| `WRONG_DOCUMENT` | Retrieved from wrong contract | Enable hybrid/dense retrieval |
| `WRONG_SECTION` | Right contract, wrong clause | Clause-aware chunking |
| `PARTIAL_SPAN_COVERAGE` | Clause truncated by chunk boundary | Clause-aware split |
| `LOW_PRECISION_DILUTION` | Answer in chunk but diluted by noise | Enable span refinement |

---

## Configuration

All hyperparameters are controlled via `PipelineConfig` (Pydantic v2):

```python
from src.config import PipelineConfig

config = PipelineConfig(
    chunk_strategy="clause_aware",    # fixed | paragraph | clause_aware
    chunk_size=1000,
    retriever_type="hybrid",           # bm25 | dense | hybrid
    bm25_weight=0.35,
    dense_weight=0.65,
    use_reranker=True,
    abstention_mode="score_threshold",
    abstention_threshold=0.22,
    span_refinement=True
)
```

Configs are saved as JSON under `configs/` per experiment and logged to `experiment_log.json`.
