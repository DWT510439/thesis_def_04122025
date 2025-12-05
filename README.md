# LLM Fine-Tuning & RAG System

A comprehensive research repository for fine-tuning embedding models using LoRA and QLoRA techniques, with focus on retrieval-augmented generation (RAG) and hyperparameter optimization via Optuna.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Installation](#installation)
5. [Core Workflows](#core-workflows)
6. [Configuration](#configuration)
7. [Results & Evaluation](#results--evaluation)
8. [Key Concepts](#key-concepts)
9. [Troubleshooting](#troubleshooting)
10. [Citation & References](#citation--references)

---

## Overview

This project implements a systematic approach to fine-tuning embedding models for information retrieval tasks. It combines:

- **Embedding Models**: Support for multiple architectures (MiniLM, Mistral, Linq)
- **Fine-Tuning Methods**: LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) for memory-efficient training
- **Hyperparameter Search**: Optuna-driven Bayesian optimization (Sobol → TPE)
- **Evaluation Framework**: In-memory nDCG and Recall @ K metrics
- **FLOP Estimation**: Model-aware computational cost prediction with LoRA adjustment
- **Visualization**: Interactive Pareto fronts, parameter importance plots, and fANOVA interaction heatmaps

The repository is structured to support:
- **Data Preparation**: CSV to JSONL conversion, train/validation/test splitting, data augmentation
- **Baseline Evaluation**: Embedding model evaluation on vector stores
- **Few-Shot Learning**: Rapid adaptation with small data subsets
- **Full Fine-Tuning**: QLoRA-based training with quantization for large models
- **Results Analysis**: Comprehensive logging, visualization, and metric reporting

---

## Project Structure

```
ana-llm-finetuning-RAG/
│
├── thesis/                          # Main project folder (production code)
│   ├── requirements.txt             # Python dependencies
│   │
│   ├── data/                        # Training/validation/test datasets
│   │   ├── train.jsonl              # Training examples (query, positive, negative)
│   │   ├── validation.jsonl         # Validation queries with ground-truth chunk IDs
│   │   ├── test.jsonl               # Held-out test set
│   │   └── merged_chunks.jsonl      # Document collection for retrieval
│   │
│   ├── data processing/             # Data preparation utilities
│   │   ├── creatingData.py          # CSV → JSONL conversion, deduplication
│   │   ├── data_augmentation_batch.py  # Data augmentation pipeline
│   │   ├── splitting_data_*.py      # Train/validation/test split utilities
│   │   ├── assembling_negatives.py  # Negative sampling for triplet loss
│   │   ├── mergen_data.py           # Data merging and cleanup
│   │   ├── analyzing_double_enters.py  # Data quality checks
│   │   ├── ETE_*.py                 # End-to-end data processing workflows
│   │   ├── FA.py / FA_*.py          # Failure analysis utilities
│   │   └── All_Banks.csv            # Input data files
│   │
│   ├── knowledge_db/                # ChromaDB vector store (local)
│   │   ├── chroma.sqlite3           # Vector database file
│   │   └── {collection_uuid}/       # Collection shards
│   │
│   ├── baseline/                    # Baseline & pre-trained model evaluation
│   │   ├── create_knowledgebase_baseline.py  # Build Chroma collection from documents
│   │   ├── evaluate_baseline.py      # Evaluate baseline model on test queries
│   │   └── wizmap_visualize.py       # Visualization of embeddings (UMAP + Wizmap)
│   │
│   ├── fewshot/                     # Few-shot learning (small training sets)
│   │   ├── fewshot_linq.py          # Optuna tuning for Linq embedder (few-shot)
│   │   ├── fewshot_mini.py          # Optuna tuning for MiniLM embedder (few-shot)
│   │   ├── fewshot_mistral.py       # Optuna tuning for Mistral embedder (few-shot)
│   │   └── fewshot_mini_VRAM_testing.py  # VRAM-constrained variant of fewshot_mini
│   │
│   ├── qlora/                       # QLoRA fine-tuning (quantized + LoRA)
│   │   ├── qlora_mistral.py         # QLoRA tuning for Mistral (Sobol + TPE, 40 trials)
│   │   └── qlora_tune_linq.py       # QLoRA tuning for Linq (TPE only, 6 trials)
│   │
│   ├── qlora_adapters/              # Saved LoRA adapter weights
│   │   └── adapter_model.safetensors  # Serialized LoRA weights
│   │
│   ├── qlora_model_mini/            # Full fine-tuned MiniLM model checkpoint
│   │   └── {model files}
│   │
│   ├── checkpoints/                 # Trial-by-trial model checkpoints
│   │   └── model_{N}/               # Checkpoint for trial N
│   │       └── emissions.csv        # GPU power/carbon tracking per trial
│   │
│   └── results/                     # Output and analysis
│       ├── optuna_plots_*/          # Optuna study visualizations
│       ├── plots/                   # Custom result plots
│       ├── summary stats/           # Aggregated metrics and summaries
│       └── top10_baseline_embeddings_linq.jsonl  # Sample retrieval results
│
├── old/                             # Legacy scripts (reference only)
│   └── {baseline, fewshot, qlora variants and experimental scripts}
│
├── PCA.py                           # PCA analysis on embeddings
├── gpu_power_log.csv                # GPU power draw monitoring
├── hyperparameter_importances.png   # Visualization of hyperparameter importance
├── paragraph_summary.tex            # Research summary document
├── thesis.code-workspace            # VS Code workspace configuration
├── FETCH_HEAD                       # Git reference (internal)
└── README.md                        # This file

```

---

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/notilyze/ana-llm-finetuning-RAG.git
cd ana-llm-finetuning-RAG

# Navigate to the thesis directory
cd thesis
```

### 2. Install Dependencies

```bash
# Create a Python virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all required packages
pip install -r requirements.txt

# If using GPU, verify PyTorch with CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Prepare Your Data

```bash
# Option A: Use existing data (if available)
# Data files should be placed in thesis/data/
# - train.jsonl
# - validation.jsonl
# - test.jsonl
# - merged_chunks.jsonl

# Option B: Create data from scratch
cd "data processing"
python creatingData.py              # CSV → JSONL conversion
python splitting_data_train.py      # Create train set
python splitting_data_validation.py # Create validation set
cd ..
```

### 4. Create the Vector Store (Baseline)

```bash
cd baseline
python create_knowledgebase_baseline.py  # Builds Chroma collection from merged_chunks.jsonl
cd ..
```

### 5. Run a Quick Baseline Evaluation

```bash
cd baseline
python evaluate_baseline.py  # Evaluates pre-trained model on test set
cd ..
```

### 6. Run Few-Shot Fine-Tuning (Recommended for Quick Testing)

```bash
cd fewshot
python fewshot_mini.py  # Fast Optuna tuning with MiniLM model
# Or try other embedders:
# python fewshot_mistral.py
# python fewshot_linq.py
cd ..

# Results will be saved to: results/optuna_plots_{config_name}/
# View the Pareto front: results/optuna_plots_{config_name}/pareto_ndcg_flops.html
```

### 7. Run QLoRA Fine-Tuning (More Intensive)

```bash
cd qlora
python qlora_mistral.py    # QLoRA tuning for Mistral (32 Sobol + 8 TPE trials)
# Or:
python qlora_tune_linq.py  # QLoRA tuning for Linq (6 TPE trials)
cd ..
```

---

## Installation

### Requirements

- **Python**: 3.8 or higher (3.10+ recommended)
- **CUDA**: 11.8 or 12.x (for GPU acceleration; CPU-only mode supported but slower)
- **RAM**: 16 GB minimum; 32 GB+ recommended for QLoRA
- **Disk**: 50 GB+ (for model checkpoints and results)

### Step-by-Step Installation

#### 1. Clone and Enter Directory

```bash
git clone https://github.com/notilyze/ana-llm-finetuning-RAG.git
cd ana-llm-finetuning-RAG/thesis
```

#### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# Or
venv\Scripts\activate      # Windows PowerShell

# Or using conda
conda create -n rag-finetuning python=3.10
conda activate rag-finetuning
```

#### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel

# Install from requirements.txt
pip install -r requirements.txt

# If installation fails due to bitsandbytes on Windows, try:
pip install bitsandbytes-windows --no-deps
```

#### 4. Verify Installation

```bash
python -c "
import torch
from sentence_transformers import SentenceTransformer
from peft import LoraConfig
import optuna
import chromadb
print('All core dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"
```

#### 5. (Optional) Download Pre-Trained Models

Models are downloaded automatically on first run. To pre-cache them:

```bash
python -c "
from sentence_transformers import SentenceTransformer

models = [
    'sentence-transformers/all-MiniLM-L6-v2',
    'intfloat/e5-mistral-7b-instruct',
    'Linq-AI-Research/Linq-Embed-Mistral'
]

for model_name in models:
    print(f'Downloading {model_name}...')
    SentenceTransformer(model_name)
"
```

---

## Core Workflows

### Workflow 1: Baseline Evaluation

**Purpose**: Evaluate a pre-trained embedding model on a vector store without fine-tuning.

**Steps**:

1. **Build vector store** from document collection:
   ```bash
   cd baseline
   python create_knowledgebase_baseline.py
   ```
   - Reads `data/merged_chunks.jsonl`
   - Encodes documents using the configured model
   - Stores embeddings in ChromaDB at `knowledge_db/`

2. **Evaluate on test queries**:
   ```bash
   python evaluate_baseline.py
   ```
   - Loads test queries from `data/test.jsonl`
   - Retrieves top-K documents for each query
   - Computes nDCG@1, nDCG@5, nDCG@10, Recall@K
   - Writes results to `results/top10_baseline_embeddings_*.jsonl`

**Configuration**: Edit `baseline/{script}.py` to change:
- `EMBED_MODEL`: Which pre-trained model to use
- `COLLECTION_NAME`: Name of the Chroma collection
- `TOP_K`: Which cutoff values to report

**Output**: Metrics in terminal; per-query retrievals in JSONL format.

---

### Workflow 2: Few-Shot Fine-Tuning (Quick Experimentation)

**Purpose**: Rapidly fine-tune models on small training subsets for quick experimentation.

**When to use**:
- Prototyping and iterating on hyperparameters
- Limited computational budget
- Quick feedback loops

**Process**:

```bash
cd fewshot
python fewshot_mini.py
```

**Key parameters** (edit at top of script):
- `FEW_SHOT_PERCENTAGE = 0.1`: Use only 10% of training data
- `SOBOL_TRIALS = 10`: Sobol initialization phase
- `TPE_TRIALS = 10`: Bayesian optimization phase
- `LEARNING_RATE_LB / UB`: Learning rate search range
- `BATCH_SIZE_OPTIONS`: Batch sizes to try
- `EPOCHS_LB / UB`: Number of training epochs

**What happens**:

1. Load few-shot training examples (small percentage of `data/train.jsonl`)
2. Initialize Optuna study with Sobol sampler (explores parameter space uniformly)
3. Switch to TPE sampler (Bayesian optimization based on trial history)
4. For each trial:
   - Suggest hyperparameters
   - Train model with those hyperparameters
   - Evaluate on validation set (nDCG@5)
   - Track VRAM usage and training time
   - Optionally evaluate on test set if best validation score improves
5. Export results:
   - `trial_results_log.csv`: All trials + metrics
   - `optuna_results.csv`: Optuna trial history
   - `pareto_ndcg_flops.html`: Interactive Pareto front
   - `history.html`, `param_importance.html`: Optuna diagnostics
   - `train_vs_valid_ndcg.html`: Training vs validation comparison

**Output location**: `results/optuna_plots_{config_name}/`

---

### Workflow 3: QLoRA Fine-Tuning (Production)

**Purpose**: Fine-tune large models efficiently using 4-bit quantization + LoRA for production deployments.

**When to use**:
- Fine-tuning models > 7B parameters
- Memory-constrained environments
- Production-grade training with FLOPs budgeting

**Process**:

```bash
cd qlora
python qlora_mistral.py  # 40 trials (32 Sobol + 8 TPE)
# Or
python qlora_tune_linq.py  # 6 trials (TPE only)
```

**Key differences from few-shot**:
- **4-bit Quantization**: Uses `bitsandbytes` for 4-bit quantized base model
- **LoRA Adapters**: Only trains low-rank adapter weights; base model frozen
- **FLOP Estimation**: Calculates FLOPs with LoRA adjustment: FLOPs_LoRA = (2/3 + 2r/N) × FLOPs_FullFT
- **More Trials**: Typically more comprehensive search for production

**QLoRA Configuration**:
- `LORA_RANK`: Adapter rank values to search (e.g., [32, 64, 128])
- `LORA_ALPHA`: Scaling factor for LoRA updates
- `LORA_DROPOUT_LB / UB`: Dropout range for adapters
- `TARGET_MODULES`: Which transformer layers to adapt (attention + MLP)

**Output location**: `results/optuna_plots_{config_name}/` (same as few-shot)

**GPU Memory Requirements**:
- QLoRA Mistral-7B: ~16-20 GB VRAM
- QLoRA Linq-Mistral: ~16-20 GB VRAM
- Few-shot MiniLM: ~8-12 GB VRAM

---

### Workflow 4: Data Preparation

**Purpose**: Process raw CSV data into training datasets.

**Steps**:

1. **Convert CSV to JSONL**:
   ```bash
   cd "data processing"
   python creatingData.py
   ```
   - Reads CSV files from current directory
   - Cleans and validates text fields
   - Outputs to `merged_chunks.jsonl`

2. **Split into train/validation/test**:
   ```bash
   python splitting_data_train.py
   python splitting_data_validation.py
   python splitting_data_test.py
   ```
   - Allocates documents to train (70%), validation (15%), test (15%)
   - Outputs `train.jsonl`, `validation.jsonl`, `test.jsonl`

3. **Assemble negative samples** (for triplet loss training):
   ```bash
   python assembling_negatives.py
   ```
   - Takes positive (query, document) pairs
   - Samples random negatives from the collection
   - Creates triplet format: {query, positive, negative}

4. **Data augmentation** (optional):
   ```bash
   python data_augmentation_batch.py
   ```
   - Expands training set using LLM-based paraphrasing
   - Creates variations to improve model robustness

---

## Configuration

### Model Selection

Edit the top of any tuning script to choose your model:

```python
# MiniLM (fastest, smallest)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Mistral-7B (large, requires significant VRAM)
EMBED_MODEL = "intfloat/e5-mistral-7b-instruct"

# Linq-Mistral (proprietary, high-quality)
EMBED_MODEL = "Linq-AI-Research/Linq-Embed-Mistral"
```

### Hyperparameter Search Ranges

Each tuning script defines bounds for Optuna to explore:

```python
LEARNING_RATE_LB = 1e-6      # Lower bound
LEARNING_RATE_UB = 1e-4      # Upper bound

BATCH_SIZE_OPTIONS = [1, 2, 4, 8, 16, 32]
EPOCHS_LB = 1
EPOCHS_UB = 3

WARMUP_RATIO_LB = 0.01
WARMUP_RATIO_UB = 0.2

SCHEDULER_OPTIONS = ["linear", "cosine", "constant"]
ACCUMULATION_STEPS_OPTIONS = [1, 2, 4, 8]
```

### Dataset Configuration

```python
TRAIN_PATH = "data/train.jsonl"
VALIDATION_PATH = "data/validation.jsonl"
CHROMA_DB_PATH = "knowledge_db"
COLLECTION_NAME = "baseline_embeddings_mini"

# For few-shot, reduce the percentage
FEW_SHOT_PERCENTAGE = 0.1  # Use only 10% of training data
```

### Results Output

```python
RESULTS_DIR = "results/optuna_plots_50_fewshot_mini_0711/"
STUDY_NAME = "50_fewshot_mini_0711_optuna"
OPTUNA_DB_PATH = os.path.join(RESULTS_DIR, f"{STUDY_NAME}.db")
```

---

## Results & Evaluation

### Output Artifacts

Each tuning run generates:

#### 1. CSV Logs
- **`trial_results_log.csv`**: All trials with hyperparameters and metrics
  ```
  trial_number,lr,batch_size,epochs,nDCG@5,Recall@5,max_vram_gb,training_time_sec,...
  0,0.0001,8,2,0.6234,0.8123,15.2,240.5,...
  1,0.00005,4,3,0.6101,0.7891,14.1,310.2,...
  ```

- **`best_model_log.csv`**: Only trials that achieved new best validation score
- **`bo_config_ranges.csv`**: Hyperparameter search space configuration

#### 2. Interactive Plots (HTML)
- **`pareto_ndcg_flops.html`**: 2D plot comparing nDCG@5 vs computational cost (FLOPs)
  - Hover to see trial number and key hyperparameters
  - Identify efficient models vs high-performance models

- **`history.html`**: Optuna optimization history over trials

- **`param_importance.html`**: Which hyperparameters matter most (global importance)

- **`param_importance_fanova.html`**: fANOVA-based importance scores

- **`parallel_coord.html`**: Parallel coordinate plot showing all trial dimensions

- **`train_vs_valid_ndcg.html`**: Overlaid training vs validation scores
  - Detect overfitting if train >> validation

- **`fanova_interactions_heatmap.png`**: Heatmap of hyperparameter interactions

#### 3. Model Checkpoints
```
results/optuna_plots_{config}/
├── trial_0/          # Trial 0 checkpoint
├── trial_1/          # Trial 1 checkpoint
├── trial_{best}/     # Best trial checkpoint
└── qlora_adapters/   # Best LoRA adapter weights (QLoRA only)
```

### Interpreting Metrics

- **nDCG@K** (Normalized Discounted Cumulative Gain):
  - Measures ranking quality; higher is better [0, 1]
  - Penalizes wrong documents, especially at top positions
  - nDCG@5: how good are the top 5 results?

- **Recall@K**:
  - Fraction of relevant documents retrieved in top-K
  - Recall@10 = 0.8: 80% of relevant docs are in top 10

- **VRAM**:
  - Peak GPU memory used during training (GB)
  - Lower is better for deployment; allows larger batch sizes

- **Training Time**:
  - Wall-clock time per trial (seconds)
  - Trade-off: more epochs → better models but longer training

### Example Analysis

From a few-shot tuning run:

```
Best Trial: #47
  nDCG@5: 0.6540
  Recall@10: 0.8920
  Learning Rate: 3.2e-5
  Batch Size: 16
  Epochs: 2
  VRAM: 14.5 GB
  Training Time: 245 sec
```

**Interpretation**:
- Model achieves 65.4% nDCG@5, outperforming baseline (e.g., 60%)
- 89% of relevant docs are in top 10
- Efficient: only 14.5 GB VRAM, trainable in 4 min on single GPU
- Good hyperparameters for this dataset

---

## Key Concepts

### Few-Shot Learning

**Concept**: Train on a small subset of data to rapidly iterate and experiment.

- Reduces computation time (hours instead of days)
- Useful for hyperparameter prototyping
- Represents approximation; may not transfer to full-data training

**When to use**:
- Exploring new model architectures
- Testing data augmentation strategies
- Quick feedback on parameter sensitivity

**Typical settings**:
- Few-Shot Percentage: 5%-20% of training data
- Trials: 20-30 total
- Time: 2-6 hours per run

---

### QLoRA (Quantized LoRA)

**Concept**: Efficient fine-tuning combining two techniques:

1. **4-bit Quantization**: Reduces base model from 32-bit → 4-bit
   - Reduces memory by ~8x
   - Negligible accuracy loss (empirically shown)
   - Trade-off: slower training

2. **LoRA (Low-Rank Adaptation)**: Add small trainable adapters
   - Freeze base model; only train adapters
   - Adapter rank r << hidden size (e.g., r=64, hidden=4096)
   - Reduces trainable params by ~99%
   - Trade-off: sometimes slightly lower final accuracy

**Benefits**:
- Train 7B+ models on single 24 GB GPU
- Much faster than full fine-tuning
- Modular: swap adapters without reloading base model

**When to use**:
- Fine-tuning large models
- Memory-constrained environments
- Production deployments (fast inference with LoRA merging)

**Papers**:
- LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
- QLoRA: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs" (2023)

---

### Optuna Hyperparameter Search

**Concept**: Automated search for optimal hyperparameters using Bayesian optimization.

**Two-phase approach**:

1. **Phase 1: Sobol Initialization** (quasi-random sampling)
   - Explores parameter space uniformly
   - Doesn't use trial history
   - Good for initial coverage
   - Typical: 25-40 trials

2. **Phase 2: TPE** (Tree-structured Parzen Estimator)
   - Bayesian optimization with learned model of objective
   - Uses history to focus on promising regions
   - Typical: 5-20 trials

**Why two phases?**
- Sobol provides broad exploration
- TPE refines based on Sobol's feedback

**Configuration**:
```python
SOBOL_TRIALS = 32
TPE_TRIALS = 8
# Total: 40 trials per run
```

---

### FLOP Estimation

**Purpose**: Estimate computational cost to compare models on cost-quality trade-off.

**Formula**:
$$\text{FLOPs}_{\text{LoRA}} = \left(\frac{2}{3} + \frac{2r}{N}\right) \times \text{FLOPs}_{\text{FullFT}}$$

where:
- $r$ = LoRA rank
- $N$ = hidden size of model
- FullFT = full fine-tuning cost

**Example**:
- Mistral-7B, hidden_size=4096, rank=64
- Ratio = 2/3 + 2×64/4096 ≈ 0.699
- ~30% reduction vs full fine-tuning

**Usage**: Pareto analysis (nDCG vs FLOPs) helps find efficient models.

---

## Troubleshooting

### Installation Issues

**Problem**: `pip install -r requirements.txt` fails on bitsandbytes

**Solution**:
```bash
# Windows: use Windows-specific build
pip install bitsandbytes-windows --no-deps

# Linux: ensure CUDA toolkit is installed
sudo apt-get install nvidia-cuda-toolkit
pip install bitsandbytes

# Fallback: install without bitsandbytes (CPU mode)
pip install -r requirements.txt --ignore-installed bitsandbytes
```

---

**Problem**: Out of memory (OOM) error during training

**Solution**:
1. Reduce batch size: `BATCH_SIZE_OPTIONS = [1, 2, 4]`
2. Enable gradient accumulation: `ACCUMULATION_STEPS_OPTIONS = [2, 4, 8]`
3. Use fewer epochs: `EPOCHS_LB = 1, EPOCHS_UB = 2`
4. Use few-shot instead of QLoRA: smaller models, less memory
5. Reduce sequence length: `FLOP_AVG_LGTH = 512` (from 700)

**Check current VRAM**:
```bash
nvidia-smi  # Linux/WSL
# Or Python:
import torch
print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

**Problem**: CUDA device not found

**Solution**:
```bash
# Verify CUDA installation
nvidia-smi

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU mode (slower but works)
# Add this to top of script:
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
```

---

### Data Issues

**Problem**: `FileNotFoundError: data/train.jsonl not found`

**Solution**: Create datasets using data processing scripts:
```bash
cd "data processing"
python creatingData.py
python splitting_data_train.py
cd ..
```

Or place your own JSONL files in `data/` with correct structure:
```json
{"query": "...", "positive": "...", "negative": "...", "chunk_id": "..."}
```

---

**Problem**: Chroma collection not found

**Solution**: Build vector store first:
```bash
cd baseline
python create_knowledgebase_baseline.py
cd ..
```

Ensure `knowledge_db` folder exists and contains `chroma.sqlite3`.

---

### Training Issues

**Problem**: Training is very slow

**Solution**:
1. Check GPU usage: `nvidia-smi`
2. Reduce dataset size: `FEW_SHOT_PERCENTAGE = 0.1`
3. Use MiniLM model (faster): `EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`
4. Reduce sequence length: `FLOP_AVG_LGTH = 256`

---

**Problem**: Validation metrics not improving over trials

**Solution**:
1. Check learning rate range: may be too small or too large
2. Increase number of trials: more exploration needed
3. Verify data quality: check `data processing/analyzing_double_enters.py`
4. Try different model: MiniLM may be too small for task

---

### Visualization Issues

**Problem**: HTML plots not opening or displaying blank

**Solution**:
1. Ensure Plotly installed: `pip install plotly kaleido`
2. Check file exists: `ls -la results/optuna_plots_*/pareto_*.html`
3. Open in modern browser (Chrome, Firefox, Edge)
4. Check browser console for JavaScript errors

---

## Results Interpretation Guide

### Pareto Front Analysis

**What you're looking at**: Trade-off between model quality (nDCG@5) and computational cost (FLOPs).

**Interpretation**:
- **Top-right**: High nDCG, high FLOPs (best quality, expensive)
- **Bottom-left**: Low nDCG, low FLOPs (cheap, lower quality)
- **Along Pareto front**: Efficient models (no better model exists with lower cost AND better quality)

**How to choose**:
- **For production (online serving)**: Choose low-FLOP model on Pareto front
- **For research (offline evaluation)**: Choose highest nDCG regardless of FLOPs
- **For deployment with constraints**: Hover on frontier to find your sweet spot

---

### Parameter Importance

**What it shows**: Which hyperparameters have the most impact on final performance.

**Interpretation**:
- **High importance**: Fine-tuning this parameter matters; spend time optimizing
- **Low importance**: Relatively insensitive; can use default value

**Example**:
- Learning rate importance: 0.45 (very important)
- Batch size importance: 0.25 (somewhat important)
- Warmup ratio importance: 0.10 (less important)

---

## Citation & References

### Papers Referenced

1. **LoRA**: Hu et al. (2021), "[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09714)"
2. **QLoRA**: Dettmers et al. (2023), "[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)"
3. **Sentence-BERT**: Reimers & Gupta (2019), "[Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)"
4. **Optuna**: Akiba et al. (2019), "[Optuna: A Next-generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)"

### Model References

- **MiniLM**: Wang et al., "MiniLM: Deep Self-Attention Distillation for Compact Sentence Embeddings"
- **Mistral**: Jiang et al., "Mistral 7B" (via HuggingFace)
- **Linq**: Proprietary model by Linq-AI-Research
- **ChromaDB**: Open-source vector database for embeddings

---

## Contributing

This is a research repository. Suggestions for improvements:

1. **Performance**: Submit plots/metrics comparing different configurations
2. **Documentation**: Clarify confusing sections or add examples
3. **Bug fixes**: Report issues via email or GitHub issues
4. **New features**: New embedding models, loss functions, evaluation metrics

---

## License

[Add your license here - e.g., MIT, Apache 2.0, or research-only]

---

## Contact

For questions or support:
- **Repository**: [GitHub link]
- **Email**: [Your contact]
- **Issues**: GitHub Issues (recommended for bugs/feature requests)

---

**Last Updated**: December 2025
**Status**: Active Research Project
