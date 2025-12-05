import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import random
import numpy as np
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import QMCSampler
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import ndcg_score
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm
from transformers import AutoTokenizer

import chromadb
import warnings
import subprocess
import threading
import time
import csv
import subprocess
import datetime
import torch
import gc
from datasets import Dataset
from calflops import calculate_flops
from optuna.importance import FanovaImportanceEvaluator
import plotly.graph_objects as go


warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
    plot_intermediate_values,
    plot_edf,
    plot_rank,
    plot_pareto_front,
    plot_timeline
)



# --------------------
# CONFIG
# --------------------

# PATHS
TRAIN_PATH = "data/train.jsonl"
CHROMA_DB_PATH = "knowledge_db"
COLLECTION_NAME = "baseline_embeddings_mistral"
VALIDATION_PATH = "data/validation.jsonl"  

# MODEL
EMBED_MODEL = "intfloat/e5-mistral-7b-instruct"
OUTPUT_DIR = "/workspace/fewshot_model/"
RESULTS_DIR = "results/optuna_plots_10_fewshot_mistral_40_trials_0311/"
RANDOM_SEED = 29
random.seed(RANDOM_SEED) 

# STUDY NAME + RESULTS
STUDY_NAME = "10_fewshot_mistral_0311_optuna"  # ← change per study run
OPTUNA_DB_PATH = os.path.join(RESULTS_DIR, f"{STUDY_NAME}.db")


# TUNING
FEW_SHOT_PERCENTAGE = 0.1  # Use subset for faster tuning
BEST_VAL_NDCG = 0.0
TOP_K = [1,5,10]  # For nDCG@5
SOBOL_TRIALS = 32
TPE_TRIALS = 8
FLOP_AVG_LGTH = 700 # in amount of tokens

# Hyperparameters for Bayesian Optimisation
LEARNING_RATE_LB = 5e-8
LEARNING_RATE_UB = 2e-6

BATCH_SIZE_OPTIONS = [1,2,4,8]

ACCUMULATION_STEPS_OPTIONS = [1, 2, 4, 8]

EPOCHS_LB = 1
EPOCHS_UB = 3

WARMUP_RATIO_LB = 0.01
WARMUP_RATIO_UB = 0.2

SCHEDULER_OPTIONS = ["linear", "cosine", "constant"]
 
# --------------------
# CREATE OUTPUT DIRS
# --------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# --------------------
# SAVE HYPERPARAMETER CONFIGURATION
# --------------------

def save_bo_config(filepath):
    config = {
        # General setup
        "RESULTS_DIR": RESULTS_DIR,
        "RANDOM_SEED": RANDOM_SEED, 

        # Tuning configuration
        "FEW_SHOT_PERCENTAGE": FEW_SHOT_PERCENTAGE,
        "TOP_K": TOP_K,
        "SOBOL_TRIALS": SOBOL_TRIALS,
        "TPE_TRIALS": TPE_TRIALS,
        "FLOP_AVG_LGTH": FLOP_AVG_LGTH,

        # Hyperparameter search space
        "LEARNING_RATE_RANGE": [LEARNING_RATE_LB, LEARNING_RATE_UB],
        "BATCH_SIZE_OPTIONS": BATCH_SIZE_OPTIONS,
        "ACCUMULATION_STEPS": ACCUMULATION_STEPS_OPTIONS,
        "EPOCHS_RANGE": [EPOCHS_LB, EPOCHS_UB],
        "WARMUP_RATIO_RANGE": [WARMUP_RATIO_LB, WARMUP_RATIO_UB],
        "SCHEDULER_OPTIONS": SCHEDULER_OPTIONS
    }

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Parameter", "Value / Range"])
        for key, val in config.items():
            writer.writerow([key, val])

# Save the configuration to your results directory
save_bo_config(os.path.join(RESULTS_DIR, "bo_config_ranges.csv"))

# --------------------
# ESTIMATING FLOPS 
# --------------------
def estimate_flops_calflops(sentence_model, seq_len=700):
    """
    Reliable FLOP estimation for SentenceTransformer models using calflops.
    Bypasses tokenizer bug in calflops by manually providing tokenized tensors.
    """
    try:
        try:
            transformer = sentence_model[0].auto_model
        except AttributeError:
            transformer = sentence_model[0].model

        model_name = transformer.name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_name)

         # --- get model max length ---
        model_max_len = getattr(tokenizer, "model_max_length", 512)
        seq_len = min(seq_len, model_max_len)  # ensure we stay within bounds

        # --- manually create dummy input ---
        text = " ".join(["test"] * seq_len)
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )

        import torch
        # convert to tensors
        for k in inputs:
            inputs[k] = torch.tensor([inputs[k]], device=next(transformer.parameters()).device)

        # --- compute FLOPs safely ---
        flops, macs, params = calculate_flops(
            model=transformer,
            kwargs=inputs,
            print_detailed=False,
        )
        # 🧹 clean up possible string outputs (if calflops returns text)
        import re
        if isinstance(flops, str):
            match = re.search(r"[\d.]+", flops)
            flops = float(match.group()) if match else 0.0

        # Always return a float value (GFLOPs)
        return float(flops)

    except Exception as e:
        print(f"⚠️ FLOP estimation failed: {e}")
        return None


# --------------------
# CREATING CSV FOR SAVING FILES
# --------------------

def log_trial_results(trial_number, params, metrics, filepath, max_vram_gb, total_flops, metrics_train, compute_time_sec):
    """Append trial parameters and evaluation results to a CSV file.

    Records hyperparameters, validation/train metrics, resource usage and
    FLOP estimates for later analysis.

    Args:
        trial_number (int): Optuna trial number.
        params (dict): Hyperparameter dictionary suggested by Optuna.
        metrics (dict): Validation metrics (expects keys 'ndcg' and 'recall').
        filepath (str): Path to the CSV file to append to.
        max_vram_gb (float): Peak GPU memory used in GB.
        total_flops (float|None): Estimated total FLOPs for the trial (optional).
        metrics_train (dict): Training-sample metrics with same structure as `metrics`.
        compute_time_sec (float): Wall-clock seconds taken by the trial.

    Side effects:
        Writes a single row to `filepath` (creates file with header if missing).
    """
    fieldnames = [
        "trial_number", "lr", "batch_size", "epochs", "warmup_ratio", "scheduler",
        "gradient_accumulation_steps", "sobol_trials", "tpe_trials",
        "nDCG@1", "nDCG@5", "nDCG@10",
        "Recall@1", "Recall@5", "Recall@10", "train_nDCG@5",
        "max_vram_gb",  "FlOPS_est", "training_time_sec"
    ]

    # Check if file exists
    file_exists = os.path.isfile(filepath)

    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "trial_number": trial_number,
            "lr": params.get("lr"),
            "batch_size": params.get("batch_size"),
            "epochs": params.get("epochs"),
            "warmup_ratio": params.get("warmup_ratio"),
            "scheduler": params.get("scheduler"),
            "gradient_accumulation_steps": params.get("gradient_accumulation_steps"),
            "sobol_trials": params.get("sobol_trials"),
            "tpe_trials": params.get("tpe_trials"),
            "nDCG@1": metrics["ndcg"][1],
            "nDCG@5": metrics["ndcg"][5],
            "nDCG@10": metrics["ndcg"][10],
            "Recall@1": metrics["recall"][1],
            "Recall@5": metrics["recall"][5],
            "Recall@10": metrics["recall"][10],
            "train_nDCG@5": round(metrics_train["ndcg"][5],2),
            "max_vram_gb": round(max_vram_gb, 2),
            "FlOPS_est": round(total_flops, 2) if total_flops is not None else None,\
            "training_time_sec": round(compute_time_sec, 2)
        })

def plot_validation_vs_training(study, save_path):
    """Produce an interactive Plotly HTML comparing validation and training nDCG@5.

    Args:
        study (optuna.study.Study): Completed Optuna study object.
        save_path (str): Output path where the HTML plot will be written.
    """
    # Extract data
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    val_scores = [t.value for t in study.trials if t.value is not None]
    train_scores = [t.user_attrs.get("train_ndcg") for t in study.trials]

    fig = go.Figure()

    # Validation curve
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=val_scores,
        mode='lines+markers',
        name='Validation nDCG@5',
        line=dict(color='blue')
    ))

    # Training curve
    fig.add_trace(go.Scatter(
        x=trial_numbers,
        y=train_scores,
        mode='lines+markers',
        name='Training nDCG@5',
        line=dict(color='orange')
    ))

    fig.update_layout(
        title="Training vs Validation nDCG@5 per Trial",
        xaxis_title="Trial number",
        yaxis_title="nDCG@5 score",
        legend=dict(x=0, y=1),
        template="plotly_white"
    )

    fig.write_html(save_path)
    print(f"✅ Saved combined plot to {save_path}")




# --------------------
# DATA LOADERS
# --------------------
def load_few_shot_data(path, percentage):
    """Load a random sample of training examples for few-shot tuning.

    Each example is converted to a SentenceTransformers `InputExample` with
    three texts: [query, positive, negative].

    Args:
        path (str): Path to a JSONL training file where each line contains
            at least 'query', 'positive', and 'negative' fields.
        percentage (float): Fraction (0..1) of the file to sample.

    Returns:
        List[InputExample]: Sampled few-shot InputExample objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    sample_size = int(len(data)*percentage)
    few_shot_data = random.sample(data, sample_size)    

    examples = []
    for item in few_shot_data:
            query = item["query"]
            positive = item["positive"]
            negative = item["negative"] 
            examples.append(InputExample(texts=[query, positive, negative])
)
    return examples

def load_validation_data(path):
    """Load validation queries and matching ground-truth chunk ids.

    Expects a JSONL file where each line contains 'query' and 'chunk_id'.

    Returns:
        tuple: (queries: List[str], ground_truth_ids: List[str])
    """
    queries, ground_truth_ids = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():  # skip empty lines
                continue
            item = json.loads(line)
            queries.append(item["query"])
            ground_truth_ids.append(str(item["chunk_id"]))
    return queries, ground_truth_ids

def load_chroma_collection():
    """Load a ChromaDB collection and return embeddings and ids.

    Returns:
        tuple: (embeddings: np.ndarray, ids: List[str])
    """
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)
    # Fetch all embeddings
    data = collection.get(include=["embeddings"])
    embeddings = np.array(data["embeddings"])
    ids = data["ids"]  # IDs for reference
    return embeddings, ids

def sample_train_eval(path, n=200):
    """Sample a small subset of the training data for a quick train-sample eval.

    Returns query texts and their matching chunk_id values (as strings) used
    as ground-truth ids for quick in-memory evaluation.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]
    sample = random.sample(data, min(n, len(data)))
    queries = [item["query"] for item in sample]
    gt_ids = [str(item["chunk_id"]) for item in sample]   # gebruik chunk_id ipv positive
    return queries, gt_ids

# Convert InputExample list → HF Dataset with explicit triplet fields
def examples_to_triplet_dataset(examples):
    """Convert a list of SentenceTransformers InputExample objects into
    a HuggingFace Dataset of explicit triplets.

    Returns a Dataset with fields: 'anchor', 'positive', 'negative', 'label'.
    """
    data = []
    for e in examples:
        # Make sure there are exactly three texts
        if len(e.texts) == 3:
            data.append({
                "anchor": e.texts[0],
                "positive": e.texts[1],
                "negative": e.texts[2],
                "label": getattr(e, "label", 1.0)
            })
    return Dataset.from_list(data)

# ---------------------------------
# EVALUATION: nDCG@5 
# ---------------------------------
def compute_ndcg_in_memory(queries, model, doc_embeddings, db_ids, ground_truth_ids, k=TOP_K):
    """Compute nDCG@k and Recall@k for queries using in-memory dot-products.

    Args:
        queries (List[str]): Query strings.
        model (SentenceTransformer): Model used to encode queries.
        doc_embeddings (np.ndarray): Precomputed document embeddings (N x D).
        db_ids (List[str]): Document ids aligned with embeddings.
        ground_truth_ids (List[str]): Ground-truth ids aligned with queries.
        k (Iterable[int]): Values of k to compute.

    Returns:
        dict: {'ndcg': {k: score,..}, 'recall': {k: score,..}}
    """
    ndcg_at_k = {kk: [] for kk in k}
    recall_at_k = {kk: [] for kk in k}

    for query, gt_id in tqdm(zip(queries, ground_truth_ids), total=len(queries), desc="Evaluating on validation"):
        # Encode query
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        scores = q_emb @ doc_embeddings.T  # cosine similarity because all embeddings are normalized
        ranked_idx = np.argsort(scores[0])[::-1]  # sort descending
        ranked_ids = [db_ids[i] for i in ranked_idx]
        ranked_scores = scores[0][ranked_idx]  # reorder scores to match ranked_ids

        # Relevance labels aligned with ranked_ids
        relevance = [1 if rid == str(gt_id) else 0 for rid in ranked_ids]
        

        # Compute metrics
        for kk in k:
            if 1 in relevance:
                ndcg_at_k[kk].append(ndcg_score([relevance], [ranked_scores.tolist()], k=int(kk)))
            else:
                ndcg_at_k[kk].append(0.0)
            recall_at_k[kk].append(1 if str(gt_id) in ranked_ids[:kk] else 0)

        #breakpoint()

    results = {
        "ndcg": {kk: float(np.mean(ndcg_at_k[kk])) for kk in k},
        "recall": {kk: float(np.mean(recall_at_k[kk])) for kk in k}
    }

    # 🔹 PRINT FINAL AVERAGES
    print("\n=== Final Results ===")
    print("nDCG:", results["ndcg"])
    print("Recall:", results["recall"])

    return results

def compute_ndcg_train_eval(queries, ground_truth_ids, model, doc_embeddings, db_ids, k=TOP_K):
    """Compute metrics on a small training sample (same semantics as compute_ndcg_in_memory)."""
    ndcg_at_k = {kk: [] for kk in k}
    recall_at_k = {kk: [] for kk in k}

    for query, gt_id in tqdm(zip(queries, ground_truth_ids), total=len(queries), desc="Evaluating on train sample"):
        # Encode query
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        scores = (q_emb @ doc_embeddings.T).flatten()  # cosine similarity because all embeddings are normalized
        ranked_idx = np.argsort(scores)[::-1]         # sort descending
        ranked_ids = [db_ids[i] for i in ranked_idx]
        ranked_scores = scores[ranked_idx]

        # Relevance labels aligned with ranked_ids
        relevance = [1 if rid == str(gt_id) else 0 for rid in ranked_ids]

        # Compute metrics
        for kk in k:
            if 1 in relevance:
                ndcg_at_k[kk].append(
                    ndcg_score([relevance], [ranked_scores.tolist()], k=int(kk))
                )
            else:
                ndcg_at_k[kk].append(0.0)

            recall_at_k[kk].append(1 if str(gt_id) in ranked_ids[:kk] else 0)

    results = {
        "ndcg": {kk: float(np.mean(ndcg_at_k[kk])) for kk in k},
        "recall": {kk: float(np.mean(recall_at_k[kk])) for kk in k}
    }

    print("\n=== Train Sample Results ===")
    print("nDCG:", results["ndcg"])
    print("Recall:", results["recall"])
    return results


# --------------------
# OBJECTIVE FUNCTION
# --------------------

def objective(trial, validation_queries, ground_truth_ids, collection, train_eval_queries, train_eval_gt_ids):
    """Optuna objective function: train a trial and return validation nDCG@5.

    Steps: suggest hyperparameters, build a few-shot dataset, train, re-embed
    collection, compute validation metrics and log results. Returns nDCG@5.
    """
    start_time = time.time()  # Start timing
    global BEST_VAL_NDCG

    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Suggest hyperparameters
    lr = trial.suggest_float("lr", LEARNING_RATE_LB, LEARNING_RATE_UB, log=True)
    batch_size = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)
    epochs = trial.suggest_int("epochs", EPOCHS_LB,EPOCHS_UB)
    warmup_ratio = trial.suggest_float("warmup_ratio", WARMUP_RATIO_LB, WARMUP_RATIO_UB)
    scheduler = trial.suggest_categorical("scheduler", SCHEDULER_OPTIONS)
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", ACCUMULATION_STEPS_OPTIONS)


    # --- Load data ---
    examples = load_few_shot_data(TRAIN_PATH, FEW_SHOT_PERCENTAGE)
    train_dataset = examples_to_triplet_dataset(examples)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Train model
    model = SentenceTransformer(EMBED_MODEL, device=device)
    model = model.to(device)    # force move to GPU
    model.gradient_checkpointing_enable()


    # CALCULATE FLOPS 
    # Estimate FLOPs for an average 700-token input
    gflops = estimate_flops_calflops(model, seq_len=FLOP_AVG_LGTH)
    
    if gflops is None:
        print("⚠️ FLOP estimation failed — skipping FLOP scaling.")
        total_flops = 0  
    else:     
        total_flops = gflops * 3 * (len(train_dataset)) * epochs

    trial.set_user_attr("total_flops", total_flops)


    # print(f"Estimated total FLOPs for this training: {total_flops:.2f} FLOPs")

    training_args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(RESULTS_DIR, f"trial_{trial.number}"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=epochs,
            warmup_ratio=warmup_ratio,
            learning_rate=lr,
            lr_scheduler_type=scheduler,  
            optim="adamw_bnb_8bit",
            weight_decay=0.01,
            logging_steps=10,
            save_strategy="no",
            report_to="none"
        )

    loss = losses.MultipleNegativesRankingLoss(model=model).to(device)

    # --- Trainer ---
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,  
        loss=loss
    )

    trainer.train()
    # 🔹 Re-embed docs with the fine-tuned model before evaluating
    docs = collection.get(include=["documents"])  # no "ids" here
    texts = docs["documents"]
    db_ids = docs["ids"]  # still works, ids are always returned
    print(f"Re-embedding {len(texts)} documents for evaluation...")
    doc_embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    print(f"embedded all docs in new model")
    # Now evaluate directly via Chroma queries
    metrics = compute_ndcg_in_memory(validation_queries, model, doc_embeddings, db_ids, ground_truth_ids, k=TOP_K)
    ndcg = metrics["ndcg"][5]   # use nDCG@5 as your main Optuna objective
    trial.set_user_attr("ndcg", ndcg)
    
  # 🔹 extra check: performance op sample uit training
    train_metrics = compute_ndcg_train_eval(
        train_eval_queries, 
        train_eval_gt_ids,   # dit zijn nu positives
        model, 
        doc_embeddings, 
        db_ids, 
        k=TOP_K
    )
    trial.set_user_attr("train_ndcg", train_metrics["ndcg"][5])

    print(f"[Trial {trial.number}] "
        f"Train nDCG@5: {train_metrics['ndcg'][5]:.4f} | "
        f"Valid nDCG@5: {ndcg:.4f}")
    
    max_memory_bytes = torch.cuda.max_memory_allocated()
    max_memory_gb = max_memory_bytes / (1024 ** 3)
    compute_time_sec = time.time() - start_time
# --------------------------------------------------------
    # Evaluate on test set if this is the new best validation nDCG@5
    # ---------------------------------------------------------
    if ndcg > BEST_VAL_NDCG:
        print(f"🌟 New best validation nDCG@5: {ndcg:.4f} (previous best {BEST_VAL_NDCG:.4f})")
        BEST_VAL_NDCG = ndcg

        TEST_PATH = "data/test.jsonl"
        if os.path.exists(TEST_PATH):
            test_queries, test_gt_ids = load_validation_data(TEST_PATH)
            docs = collection.get(include=["documents"])
            texts, db_ids = docs["documents"], docs["ids"]

            print("Evaluating best model on test set...")
            test_metrics = compute_ndcg_in_memory(
                test_queries, model, doc_embeddings, db_ids, test_gt_ids, k=TOP_K
            )

            # --- Extract all metrics ---
            test_ndcg1 = test_metrics["ndcg"].get(1, None)
            test_ndcg5 = test_metrics["ndcg"].get(5, None)
            test_ndcg10 = test_metrics["ndcg"].get(10, None)

            test_recall1 = test_metrics["recall"].get(1, None)
            test_recall5 = test_metrics["recall"].get(5, None)
            test_recall10 = test_metrics["recall"].get(10, None)

            print(
                f"🧪 Test evaluation for new best model: "
                f"nDCG@1={test_ndcg1:.4f}, nDCG@5={test_ndcg5:.4f}, nDCG@10={test_ndcg10:.4f}, "
                f"Recall@1={test_recall1:.4f}, Recall@5={test_recall5:.4f}, Recall@10={test_recall10:.4f}"
            )

            # ------------------------------
            # Log snapshot results to CSV
            # ------------------------------
            best_log_path = os.path.join(RESULTS_DIR, "best_model_log.csv")
            file_exists = os.path.isfile(best_log_path)
            with open(best_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "trial_number",
                        "val_nDCG@5",
                        "test_nDCG@1", "test_nDCG@5", "test_nDCG@10",
                        "test_Recall@1", "test_Recall@5", "test_Recall@10",
                        "FlOPS_est", "training_time_sec",
                        "lr", "batch_size", "scheduler", "timestamp"
                    ])
                writer.writerow([
                    trial.number,
                    round(ndcg, 4),
                    round(test_ndcg1, 4),
                    round(test_ndcg5, 4),
                    round(test_ndcg10, 4),
                    round(test_recall1, 4),
                    round(test_recall5, 4),
                    round(test_recall10, 4),
                    round(total_flops, 2) if 'total_flops' in locals() else None,
                    round(compute_time_sec, 2),
                    lr,
                    batch_size,
                    scheduler,
                    datetime.datetime.now().isoformat()
                ])

            # ------------------------------
            # Save top-10 retrieval results per query
            # ------------------------------
            output_path = os.path.join(RESULTS_DIR, f"trial_{trial.number}_top10_results.jsonl")
            print(f"💾 Saving top-10 retrievals per query to {output_path}")
            with open(output_path, "w", encoding="utf-8") as f_out:
                for query, gt_id in tqdm(zip(test_queries, test_gt_ids), total=len(test_queries), desc="Top-10 retrievals"):
                    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                    scores = q_emb @ doc_embeddings.T
                    ranked_idx = np.argsort(scores[0])[::-1]
                    top10_ids = [db_ids[i] for i in ranked_idx[:10]]
                    top10_scores = scores[0][ranked_idx[:10]].tolist()
                    json.dump({
                        "query": query,
                        "ground_truth": gt_id,
                        "top10_ids": top10_ids,
                        "top10_scores": top10_scores
                    }, f_out)
                    f_out.write("\n")
            print("✅ Top-10 retrievals saved for all test queries.")
        else:
            print("⚠️ No test.jsonl found — skipping test evaluation for best model.")


    # Save results to CSV
    log_path = os.path.join(RESULTS_DIR, "trial_results_log.csv")
    log_trial_results(trial.number, {
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "warmup_ratio": warmup_ratio,
        "scheduler": scheduler,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "sobol_trials": SOBOL_TRIALS,
        "tpe_trials": TPE_TRIALS,
    }, metrics, log_path, max_memory_gb, total_flops, train_metrics, compute_time_sec)

    del trainer, model, loss
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)


    return ndcg

# --------------------
# MAIN
# --------------------

def main():
    
    # Load validation data and precompute collection embeddings
    validation_queries, ground_truth_ids = load_validation_data(VALIDATION_PATH)
    train_eval_queries, train_eval_gt_ids = sample_train_eval(TRAIN_PATH, n=150)

    # Instead of loading old embeddings, just pass the collection
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Phase 1: Sobol initialization
    sobol_sampler = QMCSampler(qmc_type="sobol", scramble=True, seed=28)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{OPTUNA_DB_PATH}",
        direction="maximize",
        sampler=sobol_sampler,
        load_if_exists=True
    )

    print("Running Sobol initialization...")

    study.optimize(
        lambda t: objective(
            t,
            validation_queries,
            ground_truth_ids,
            collection,
            train_eval_queries,
            train_eval_gt_ids
        ),
        n_trials=SOBOL_TRIALS
    )
        # Phase 2: Switch to TPE
    print("Switching to TPE for Bayesian optimization...")
    study.sampler = optuna.samplers.TPESampler()
    study.optimize(
        lambda t: objective(
            t,
            validation_queries,
            ground_truth_ids,
            collection,
            train_eval_queries,
            train_eval_gt_ids
        ),
        n_trials=TPE_TRIALS
    )
    # Save Optuna results
    df = study.trials_dataframe()
    df.to_csv(os.path.join(RESULTS_DIR, "optuna_results.csv"), index=False)

    print("Best trial:")
    print(study.best_trial.params)

    # Save Optuna visualizations
    plot_optimization_history(study).write_html(os.path.join(RESULTS_DIR, "history.html"))
    plot_param_importances(study).write_html(os.path.join(RESULTS_DIR, "param_importance.html"))
    plot_param_importances(study, evaluator=FanovaImportanceEvaluator()).write_html(os.path.join(RESULTS_DIR, "param_importance_fanova.html"))
    plot_parallel_coordinate(study).write_html(os.path.join(RESULTS_DIR, "parallel_coord.html"))
    plot_slice(study).write_html(os.path.join(RESULTS_DIR, "slice.html"))
    plot_contour(study).write_html(os.path.join(RESULTS_DIR, "contour.html"))
    plot_intermediate_values(study).write_html(os.path.join(RESULTS_DIR, "intermediate.html"))
    plot_edf(study).write_html(os.path.join(RESULTS_DIR, "edf.html"))
    plot_rank(study).write_html(os.path.join(RESULTS_DIR, "rank.html"))
    plot_timeline(study).write_html(os.path.join(RESULTS_DIR, "timeline.html"))
    
    import optuna.visualization as vis

    # Directly access the stored FLOPs attribute
    fig = vis.plot_pareto_front(
        study,
        targets=lambda t: (t.value, t.user_attrs.get("total_flops")),
        target_names=["nDCG@5", "Total FLOPs"],
        include_dominated_trials=True
    )

    # Optional: use log scale for FLOPs axis (helps if FLOPs differ by large magnitudes)
    fig.update_yaxes(type="log")

    # Optional: customize hover tooltips with key parameters
    for trace in fig.data:
        trace.hovertemplate = (
            "Trial %{customdata[0]}<br>"
            "nDCG@5: %{x:.4f}<br>"
            "Total FLOPs: %{y:.2e}<br>"
            "LR: %{customdata[1]}<br>"
            "Batch size: %{customdata[2]}<br>"
            "<extra></extra>"
        )

    # Attach trial metadata for tooltips
    trial_custom_data = [
        [t.number, t.params.get("lr"), t.params.get("batch_size")] for t in study.trials if t.value is not None
    ]
    for trace in fig.data:
        trace.customdata = trial_custom_data

    # Save interactive HTML
    pareto_path = os.path.join(RESULTS_DIR, "pareto_ndcg_flops.html")
    fig.write_html(pareto_path)

    # --- fANOVA interactions heatmap
    try:
        # 1️⃣ Collect numeric parameters only
        numeric_params = set()
        for t in study.trials:
            for k, v in t.params.items():
                if isinstance(v, (int, float)):
                    numeric_params.add(k)

        if len(numeric_params) < 2:
            print("⚠️ Not enough numeric parameters for fANOVA interactions.")
        else:
            print(f"Using numeric parameters: {', '.join(sorted(numeric_params))}")

            # 2️⃣ Create filtered trials (drop non-numeric params)
            filtered_trials = []
            for t in study.trials:
                filtered_params = {k: v for k, v in t.params.items() if k in numeric_params}
                t_filtered = optuna.trial.create_trial(
                    params=filtered_params,
                    distributions={k: t.distributions[k] for k in filtered_params},
                    value=t.value
                )
                filtered_trials.append(t_filtered)

            # 3️⃣ Create temporary filtered study
            study_filtered = optuna.create_study(direction=study.direction)
            for t in filtered_trials:
                study_filtered.add_trial(t)

            # 4️⃣ Compute interactions safely
            evaluator = FanovaImportanceEvaluator(seed=42)
            interactions = evaluator.evaluate_interactions(study_filtered)

            if not interactions:
                print("⚠️ No valid interactions found.")
            else:
                params = sorted({p for pair in interactions.keys() for p in pair})
                matrix = np.zeros((len(params), len(params)))
                for (p1, p2), val in interactions.items():
                    i, j = params.index(p1), params.index(p2)
                    matrix[i, j] = val
                    matrix[j, i] = val

                plt.figure(figsize=(8, 6))
                im = plt.imshow(matrix, cmap="Blues", interpolation="nearest")
                plt.colorbar(im, label="Interaction importance")
                plt.xticks(range(len(params)), params, rotation=45, ha="right")
                plt.yticks(range(len(params)), params)
                plt.title("fANOVA Hyperparameter Interactions (numeric only)")
                plt.tight_layout()

                output_path = os.path.join(RESULTS_DIR, "fanova_interactions_heatmap.png")
                plt.savefig(output_path, dpi=300)
                plt.close()
                print(f"✅ Saved fANOVA interaction heatmap to: {output_path}")

    except Exception as e:
        print(f"⚠️ Could not compute fANOVA interactions: {e}")

    # Plot NCDG validation against NCDG TRAIN
    plot_validation_vs_training(study, os.path.join(RESULTS_DIR, "train_vs_valid_ndcg.html"))
    print(f"Plots saved in {RESULTS_DIR}")

if __name__ == "__main__":
   main()
