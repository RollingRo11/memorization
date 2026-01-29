# Project Overview: K-FAC Curvature Edit

This repository implements **K-FAC (Kronecker-Factored Approximate Curvature)** treatment for Large Language Models (LLMs), specifically targeting OLMo models. 

## What It Does

The primary goal is to **suppress rote memorization** in LLMs while preserving their general reasoning capabilities and shared knowledge structure. 

It achieves this by:
1.  **Analyzing Curvature:** Computing the curvature of the loss landscape (Hessian) using the K-FAC approximation. This involves calculating activation covariance ($A = E[aa^T]$) and gradient covariance ($G = E[gg^T]$) for linear layers (MLP projections).
2.  **Editing Weights:** Projecting the weights of specific layers onto the top eigenspace of the curvature. This retains the "mass" of the curvature that corresponds to general structure while removing components associated with specific, rote memorization.
3.  **Evaluating:** Measuring the impact on memorization (using specific metrics like strict/loose accuracy on memorized sequences) and general performance (perplexity, nDCG).

## How It Works

The workflow consists of two main stages:

### 1. Factor Collection
First, you stream data (e.g., from Dolma or OLMo datasets) through the model. For selected layers (gate, up, down projections in MLPs), the script collects the K-FAC factors ($A$ and $G$). These matrices capture how the model uses these weights.

### 2. Application & Evaluation
Next, you load the collected factors and the model. You specify which layers to edit and how much "variance" (curvature mass) to retain. The script:
*   Decomposes $A$ and $G$ (eigendecomposition).
*   Selects the top eigenvectors that explain the target variance ratio ($ho$).
*   Projects the original weights onto this retained subspace.
*   Evaluates the "edited" model against baselines.

## Repository Structure

*   **`data/`**: Scripts for data handling and factor collection.
    *   `collect_kfac_multilayer.py`: The main script for collecting K-FAC factors ($A$ and $G$).
    *   `baseline_generator.py`: Generates baseline top-k predictions for nDCG evaluation.
    *   `paths.py`: Helper for defining data paths.
*   **`evaluations/`**: Scripts for applying the edit and running evaluations.
    *   `eval_mem_kfac.py`: The main entry point for applying K-FAC edits and running the full evaluation suite.
*   **`metrics/`**: Evaluation metrics implementation.
    *   `memorization_evaluator.py`: Computes memorization accuracy (strict/loose) and Levenshtein distance.
    *   `ndcg_evaluator.py`: Computes Normalized Discounted Cumulative Gain (nDCG) to measure ranking quality.
    *   `perplexity.py`: Computes perplexity.
*   **`kfac_treatment_pairwise.py`**: Core logic for the K-FAC treatment. Contains the `KFACTreatment` class that handles the math of eigendecomposition and weight projection.
*   **`main.py`**: (Currently a placeholder).

## Setup

1.  **Requirements:**
    *   Python 3.10+
    *   PyTorch (CUDA recommended)
    *   Transformers, Datasets, Accelerate
    *   NumPy

2.  **Installation:**
    Install dependencies using pip or uv:
    ```bash
    uv pip install -r requirements.txt
    # OR
    pip install torch transformers datasets accelerate numpy tqdm
    ```

## Usage Guide

### Step 1: Collect K-FAC Factors

Use `data/collect_kfac_multilayer.py` to collect $A$ and $G$ matrices. This step requires a GPU.

```bash
python data/collect_kfac_multilayer.py \
  --model "allenai/OLMo-2-1124-7B" \
  --corpus dolmo \
  --layers_per_pass 4 \
  --target_blocks 28 29 30 31 \
  --save_dir "data/kfac_factors/olmo2-7b"
```

**Key Arguments:**
*   `--model`: Model identifier (default: `allenai/OLMo-2-1124-7B`).
*   `--corpus`: Dataset source, either `olmo` or `dolmo` (default: `dolmo`).
*   `--target_blocks`: Space-separated list of 0-based block indices to process (e.g., `0 1 2` or `range(16)`).
*   `--layers_per_pass`: Number of layers to process in one go to manage GPU memory.
*   `--save_dir`: Directory to save the `.pt` factor files.

### Step 2: Apply Edit & Evaluate

Use `evaluations/eval_mem_kfac.py` to apply the collected factors and evaluate the result.

```bash
python evaluations/eval_mem_kfac.py \
  --model-size 7b \
  --layers-json '{"31": {"gate": 0.8, "up": 0.8, "down": 0.8}}' \
  --use-cache \
  --dtype bfloat16
```

**Key Arguments:**
*   `--model-size`: `1b` or `7b`.
*   `--layers-json`: JSON string defining which layers to edit and the variance ratio ($ho$) for each projection (`gate`, `up`, `down`). $ho \in [0, 1]$.
    *   Example: `'{"31": {"gate": 0.8, "up": 0.8, "down": 0.8}}'` means edit block 31, keeping 80% of curvature mass for all projections.
*   `--layers-file`: Alternatively, provide a path to a JSON file with the configuration.
*   `--use-cache`: Use cached eigenvectors/weights if available (speeds up repeated runs).
*   `--perplexity`: Compute perplexity on BSN clean set.
*   `--skip-baseline`: Skip the pre-edit evaluation if you only care about the post-edit result.

## Metrics

The evaluation reports:
*   **Memorization:**
    *   `strict_acc`: Exact match accuracy on memorized sequences.
    *   `loose_acc`: Accuracy allowing for some deviation (controlled by threshold).
    *   `avg_levenshtein_norm`: Normalized Levenshtein edit distance.
*   **General Performance:**
    *   `nDCG@10`: Ranking quality of next-token predictions compared to the original model.
    *   `Perplexity`: Standard language modeling metric (lower is better).

## Outputs

*   **Console:** Detailed logs of retained eigenvectors, compression ratios, and evaluation metrics.
*   **Files:**
    *   Saved edited model checkpoint (if configured).
    *   Results summary in `results/` (e.g., `kfac_hits_{model_size}.jsonl`).
