# K-FAC curvature edit

Reproduce the K-FAC treatment (used in the paper) and **Hessian Union** strategy.

- **Paper:** [From Memorization to Reasoning in the Spectrum of Loss Curvature](https://arxiv.org/abs/2510.24256)
- **Blog:** [Understanding Memorization via Loss Curvature](https://goodfire-staging.webflow.io/research/understanding-memorization-via-loss-curvature)
- **Scope:** generate K-FAC factors A, G, compute KFAC edit (single-corpus or mixed-Hessian union) and eval.

## TL;DR
Compute A = E[aa^T] (pre-activation inputs) and G = E[gg^T] (pre-activation gradients) per MLP projection, decompose them, and keep only the top curvature mass when editing each weight W. This suppresses rote recitation while preserving shared structure.

The **Hessian Union** extension collects factors on both general and domain-specific data (e.g., math), then combines them so that domain-critical circuits are not accidentally pruned.

## Why Hessians detect memorization

The Hessian (second derivative of the loss) measures curvature across weight perturbations. K-FAC approximates it efficiently via the Kronecker product of two smaller matrices:

```
F_W ≈ G ⊗ A
```

Eigendecomposition of A and G yields per-component curvature scores **Π_ij = λ_i · μ_j** (products of the activation and gradient eigenvalues). Intuitively:

- **High-curvature (sharp) directions** — large eigenvalues — correspond to circuits that are exercised consistently across many diverse inputs. Perturbing them changes the loss everywhere, which means they encode *generalizable* knowledge.
- **Low-curvature (flat) directions** — small eigenvalues — barely affect the overall loss. They store information that matters for only a handful of examples: rote memorization of isolated sequences.

By projecting weights onto the top-curvature subspace (retaining mass ρ) we keep the generalizable circuitry and discard memorized artifacts.

### The domain-dilution problem and Hessian Union

When the Hessian is estimated on a broad general corpus, circuits that matter for a *specific* domain (e.g., math reasoning) may appear low-curvature simply because that domain is a small fraction of the data. Pruning those directions hurts domain performance.

The fix is to compute a **separate domain Hessian** and mix:

```
S_final = S_general + α · S_domain
```

This "Hessian Union" boosts the curvature scores of domain-critical components, preventing them from being pruned while still suppressing rote memorization.

## Requirements

- Python 3.10+
- PyTorch (CUDA recommended)
- NumPy
- `transformers`, `datasets`, `accelerate`

```bash
uv pip install -r requirements.txt
# OR
pip install torch transformers datasets accelerate numpy tqdm
```

## Usage

### Quick: full Hessian Union pipeline

`main.py` runs the entire pipeline in one command — collecting general factors, domain factors, applying the mixed edit, and evaluating:

```bash
python main.py \
  --model "allenai/OLMo-2-1124-7B" \
  --general-corpus dolmo \
  --math-corpus ProCreations/SimpleMath \
  --alpha 1.0 \
  --target-blocks 23 24 25 \
  --variance 0.6
```

| Flag | Purpose |
|---|---|
| `--alpha` | Mixing weight for domain factors (0 = general-only, 1 = equal mix) |
| `--variance` | Curvature mass ρ to retain ∈ [0, 1] |
| `--target-blocks` | Transformer block indices to edit |
| `--run-gsm8k` | Run a 3-way GSM8K comparison (Base vs General-only vs Union) |

### Step-by-step

#### 1. Collect K-FAC factors
```bash
# General corpus
python data/collect_kfac_multilayer.py \
  --model "allenai/OLMo-2-1124-7B" \
  --corpus dolmo \
  --target_blocks 28 29 30 31 \
  --save_dir data/kfac_factors/olmo2-7b_general

# Domain corpus (e.g., math)
python data/collect_kfac_multilayer.py \
  --model "allenai/OLMo-2-1124-7B" \
  --corpus ProCreations/SimpleMath \
  --target_blocks 28 29 30 31 \
  --save_dir data/kfac_factors/olmo2-7b_math
```

#### 2. Apply edit & evaluate
```bash
python evaluations/eval_mem_kfac.py \
  --model-size 7b \
  --layers-json '{"31": {"gate": 0.8, "up": 0.8, "down": 0.8}}' \
  --general-factors-dir data/kfac_factors/olmo2-7b_general \
  --math-factors-path data/kfac_factors/olmo2-7b_math \
  --alpha 1.0 \
  --use-cache
```

## Outputs

- Printed/saved metrics from the evaluator (memorization accuracy, perplexity, nDCG).
- Edited model checkpoint saved under `edited_models/`.
- Results summary in `results/` (e.g., `kfac_hits_{model_size}.jsonl`).

## Citation

If this code helps your work, please cite the paper:
**OpenReview:** https://openreview.net/pdf?id=MzRDxPUmgK
