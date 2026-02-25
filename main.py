#!/usr/bin/env python
"""
Memorization Project Pipeline

One-command runner for the "Hessian Union" strategy:
1. Collect General K-FAC Factors (if missing)
2. Collect Math K-FAC Factors (if missing)
3. Apply Mixed-Hessian Edit & Evaluate

Usage:
    python main.py \
        --model "allenai/OLMo-2-1124-7B" \
        --general-corpus dolmo \
        --math-corpus ./data/SimpleMath.jsonl \
        --alpha 1.0 \
        --target-blocks 31 \
        --variance 0.8
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
import shutil

def run_command(cmd, desc):
    """Run a shell command with logging."""
    print(f"\n{'='*60}")
    print(f"STEP: {desc}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{ '='*60}\n")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run the full Hessian Union pipeline.")
    
    # Model config
    parser.add_argument("--model", default="allenai/OLMo-2-1124-7B", help="Model HF path")
    parser.add_argument("--model-size", default="7b", choices=["1b", "7b"], help="Short size code for eval script")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    # Data config
    parser.add_argument("--general-corpus", default="dolmo", help="General dataset (olmo/dolmo or path)")
    parser.add_argument("--math-corpus", default="ProCreations/SimpleMath", help="HF dataset name or path for math domain")
    
    # Pipeline config
    parser.add_argument("--output-dir", default="data/kfac_factors", help="Base dir for factors")
    parser.add_argument("--target-blocks", nargs="+", default=["23", "24", "25"], help="List of block indices to process")
    parser.add_argument("--projections", nargs="+", default=["gate", "up"], help="MLP projections to edit (gate/up/down)")
    parser.add_argument("--layers-per-pass", default="4", help="Layers to collect per pass")
    parser.add_argument("--force-collect", action="store_true", help="Force re-collection of factors")
    
    # Mixing / Edit config
    parser.add_argument("--alpha", type=float, default=1.0, help="Mixing strength for math factors")
    parser.add_argument("--variance", type=float, default=0.6, help="Curvature mass to retain (rho)")
    parser.add_argument("--ablate", action="store_true",
                        help="Ablation mode: REMOVE the top curvature directions from the domain corpus "
                             "instead of protecting them. Use to unlearn capabilities (e.g. eval awareness).")
    parser.add_argument("--run-gsm8k", action="store_true", help="Run 3-way GSM8K comparison (Base vs Gen vs Union)")
    parser.add_argument("--skip-evals", action="store_true", help="Skip all evaluations (just apply K-FAC and save model)")
    parser.add_argument("--refresh-cache", action="store_true", help="Force refresh of K-FAC weight cache")
    parser.add_argument("--eval-union-only", action="store_true", help="For GSM8K: Skip Base and General-Only models, evaluate only Union")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.output_dir)
    model_safe = args.model.replace("/", "__")
    
    # We create specific subdirs for general vs math factors
    # e.g. data/kfac_factors/allenai__OLMo-2-1124-7B_general
    gen_dir = base_dir / f"{model_safe}_general"
    
    # Make math dir specific to the corpus used
    math_corpus_safe = args.math_corpus.replace("/", "__").replace(".", "_")
    math_dir = base_dir / f"{model_safe}_{math_corpus_safe}"
    
    gen_dir.mkdir(parents=True, exist_ok=True)
    math_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Helper to define layers config ---
    def get_layer_json(variance):
        cfg = {}
        for b in args.target_blocks:
            cfg[b] = {p: variance for p in args.projections}
        return json.dumps(cfg)
    
    layers_json = get_layer_json(args.variance)

    # --- Factor Collection ---
    # Robust check: Ensure ALL requested blocks are present in the existing files
    target_blocks_int = set(int(b) for b in args.target_blocks)

    def check_missing_blocks(directory, required_blocks):
        if not directory.exists():
            return required_blocks

        covered_blocks = set()
        for f in directory.glob("*.pt"):
            try:
                # Filename format: kfac_factors_blk_28_29_30_31.pt
                parts = f.stem.split("blk_")[-1].split("_")
                for p in parts:
                    if p.isdigit():
                        covered_blocks.add(int(p))
            except Exception:
                continue

        return required_blocks - covered_blocks

    # --- Step 1: Collect General Factors (skip in ablate mode) ---
    if not args.ablate:
        missing_gen = check_missing_blocks(gen_dir, target_blocks_int)
        need_gen = args.force_collect or bool(missing_gen)

        if not need_gen:
            print(f"Found factors for all requested blocks in {gen_dir}, skipping collection.")
        else:
            if missing_gen and not args.force_collect:
                print(f"Missing general factors for blocks: {missing_gen}. Starting collection...")

        if need_gen:
            cmd_gen = [
                sys.executable, "data/collect_kfac_multilayer.py",
                "--model", args.model,
                "--device", args.device,
                "--corpus", args.general_corpus,
                "--save_dir", str(gen_dir),
                "--target_blocks", *args.target_blocks,
                "--layers_per_pass", args.layers_per_pass
            ]
            run_command(cmd_gen, "Collect General Factors")
    else:
        print("Ablate mode: skipping general factor collection.")

    # --- Step 2: Collect Domain Factors ---
    missing_math = check_missing_blocks(math_dir, target_blocks_int)
    need_math = args.force_collect or bool(missing_math)

    if not need_math:
        print(f"Found factors for all requested blocks in {math_dir}, skipping collection.")
    else:
        if missing_math and not args.force_collect:
            corpus_label = "domain" if args.ablate else "math"
            print(f"Missing {corpus_label} factors for blocks: {missing_math}. Starting collection...")

    if need_math:
        cmd_math = [
            sys.executable, "data/collect_kfac_multilayer.py",
            "--model", args.model,
            "--device", args.device,
            "--corpus", args.math_corpus,
            "--save_dir", str(math_dir),
            "--target_blocks", *args.target_blocks,
            "--layers_per_pass", args.layers_per_pass
        ]
        step_label = "Collect Domain Factors (for ablation)" if args.ablate else "Collect Math Factors"
        run_command(cmd_math, step_label)

    # --- Step 3: Eval with K-FAC Edit ---

    if args.ablate:
        # Ablation mode: use domain factors as primary, subtract top curvature directions
        cmd_eval = [
            sys.executable, "evaluations/eval_mem_kfac.py",
            "--model-size", args.model_size,
            "--layers-json", layers_json,
            "--general-factors-dir", str(math_dir),  # domain factors as primary
            "--ablate",
            "--use-cache"
        ]
    else:
        # Union mode: combine general + domain factors
        cmd_eval = [
            sys.executable, "evaluations/eval_mem_kfac.py",
            "--model-size", args.model_size,
            "--layers-json", layers_json,
            "--general-factors-dir", str(gen_dir),
            "--math-factors-path", str(math_dir),
            "--alpha", str(args.alpha),
            "--use-cache"
        ]

    if args.refresh_cache:
        cmd_eval.append("--refresh-cache")

    step_desc = "Ablate domain curvature" if args.ablate else f"Apply Hessian Union (alpha={args.alpha})"
    run_command(cmd_eval, f"{step_desc} & Evaluate")

    # --- Step 4: Optional GSM8K Comparison ---
    if args.run_gsm8k:
        print("\n" + "="*60)
        print("STARTING GSM8K 3-WAY COMPARISON")
        print("="*60 + "\n")

        # Define model paths
        # Note: eval_mem_kfac saves models to edited_models/models_kfac/{size}/{model_name}/{layer_cfg}/
        models_root = Path("edited_models/models_kfac") / args.model_size / model_safe
        
        def find_checkpoint(target_alpha):
            # Look for directories ending with _alpha{target_alpha}
            alpha_tag = f"_alpha{target_alpha}"
            # The structure is models_kfac/{size}/{model}/{layer_tag}/model.pt
            # We iterate over subdirs of models_root
            if not models_root.exists():
                return None
            
            # Find dir matching alpha
            candidates = []
            for d in models_root.iterdir():
                if d.is_dir() and str(d).endswith(str(target_alpha)): # Simple suffix check might be risky with floats like 1.0 vs 0.0
                    # Better check: check if alpha_tag is in the name
                    if alpha_tag in d.name:
                        candidates.append(d)
            
            if not candidates:
                return None
            
            # If multiple (maybe from different variance runs), pick the latest
            best_dir = max(candidates, key=lambda d: d.stat().st_mtime)
            return best_dir / f"{model_safe}.pt"

        if not args.eval_union_only:
            # 1. Base Model
            run_command([
                sys.executable, "evaluations/eval_gsm8k.py",
                "--model", args.model,
                "--device", args.device
            ], "Eval Base Model on GSM8K")

            # 2. General-Only Model (Alpha=0.0)
            # We need to run eval_mem_kfac specifically to generate this checkpoint
            # We use --skip-baseline to speed it up
            print("Generating General-Only (Alpha=0) Model...")
            cmd_gen_only = [
                sys.executable, "evaluations/eval_mem_kfac.py",
                "--model-size", args.model_size,
                "--layers-json", layers_json,
                "--general-factors-dir", str(gen_dir),
                "--math-factors-path", str(math_dir),
                "--alpha", "0.0",
                "--use-cache",
                "--skip-baseline"
            ]
            # We define a custom results tag so we don't pollute main logs
            cmd_gen_only += ["--results-tag", "gen_only_temp"]
            run_command(cmd_gen_only, "Generate General-Only Model")
            
            ckpt_gen = find_checkpoint(0.0)
            if ckpt_gen:
                run_command([
                    sys.executable, "evaluations/eval_gsm8k.py",
                    "--model", args.model,
                    "--checkpoint", str(ckpt_gen),
                    "--device", args.device
                ], "Eval General-Only Model on GSM8K")
            else:
                print("Error: Could not find General-Only checkpoint.")

        # 3. Union Model (Already generated in Step 3)
        # We assume Step 3 ran with args.alpha.
        ckpt_union = find_checkpoint(args.alpha)
        if ckpt_union:
             run_command([
                sys.executable, "evaluations/eval_gsm8k.py",
                "--model", args.model,
                "--checkpoint", str(ckpt_union),
                "--device", args.device
            ], f"Eval Union Model (alpha={args.alpha}) on GSM8K")
        else:
             print("Error: Could not find Union checkpoint.")

    print("\n✓ Pipeline completed successfully.")

if __name__ == "__main__":
    main()