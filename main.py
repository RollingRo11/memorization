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
    parser.add_argument("--run-gsm8k", action="store_true", help="Run 3-way GSM8K comparison (Base vs Gen vs Union)")
    
    args = parser.parse_args()
    
    # Setup paths
    base_dir = Path(args.output_dir)
    model_safe = args.model.replace("/", "__")
    
    # We create specific subdirs for general vs math factors
    # e.g. data/kfac_factors/allenai__OLMo-2-1124-7B_general
    gen_dir = base_dir / f"{model_safe}_general"
    math_dir = base_dir / f"{model_safe}_math"
    
    gen_dir.mkdir(parents=True, exist_ok=True)
    math_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Helper to define layers config ---
    def get_layer_json(variance):
        cfg = {}
        for b in args.target_blocks:
            cfg[b] = {p: variance for p in args.projections}
        return json.dumps(cfg)
    
    layers_json = get_layer_json(args.variance)

    # --- Step 1: Collect General Factors ---
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

    # --- Step 2: Collect Math Factors ---
    missing_math = check_missing_blocks(math_dir, target_blocks_int)
    need_math = args.force_collect or bool(missing_math)
    
    if not need_math:
        print(f"Found factors for all requested blocks in {math_dir}, skipping collection.")
    else:
        if missing_math and not args.force_collect:
            print(f"Missing math factors for blocks: {missing_math}. Starting collection...")

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
        run_command(cmd_math, "Collect Math Factors")

    # --- Step 3: Eval with Hessian Union (The "Union" Model) ---
    
    cmd_eval = [
        sys.executable, "evaluations/eval_mem_kfac.py",
        "--model-size", args.model_size,
        "--layers-json", layers_json,
        "--general-factors-dir", str(gen_dir),
        "--math-factors-path", str(math_dir),
        "--alpha", str(args.alpha),
        "--use-cache"
    ]
    
    run_command(cmd_eval, f"Apply Hessian Union (alpha={args.alpha}) & Evaluate")

    # --- Step 4: Optional GSM8K Comparison ---
    if args.run_gsm8k:
        print("\n" + "="*60)
        print("STARTING GSM8K 3-WAY COMPARISON")
        print("="*60 + "\n")

        # Define model paths
        # Note: eval_mem_kfac saves models to data/models_kfac/{size}/{model_name}/{layer_cfg}/
        # We need to predictably find them. The layer_cfg tag is non-trivial to reconstruct perfectly
        # so we will look for the most recently modified .pt file in the expected dir.
        
        models_root = Path("data/models_kfac") / args.model_size / model_safe
        
        def find_latest_checkpoint():
            # Find the latest .pt file in the models directory (recursive)
            pts = list(models_root.rglob(f"{model_safe}.pt"))
            if not pts: return None
            return max(pts, key=lambda f: f.stat().st_mtime)

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
        
        ckpt_gen = find_latest_checkpoint()
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
        # We need to re-run eval_mem_kfac just to touch the file or we rely on finding it?
        # Better to re-generate it to be sure we pick up the right one, 
        # or we could have captured it from Step 3. 
        # For simplicity, we assume Step 3 was the LAST run, so find_latest works.
        # But wait! We just ran Gen-Only, so THAT is the latest.
        # So we actually need to re-run the Union generation (Step 3 logic) briefly or 
        # ideally we should have saved the path. 
        
        # Let's just re-run the Union application (it's fast with cache) to ensure it's the latest file
        print("Restoring Union Model...")
        run_command(cmd_eval + ["--skip-baseline", "--results-tag", "union_temp"], "Restore Union Model")
        
        ckpt_union = find_latest_checkpoint()
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