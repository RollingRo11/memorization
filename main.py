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
    
    # --- Step 1: Collect General Factors ---
    # Check if we already have factors for the requested blocks
    # Simplistic check: look for any .pt file with the block number
    need_gen = args.force_collect
    if not need_gen:
        # If any block is missing a corresponding file, re-run
        # This is a loose check; a robust one would parse the filenames strictly
        existing = list(gen_dir.glob("*.pt"))
        if not existing:
            need_gen = True
        else:
            print(f"Found {len(existing)} general factor files in {gen_dir}, skipping collection.")

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
    need_math = args.force_collect
    if not need_math:
        existing = list(math_dir.glob("*.pt"))
        if not existing:
            need_math = True
        else:
            print(f"Found {len(existing)} math factor files in {math_dir}, skipping collection.")

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

    # --- Step 3: Eval with Hessian Union ---
    
    # Construct layers-json config
    # e.g. {"31": {"gate": 0.8, "up": 0.8, "down": 0.8}}
    layers_config = {}
    for b in args.target_blocks:
        layers_config[b] = {p: args.variance for p in args.projections}
    layers_json = json.dumps(layers_config)
    
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

    print("\n✓ Pipeline completed successfully.")

if __name__ == "__main__":
    main()