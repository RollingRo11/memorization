#!/usr/bin/env python
"""
Evaluate a model (base or edited) on GSM8K using lm-evaluation-harness.
"""

import argparse
import json
import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator, tasks
from lm_eval.models.huggingface import HFLM

def load_edited_model(base_model_name, checkpoint_path, device="cuda"):
    """Load base model and apply edited weights."""
    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    
    print(f"Loading edited weights from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # Handle both full state dicts and nested "model_state_dict" keys
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
        
    # Apply weights
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Load result: {msg}")
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model name (e.g. allenai/OLMo-2-1124-7B)")
    parser.add_argument("--checkpoint", default=None, help="Path to edited .pt checkpoint (optional)")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for testing)")
    parser.add_argument("--output-file", default=None, help="Save results to json")
    args = parser.parse_args()

    # Initialize model
    if args.checkpoint:
        # Load edited model manually
        hf_model = load_edited_model(args.model, args.checkpoint, args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        
        # Wrap in HFLM for lm-eval
        lm_obj = HFLM(pretrained=hf_model, tokenizer=tokenizer, batch_size=args.batch_size)
    else:
        # Load standard base model via lm-eval
        print(f"Evaluating BASE model: {args.model}")
        lm_obj = HFLM(pretrained=args.model, device=args.device, batch_size=args.batch_size, dtype="bfloat16", trust_remote_code=True)

    # Run GSM8K (Chain of Thought)
    print("Starting GSM8K (CoT) evaluation...")
    results = evaluator.simple_evaluate(
        model=lm_obj,
        tasks=["gsm8k_cot"],  # Use CoT version
        num_fewshot=5,
        batch_size=args.batch_size,
        limit=args.limit
    )

    # Extract score
    try:
        task_res = results["results"]["gsm8k_cot"]
        
        # Debug: Print keys to see what we got
        # print(f"DEBUG: Available keys: {list(task_res.keys())}")
        
        if "exact_match,strict-match" in task_res:
             score = task_res["exact_match,strict-match"]
        elif "exact_match,flexible-extract" in task_res:
             score = task_res["exact_match,flexible-extract"]
        elif "exact_match,none" in task_res:
            score = task_res["exact_match,none"]
        elif "acc,none" in task_res:
            score = task_res["acc,none"]
        else:
            # Fallback: look for any key with 'acc' or 'match'
            found = False
            for k, v in task_res.items():
                if isinstance(v, (int, float)) and ("acc" in k or "match" in k):
                    score = v
                    print(f"Using fallback metric key: {k}")
                    found = True
                    break
            if not found:
                print(f"Warning: Could not find accuracy key. Available keys: {list(task_res.keys())}")
                score = 0.0
    except Exception as e:
        print(f"Error extracting score: {e}")
        score = 0.0
        
    print(f"\n{'='*40}")
    print(f"GSM8K (CoT) Accuracy: {score:.2%}")
    print(f"{'='*40}\n")
    
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
