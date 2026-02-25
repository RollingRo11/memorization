#!/usr/bin/env python
"""
Download and prepare the jjpn2/eval_awareness dataset for K-FAC factor collection.

Produces a JSONL file with flattened conversation text from eval-labeled transcripts,
suitable for use as a corpus in collect_kfac_multilayer.py.

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login          # must accept gated access at
                                   # https://huggingface.co/datasets/jjpn2/eval_awareness

Usage:
    python data/prepare_eval_awareness.py
    python data/prepare_eval_awareness.py --include-real   # also include organic conversations
    python data/prepare_eval_awareness.py --out data/eval_awareness_all.jsonl --include-real
"""

import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO_ID = "jjpn2/eval_awareness"
DEFAULT_OUT = Path(__file__).parent / "eval_awareness.jsonl"


# ------------------------------------------------------------------
# Conversation → flat text
# ------------------------------------------------------------------
def flatten_input(inp) -> str:
    """Convert an inspect_ai-style input (str or list of messages) to flat text."""
    if isinstance(inp, str):
        return inp.strip()

    if isinstance(inp, list):
        parts = []
        for msg in inp:
            if isinstance(msg, dict):
                role = msg.get("role", "unknown")
                # content can be a string or a list of content blocks
                content = msg.get("content", "")
                if isinstance(content, list):
                    # content blocks: [{"type": "text", "text": "..."}, ...]
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            text_parts.append(block.get("text", str(block)))
                        else:
                            text_parts.append(str(block))
                    content = " ".join(text_parts)
                parts.append(f"{role}: {content}")
            elif isinstance(msg, str):
                parts.append(msg)
        return "\n".join(parts)

    return str(inp).strip()


# ------------------------------------------------------------------
# Download dataset from HuggingFace
# ------------------------------------------------------------------
def download_dataset(cache_dir: Path) -> Path:
    """Download dataset.zip from HF and return path to extracted dataset.json."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading dataset.zip from {REPO_ID} ...")
    zip_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="dataset.zip",
        repo_type="dataset",
        local_dir=str(cache_dir),
    )
    zip_path = Path(zip_path)
    print(f"  Downloaded to {zip_path}")

    # Try unzipping — might need a password
    extract_dir = cache_dir / "extracted"
    extract_dir.mkdir(exist_ok=True)

    # First try without password (some gated datasets just gate access, not encrypt)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        print(f"  Extracted to {extract_dir}")
    except RuntimeError as e:
        if "password" in str(e).lower() or "encrypted" in str(e).lower():
            # Try the decrypt.sh approach — download and run it
            print("  ZIP is password-protected, trying decrypt script...")
            try:
                decrypt_script = hf_hub_download(
                    repo_id=REPO_ID,
                    filename="scripts/decrypt.sh",
                    repo_type="dataset",
                    local_dir=str(cache_dir),
                )
                subprocess.run(
                    ["bash", decrypt_script],
                    cwd=str(cache_dir),
                    check=True,
                )
            except Exception as e2:
                print(f"  Auto-decrypt failed: {e2}")
                print("\n  Manual steps:")
                print(f"    1. cd {cache_dir}")
                print(f"    2. Check scripts/decrypt.sh for the password")
                print(f"    3. unzip -P <password> dataset.zip -d extracted/")
                print(f"    4. Re-run this script with --json {extract_dir / 'dataset.json'}")
                sys.exit(1)
        else:
            raise

    # Find dataset.json
    candidates = list(extract_dir.rglob("dataset.json"))
    if not candidates:
        # Maybe it extracted directly into cache_dir
        candidates = list(cache_dir.rglob("dataset.json"))
    if not candidates:
        print(f"ERROR: Could not find dataset.json in {extract_dir}")
        print(f"  Contents: {list(extract_dir.rglob('*'))}")
        sys.exit(1)

    return candidates[0]


# ------------------------------------------------------------------
# Process dataset.json → JSONL
# ------------------------------------------------------------------
def process_dataset(json_path: Path, out_path: Path, include_real: bool):
    """Filter and flatten conversations into a JSONL file."""
    print(f"Loading {json_path} ...")
    with open(json_path) as f:
        data = json.load(f)

    # Handle both list-of-samples and dict-with-samples-key
    if isinstance(data, dict):
        # Try common keys
        for key in ["samples", "data", "dataset", "items", "records"]:
            if key in data:
                data = data[key]
                break
        else:
            # Maybe it's a dict of id→sample
            if all(isinstance(v, dict) for v in data.values()):
                data = list(data.values())
            else:
                print(f"ERROR: Unexpected top-level structure. Keys: {list(data.keys())[:10]}")
                sys.exit(1)

    print(f"  Total entries: {len(data)}")

    kept = 0
    skipped = 0
    with open(out_path, "w") as out:
        for entry in data:
            meta = entry.get("metadata", {})

            # Filter: keep eval-category entries (and optionally real ones)
            is_eval = meta.get("eval_category", False)
            if not include_real and not is_eval:
                skipped += 1
                continue

            # Extract and flatten the conversation
            inp = entry.get("input", "")
            text = flatten_input(inp)

            if not text or len(text) < 20:
                skipped += 1
                continue

            # Write as JSONL with text field (compatible with collect_kfac_multilayer.py)
            record = {"text": text}

            # Preserve useful metadata
            if "id" in entry:
                record["id"] = entry["id"]
            if meta.get("category"):
                record["category"] = meta["category"]
            record["eval_category"] = is_eval

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    print(f"  Kept: {kept}  |  Skipped: {skipped}")
    print(f"  Output: {out_path}")
    return kept


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prepare eval_awareness dataset for K-FAC.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output JSONL path")
    parser.add_argument("--json", type=Path, default=None,
                        help="Path to already-extracted dataset.json (skip download)")
    parser.add_argument("--include-real", action="store_true",
                        help="Include organic/real conversations too (default: eval-only)")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Cache directory for downloads")
    args = parser.parse_args()

    if args.json and args.json.exists():
        json_path = args.json
    else:
        cache = args.cache_dir or Path(tempfile.mkdtemp(prefix="eval_awareness_"))
        json_path = download_dataset(cache)

    n = process_dataset(json_path, args.out, args.include_real)

    if n > 0:
        print(f"\nReady! Use with the pipeline:")
        print(f"  python main.py \\")
        print(f"    --general-corpus dolmo \\")
        print(f"    --math-corpus {args.out} \\")
        print(f"    --alpha 1.0 \\")
        print(f"    --target-blocks 23 24 25")
    else:
        print("\nWARNING: No entries written. Check dataset access / format.")


if __name__ == "__main__":
    main()
