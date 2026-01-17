#!/usr/bin/env python3
"""
Upload a nanochat checkpoint to Hugging Face Hub.

Usage:
    python -m scripts.upload_to_hf --checkpoint_dir=base_checkpoints/d12 --step=5000 --repo_id=tomzhengy/model-name
"""
import os
import sys
import json
import argparse
import torch
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload checkpoint to HF Hub")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to checkpoint directory")
    parser.add_argument("--step", type=int, required=True, help="Checkpoint step number")
    parser.add_argument("--repo_id", type=str, required=True, help="HF repo id (e.g., username/model-name)")
    parser.add_argument("--private", action="store_true", help="Make the repo private")
    args = parser.parse_args()

    # paths
    model_path = os.path.join(args.checkpoint_dir, f"model_{args.step:06d}.pt")
    meta_path = os.path.join(args.checkpoint_dir, f"meta_{args.step:06d}.json")

    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(meta_path):
        print(f"Metadata not found: {meta_path}")
        sys.exit(1)

    # load metadata
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"Uploading checkpoint from step {args.step}")
    print(f"  Model config: {meta.get('model_config', {})}")
    print(f"  Val BPB: {meta.get('val_bpb', 'N/A')}")

    # create repo if it doesn't exist
    api = HfApi()
    try:
        create_repo(args.repo_id, private=args.private, exist_ok=True)
        print(f"Created/verified repo: {args.repo_id}")
    except Exception as e:
        print(f"Warning: Could not create repo: {e}")

    # upload files
    print(f"Uploading {model_path}...")
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=f"model_{args.step:06d}.pt",
        repo_id=args.repo_id,
    )

    print(f"Uploading {meta_path}...")
    api.upload_file(
        path_or_fileobj=meta_path,
        path_in_repo=f"meta_{args.step:06d}.json",
        repo_id=args.repo_id,
    )

    # create a README with model info
    readme_content = f"""# mHC-GPT Checkpoint

Trained with nanochat-mHC (multi-head communication).

## Model Config
```json
{json.dumps(meta.get('model_config', {}), indent=2)}
```

## Training Config
```json
{json.dumps(meta.get('user_config', {}), indent=2)}
```

## Results
- Step: {args.step}
- Val BPB: {meta.get('val_bpb', 'N/A')}

## Usage
```python
from nanochat.checkpoint_manager import build_model

model, tokenizer = build_model("path/to/checkpoint", step={args.step}, device="cuda", phase="inference")
```
"""

    readme_path = "/tmp/README.md"
    with open(readme_path, "w") as f:
        f.write(readme_content)

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=args.repo_id,
    )

    print(f"\nDone! Model uploaded to: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
