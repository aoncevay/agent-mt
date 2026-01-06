#!/usr/bin/env python3
"""
Download only the tokenizer files from xlm-roberta-large model.

This script downloads the minimal files needed for the XLM-RoBERTa tokenizer
to work with COMET, without downloading the full 11GB model.

Usage:
    python metrics/download_xlmr_tokenizer.py --output_dir ~/user-default-efs/HF_models/xlm-roberta-large
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface-hub")
    sys.exit(1)


def download_tokenizer_files(output_dir: Path, model_id: str = "xlm-roberta-large"):
    """
    Download only the tokenizer files from a HuggingFace model.
    
    Args:
        output_dir: Directory where tokenizer files will be saved
        model_id: HuggingFace model ID (default: "xlm-roberta-large")
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading tokenizer files for {model_id}...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Required tokenizer files for XLM-RoBERTa
    tokenizer_files = [
        "tokenizer.json",           # Fast tokenizer configuration
        "tokenizer_config.json",    # Tokenizer settings
        "sentencepiece.bpe.model",  # SentencePiece BPE model (required!)
        "special_tokens_map.json",  # Special token mappings (optional but recommended)
        "config.json",              # Model config (optional but recommended)
    ]
    
    downloaded_files = []
    failed_files = []
    
    for filename in tokenizer_files:
        try:
            print(f"\nDownloading {filename}...")
            file_path = hf_hub_download(
                repo_id=model_id,
                filename=filename,
                local_dir=str(output_dir),
                local_dir_use_symlinks=False,  # Copy files, don't symlink
            )
            downloaded_files.append(filename)
            print(f"  ✓ Saved to: {file_path}")
        except Exception as e:
            print(f"  ✗ Failed to download {filename}: {e}")
            failed_files.append(filename)
    
    print("\n" + "=" * 60)
    print("Download Summary:")
    print(f"  ✓ Successfully downloaded: {len(downloaded_files)}/{len(tokenizer_files)} files")
    
    if downloaded_files:
        print(f"\n  Downloaded files:")
        for f in downloaded_files:
            print(f"    - {f}")
    
    if failed_files:
        print(f"\n  ⚠ Failed files:")
        for f in failed_files:
            print(f"    - {f}")
        print(f"\n  Note: Some files may not be critical. The minimum required files are:")
        print(f"    - sentencepiece.bpe.model (REQUIRED)")
        print(f"    - tokenizer.json (REQUIRED for fast tokenizer)")
        print(f"    - tokenizer_config.json (REQUIRED)")
    
    # Verify critical files
    critical_files = ["sentencepiece.bpe.model", "tokenizer.json", "tokenizer_config.json"]
    all_critical_present = all((output_dir / f).exists() for f in critical_files)
    
    if all_critical_present:
        print(f"\n✓ All critical tokenizer files are present!")
        print(f"\nYou can now use this path in COMET:")
        print(f"  {output_dir}")
    else:
        print(f"\n⚠ Warning: Some critical files are missing!")
        missing = [f for f in critical_files if not (output_dir / f).exists()]
        print(f"  Missing: {', '.join(missing)}")
        print(f"\nPlease ensure these files are downloaded manually if needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Download tokenizer files from xlm-roberta-large model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/user-default-efs/HF_models/xlm-roberta-large",
        help="Directory to save tokenizer files (default: ~/user-default-efs/HF_models/xlm-roberta-large)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="xlm-roberta-large",
        help="HuggingFace model ID (default: xlm-roberta-large)"
    )
    
    args = parser.parse_args()
    
    download_tokenizer_files(Path(args.output_dir), args.model_id)


if __name__ == "__main__":
    main()

