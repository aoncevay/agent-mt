#!/usr/bin/env python3
"""
Download only the tokenizer files from mT5 model (needed for MetricX-24).

MetricX-24 is based on mT5, so it requires mT5 tokenizer files.

Usage:
    python metrics/download_mt5_tokenizer.py --output_dir ~/user-default-efs/HF_models/mt5-base
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface-hub")
    sys.exit(1)


def download_mt5_tokenizer_files(output_dir: Path, model_id: str = "google/mt5-base"):
    """
    Download only the tokenizer files from an mT5 model.
    
    Args:
        output_dir: Directory where tokenizer files will be saved
        model_id: HuggingFace model ID (default: "google/mt5-base")
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading mT5 tokenizer files for {model_id}...")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Required tokenizer files for mT5
    tokenizer_files = [
        "spiece.model",             # SentencePiece model (REQUIRED) - mT5 uses 'spiece.model'
        "tokenizer_config.json",    # Tokenizer settings (REQUIRED)
        "special_tokens_map.json",  # Special token mappings (REQUIRED)
        "config.json",              # Model config (recommended)
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
        print(f"    - spiece.model (REQUIRED) - mT5 uses 'spiece.model'")
        print(f"    - tokenizer_config.json (REQUIRED)")
        print(f"    - special_tokens_map.json (REQUIRED)")
    
    # Verify critical files
    critical_files = ["sentencepiece.model", "tokenizer_config.json", "special_tokens_map.json"]
    all_critical_present = all((output_dir / f).exists() for f in critical_files)
    
    if all_critical_present:
        print(f"\n✓ All critical tokenizer files are present!")
        print(f"\nYou can now use this path for MetricX-24:")
        print(f"  {output_dir}")
    else:
        print(f"\n⚠ Warning: Some critical files are missing!")
        missing = [f for f in critical_files if not (output_dir / f).exists()]
        print(f"  Missing: {', '.join(missing)}")
        print(f"\nPlease ensure these files are downloaded manually if needed.")


def main():
    parser = argparse.ArgumentParser(
        description="Download tokenizer files from mT5 model (for MetricX-24)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/user-default-efs/HF_models/mt5-base",
        help="Directory to save tokenizer files (default: ~/user-default-efs/HF_models/mt5-base)"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/mt5-base",
        help="HuggingFace model ID (default: google/mt5-base, can also use google/mt5-large)"
    )
    
    args = parser.parse_args()
    
    download_mt5_tokenizer_files(Path(args.output_dir), args.model_id)


if __name__ == "__main__":
    main()

