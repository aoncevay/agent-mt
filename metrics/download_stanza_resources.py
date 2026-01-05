#!/usr/bin/env python3
"""
Download Stanza language models for offline use.

This script downloads the required Stanza models to a local directory
so they can be used in environments without internet access (e.g., SageMaker).

Usage:
    python metrics/download_stanza_resources.py --output_dir ~/stanza_resources
    python metrics/download_stanza_resources.py --output_dir ~/stanza_resources --languages en zh es de fr it
"""

import argparse
import stanza
from pathlib import Path


def download_stanza_models(languages: list, output_dir: Path):
    """
    Download Stanza models for specified languages.
    
    Args:
        languages: List of language codes (e.g., ['en', 'zh', 'es', 'de', 'fr', 'it'])
        output_dir: Directory to save Stanza resources
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Stanza models to: {output_dir}")
    print(f"Languages: {', '.join(languages)}")
    print()
    
    # Set STANZA_RESOURCES_DIR environment variable
    import os
    os.environ['STANZA_RESOURCES_DIR'] = str(output_dir)
    
    # Map language codes to Stanza language codes
    # Note: Stanza uses 'zh-hant' for Traditional Chinese, but 'zh' also works
    lang_map = {
        'zht': 'zh',  # Traditional Chinese -> use 'zh' (Stanza doesn't distinguish)
        'zh': 'zh',   # Chinese
    }
    
    for lang in languages:
        # Map to Stanza language code if needed
        stanza_lang = lang_map.get(lang, lang)
        
        print(f"Downloading {lang} model (Stanza code: {stanza_lang})...")
        try:
            # Download the model
            # Stanza will automatically download to STANZA_RESOURCES_DIR
            stanza.download(stanza_lang, processors='tokenize,lemma', model_dir=str(output_dir))
            print(f"  ✓ {lang} model downloaded successfully")
        except Exception as e:
            print(f"  ✗ Error downloading {lang} model: {e}")
        print()
    
    print(f"All models downloaded to: {output_dir}")
    print()
    print("To use these models offline, set the environment variable:")
    print(f"  export STANZA_RESOURCES_DIR={output_dir}")
    print()
    print("Or in Python, before importing stanza:")
    print(f"  import os")
    print(f"  os.environ['STANZA_RESOURCES_DIR'] = '{output_dir}'")


def main():
    parser = argparse.ArgumentParser(
        description="Download Stanza language models for offline use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/stanza_resources",
        help="Directory to save Stanza resources (default: ~/stanza_resources)"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en", "zh", "es", "de", "fr", "it"],
        help="Language codes to download (default: en zh es de fr it)"
    )
    
    args = parser.parse_args()
    
    download_stanza_models(args.languages, args.output_dir)


if __name__ == "__main__":
    main()

