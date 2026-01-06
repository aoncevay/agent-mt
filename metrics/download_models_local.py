#!/usr/bin/env python3
"""
Download Stanza and spaCy models locally for offline use.

This script downloads models to local directories that can be zipped and
committed to the repository for use in SageMaker.

Usage:
    python metrics/download_models_local.py --output_dir metrics/models
    python metrics/download_models_local.py --output_dir metrics/models --spacy-only
    python metrics/download_models_local.py --output_dir metrics/models --stanza-only
"""

import argparse
import subprocess
import sys
from pathlib import Path


def download_spacy_models(languages: list, output_dir: Path):
    """
    Download spaCy models using python -m spacy download.
    
    Args:
        languages: List of language codes
        output_dir: Directory to save models
    """
    spacy_model_map = {
        'en': 'en_core_web_sm',
        'de': 'de_core_news_sm',
        'es': 'es_core_news_sm',
        'fr': 'fr_core_news_sm',
        'it': 'it_core_news_sm',
        'zh': 'zh_core_web_sm',
        'zht': 'zh_core_web_sm',
    }
    
    print("="*60)
    print("Downloading spaCy models...")
    print("="*60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    spacy_dir = output_dir / "spacy"
    spacy_dir.mkdir(exist_ok=True)
    
    downloaded = []
    failed = []
    
    for lang in languages:
        model_name = spacy_model_map.get(lang, None)
        if not model_name:
            print(f"  ⚠ No spaCy model available for {lang}, skipping")
            continue
        
        print(f"\nDownloading {lang} model ({model_name})...")
        try:
            # Download to default location first
            result = subprocess.run(
                [sys.executable, '-m', 'spacy', 'download', model_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Find where it was installed
            find_result = subprocess.run(
                [sys.executable, '-c', f'import spacy.util; print(spacy.util.find_model("{model_name}"))'],
                capture_output=True,
                text=True,
                check=True
            )
            model_path = Path(find_result.stdout.strip())
            
            if model_path.exists():
                # Copy to our output directory
                import shutil
                dest_path = spacy_dir / model_name
                if dest_path.exists():
                    shutil.rmtree(dest_path)
                shutil.copytree(model_path, dest_path)
                print(f"  ✓ {lang} model copied to {dest_path}")
                downloaded.append((lang, model_name, dest_path))
            else:
                print(f"  ⚠ Model downloaded but path not found: {model_path}")
                failed.append((lang, model_name))
                
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error downloading {lang} model: {e}")
            if e.stderr:
                print(f"    stderr: {e.stderr[:200]}")
            failed.append((lang, model_name))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append((lang, model_name))
    
    print("\n" + "="*60)
    print("spaCy Download Summary:")
    print(f"  Successfully downloaded: {len(downloaded)}")
    print(f"  Failed: {len(failed)}")
    
    if downloaded:
        print("\nDownloaded models:")
        for lang, model, path in downloaded:
            print(f"  - {lang}: {model} -> {path}")
    
    return len(downloaded), len(failed)


def download_stanza_models(languages: list, output_dir: Path):
    """
    Download Stanza models using stanza.download().
    
    Args:
        languages: List of language codes
        output_dir: Directory to save models
    """
    print("\n" + "="*60)
    print("Downloading Stanza models...")
    print("="*60)
    print("\n⚠ Note: Stanza requires Python < 3.14 due to networkx compatibility")
    print("   If you get errors, try with Python 3.11 or 3.12\n")
    
    try:
        import stanza
    except ImportError:
        print("  ✗ Stanza not installed. Install with: pip install stanza")
        return 0, len(languages)
    except Exception as e:
        print(f"  ✗ Error importing stanza: {e}")
        print("  ⚠ This might be a Python version compatibility issue")
        print("  ⚠ Try with Python 3.11 or 3.12")
        return 0, len(languages)
    
    stanza_dir = output_dir / "stanza"
    stanza_dir.mkdir(exist_ok=True)
    
    # Set STANZA_RESOURCES_DIR to our output directory
    import os
    os.environ['STANZA_RESOURCES_DIR'] = str(stanza_dir.resolve())
    
    lang_map = {
        'zht': 'zh',
        'zh': 'zh',
    }
    
    downloaded = []
    failed = []
    
    for lang in languages:
        stanza_lang = lang_map.get(lang, lang)
        
        print(f"\nDownloading {lang} model (Stanza code: {stanza_lang})...")
        try:
            stanza.download(stanza_lang, processors='tokenize,lemma', model_dir=str(stanza_dir))
            print(f"  ✓ {lang} model downloaded to {stanza_dir}")
            downloaded.append((lang, stanza_lang))
        except Exception as e:
            print(f"  ✗ Error downloading {lang} model: {e}")
            failed.append((lang, stanza_lang))
    
    print("\n" + "="*60)
    print("Stanza Download Summary:")
    print(f"  Successfully downloaded: {len(downloaded)}")
    print(f"  Failed: {len(failed)}")
    
    if downloaded:
        print("\nDownloaded models:")
        for lang, stanza_lang in downloaded:
            print(f"  - {lang}: {stanza_lang}")
    
    return len(downloaded), len(failed)


def main():
    parser = argparse.ArgumentParser(
        description="Download Stanza and spaCy models locally for offline use"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="metrics/models",
        help="Directory to save models (default: metrics/models)"
    )
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["en", "zh", "es", "de", "fr", "it"],
        help="Language codes to download (default: en zh es de fr it)"
    )
    parser.add_argument(
        "--spacy-only",
        action="store_true",
        help="Download only spaCy models"
    )
    parser.add_argument(
        "--stanza-only",
        action="store_true",
        help="Download only Stanza models"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Languages: {', '.join(args.languages)}")
    print()
    
    spacy_success = 0
    spacy_failed = 0
    stanza_success = 0
    stanza_failed = 0
    
    # Download spaCy models
    if not args.stanza_only:
        spacy_success, spacy_failed = download_spacy_models(args.languages, output_dir)
    
    # Download Stanza models
    if not args.spacy_only:
        stanza_success, stanza_failed = download_stanza_models(args.languages, output_dir)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"spaCy: {spacy_success} successful, {spacy_failed} failed")
    print(f"Stanza: {stanza_success} successful, {stanza_failed} failed")
    print()
    print(f"Models saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Review the downloaded models")
    print("  2. Zip the models directory: zip -r models.zip metrics/models/")
    print("  3. Commit to repo (if small) or upload separately")
    print("  4. In SageMaker, unzip in metrics/ directory")
    print("  5. The code will automatically detect models in metrics/models/")


if __name__ == "__main__":
    main()

