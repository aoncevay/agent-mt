#!/usr/bin/env python3
"""
Setup script to configure SEGALE to use local COMET-DA model (reference-based evaluation).

This script:
1. Creates a patched version of segale_eval.py that uses local COMET-DA model path
2. Sets up environment variables
3. Provides instructions for running SEGALE offline with COMET-DA

Usage:
    python segale/setup_local_segale.py --models-dir /path/to/models
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SEGALE_DIR = PROJECT_ROOT / "other_repos" / "SEGALE"


def patch_segale_eval_for_local_comet(models_dir: Path):
    """
    Create a patched version of segale_eval.py that uses local COMET-DA model (reference-based only).
    
    Args:
        models_dir: Path to directory containing local COMET-DA model
    """
    segale_eval_file = SEGALE_DIR / "segale_eval.py"
    backup_file = SEGALE_DIR / "segale_eval.py.backup"
    patched_file = SEGALE_DIR / "segale_eval.py"
    
    models_dir_str = str(models_dir.resolve())
    
    # Create backup
    if not backup_file.exists():
        shutil.copy2(segale_eval_file, backup_file)
        print(f"✓ Created backup: {backup_file}")
    
    # Read original file
    with open(segale_eval_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Patch COMET-DA model loading - use local path instead of download
    old_comet_da = 'model_path = download_model("Unbabel/wmt22-comet-da")'
    new_comet_da = f'model_path = "{models_dir_str}/wmt22-comet-da"'
    content = content.replace(old_comet_da, new_comet_da)
    
    # Disable COMET-KIWI (quality estimation, not needed for reference-based)
    content = content.replace(
        'comet_qe_scores   = run_comet_qe_evaluation(comet_qe_windows)',
        'comet_qe_scores   = [-1] * len(comet_qe_windows)  # Disabled: using reference-based COMET only'
    )
    
    # Disable MetricX evaluation (we're using COMET only)
    content = content.replace(
        'metricx_scores    = run_metricx_evaluation(metricx_windows)',
        'metricx_scores    = [-1] * len(metricx_windows)  # Disabled: using COMET only'
    )
    content = content.replace(
        'metricx_qe_scores = run_metricx_qe_evaluation(metricx_qe_windows)',
        'metricx_qe_scores = [-1] * len(metricx_qe_windows)  # Disabled: using COMET only'
    )
    
    # Write patched file
    with open(patched_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Patched segale_eval.py to use local COMET-DA model (reference-based only)")
    return True


def create_env_script(models_dir: Path, output_file: Path):
    """Create a shell script to set environment variables."""
    models_dir_str = str(models_dir.resolve())
    
    script_content = f"""#!/bin/bash
# Environment variables for SEGALE with local COMET-DA model
# Source this file before running SEGALE: source {output_file.name}

export TRANSFORMERS_CACHE="{models_dir_str}"
export HF_HOME="{models_dir_str}"
export HUGGINGFACE_HUB_CACHE="{models_dir_str}"
export HF_DATASETS_OFFLINE=1

# Disable HuggingFace downloads
export TRANSFORMERS_OFFLINE=1

echo "✓ Environment variables set for local COMET-DA model"
echo "  Models directory: {models_dir_str}"
"""
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    output_file.chmod(0o755)
    print(f"✓ Created environment script: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Configure SEGALE to use local COMET-DA model (reference-based evaluation)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Path to directory containing local COMET-DA model"
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Restore original segale_eval.py from backup"
    )
    
    args = parser.parse_args()
    
    models_dir = Path(args.models_dir)
    backup_file = SEGALE_DIR / "segale_eval.py.backup"
    segale_eval_file = SEGALE_DIR / "segale_eval.py"
    
    if args.restore:
        if backup_file.exists():
            shutil.copy2(backup_file, segale_eval_file)
            print(f"✓ Restored original segale_eval.py from backup")
            return 0
        else:
            print("✗ No backup file found")
            return 1
    
    if not models_dir.exists():
        print(f"✗ Error: Models directory does not exist: {models_dir}")
        return 1
    
    # Check for required COMET-DA model (reference-based only)
    print("Checking for required COMET-DA model (reference-based evaluation)...")
    
    required_model = "wmt22-comet-da"
    model_path = models_dir / required_model
    
    if model_path.exists():
        print(f"  ✓ {required_model}: COMET-DA model (reference-based)")
    else:
        print(f"  ✗ {required_model}: COMET-DA model (reference-based) - MISSING")
        print(f"\n✗ Missing model: {required_model}")
        print("\nTo download COMET-DA model on a machine with internet:")
        print(f"  huggingface-cli download Unbabel/wmt22-comet-da --local-dir <models_dir>/wmt22-comet-da")
        return 1
    
    print("\n" + "=" * 80)
    print("Patching SEGALE to use local COMET-DA model (reference-based only)...")
    print("=" * 80)
    
    # Patch segale_eval.py
    if not patch_segale_eval_for_local_comet(models_dir):
        return 1
    
    # Create environment script
    env_script = PROJECT_ROOT / "segale" / "segale_local_env.sh"
    create_env_script(models_dir, env_script)
    
    print("\n" + "=" * 80)
    print("Setup Complete!")
    print("=" * 80)
    print("\nTo use SEGALE with local COMET-DA model (reference-based evaluation):")
    print(f"1. Source the environment script: source {env_script}")
    print("2. Run your SEGALE commands as normal")
    print("\nExample:")
    print(f"  source {env_script}")
    print("  python segale/test_segale.py --output-dir outputs/...")
    print("\nTo restore original SEGALE:")
    print("  python segale/setup_local_segale.py --models-dir <dir> --restore")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

