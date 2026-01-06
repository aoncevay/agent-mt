#!/usr/bin/env python3
"""
Test script to verify COMET repository can be loaded correctly.
Run this after installing dependencies from requirements.txt.
"""

import sys
from pathlib import Path

def test_comet_loading():
    """Test if COMET can be imported and loaded."""
    print("Testing COMET repository loading...")
    print("=" * 60)
    
    # Test 1: Check if repo is found
    print("\n1. Checking if COMET repository is found...")
    from metrics.comet_evaluator import _find_comet_repo
    comet_repo = _find_comet_repo()
    
    if comet_repo is None:
        print("   ✗ COMET repository not found!")
        print("   Please clone it to one of:")
        print("     - other_repos/COMET (local development)")
        print("     - ~/user-default-efs/tools/COMET (SageMaker)")
        print("     - /mnt/custom-file-systems/efs/.../tools/COMET (SageMaker EFS)")
        return False
    
    print(f"   ✓ Found COMET repository at: {comet_repo}")
    
    # Test 2: Check if module can be imported
    print("\n2. Testing COMET module import...")
    try:
        from metrics.comet_evaluator import _load_comet_module
        load_from_checkpoint, download_model = _load_comet_module()
        print("   ✓ COMET module imported successfully")
        print(f"   ✓ load_from_checkpoint: {load_from_checkpoint}")
        print(f"   ✓ download_model: {download_model}")
    except ImportError as e:
        print(f"   ✗ Failed to import COMET module: {e}")
        print("\n   This usually means missing dependencies.")
        print("   Please install: pip install -r metrics/requirements.txt")
        return False
    except Exception as e:
        print(f"   ✗ Unexpected error: {e}")
        return False
    
    # Test 3: Check if we can check for a model (without loading it)
    print("\n3. Testing COMET model path detection...")
    from metrics.comet_evaluator import compute_comet_scores
    
    # Try to find model path (but don't load it)
    possible_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/wmt22-comet-da"),
        Path.home() / "user-default-efs" / "HF_models" / "wmt22-comet-da",
        Path("/mnt/custom-file-systems/efs") / "HF_models" / "wmt22-comet-da",
    ]
    
    model_found = False
    for path in possible_paths:
        if path.exists():
            print(f"   ✓ Found COMET model at: {path}")
            model_found = True
            break
    
    if not model_found:
        print("   ⚠ COMET model not found (this is okay if you're just testing imports)")
        print("   Model paths checked:")
        for path in possible_paths:
            print(f"     - {path}")
    
    print("\n" + "=" * 60)
    print("✓ COMET loading test completed successfully!")
    print("\nNext steps:")
    print("  1. Ensure all dependencies are installed: pip install -r metrics/requirements.txt")
    print("  2. Ensure COMET model is available at one of the paths above")
    print("  3. Run evaluate_experiments.py to compute metrics")
    
    return True


if __name__ == "__main__":
    success = test_comet_loading()
    sys.exit(0 if success else 1)

