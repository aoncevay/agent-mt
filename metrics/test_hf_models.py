#!/usr/bin/env python3
"""
Test script to verify loading of all HuggingFace models and tools needed for metrics evaluation.

This script tests:
1. LaBSE model (for document alignment)
2. awesome-align-with-co model (for term alignment)
3. wmt22-comet-da model (for COMET evaluation)
4. COMET repository (for COMET evaluation)

Usage:
    python metrics/test_hf_models.py
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Set environment variables to prevent HuggingFace connections
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"


def _log_with_time(msg: str):
    """Print message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def find_model_path(model_name: str, possible_paths: list) -> Optional[Path]:
    """Find the first existing path from the list."""
    for path in possible_paths:
        path_obj = Path(path)
        if path_obj.exists():
            return path_obj.resolve()
    return None


def test_labse_model() -> Tuple[bool, str]:
    """Test loading LaBSE model for document alignment."""
    _log_with_time("\n" + "="*80)
    _log_with_time("Testing LaBSE Model")
    _log_with_time("="*80)
    
    # Possible paths for LaBSE
    possible_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/LaBSE"),
        Path.home() / "user-default-efs" / "HF_models" / "LaBSE",
        Path("/mnt/custom-file-systems/efs") / "HF_models" / "LaBSE",
    ]
    
    labse_path = find_model_path("LaBSE", possible_paths)
    
    if labse_path is None:
        return False, f"LaBSE model not found in any of the expected paths:\n  " + "\n  ".join(str(p) for p in possible_paths)
    
    _log_with_time(f"Found LaBSE at: {labse_path}")
    _log_with_time(f"Absolute path: {labse_path.resolve()}")
    
    # Check if it's a symlink
    if labse_path.is_symlink():
        _log_with_time(f"  ⚠ Path is a symlink, pointing to: {labse_path.readlink()}")
    
    # Check directory structure
    _log_with_time("\nChecking directory structure...")
    required_files = ["config.json", "modules.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    
    found_files = []
    missing_files = []
    for file in required_files:
        file_path = labse_path / file
        if file_path.exists():
            found_files.append(file)
            _log_with_time(f"  ✓ Found {file}")
            # Check if it's a symlink
            if file_path.is_symlink():
                _log_with_time(f"    → Symlink to: {file_path.readlink()}")
        else:
            missing_files.append(file)
            _log_with_time(f"  ✗ Missing {file}")
    
    model_file_found = False
    model_file_path = None
    for file in model_files:
        file_path = labse_path / file
        if file_path.exists():
            found_files.append(file)
            model_file_found = True
            model_file_path = file_path
            _log_with_time(f"  ✓ Found {file}")
            # Check file size
            try:
                stat = file_path.stat()
                size_mb = stat.st_size / (1024*1024)
                _log_with_time(f"    Size: {size_mb:.1f} MB")
                _log_with_time(f"    Permissions: {oct(stat.st_mode)}")
                # Check readability
                readable = os.access(file_path, os.R_OK)
                _log_with_time(f"    Readable: {readable}")
                # Check if it's a symlink
                if file_path.is_symlink():
                    _log_with_time(f"    → Symlink to: {file_path.readlink()}")
                # Try reading first few bytes
                try:
                    with open(file_path, 'rb') as f:
                        header = f.read(16)
                        _log_with_time(f"    Header (hex): {header.hex()[:32]}...")
                except Exception as e:
                    _log_with_time(f"    ⚠ Could not read file: {e}")
            except Exception as e:
                _log_with_time(f"    ⚠ Could not stat file: {e}")
    
    if not model_file_found:
        return False, f"No model weight files found ({', '.join(model_files)})"
    
    if missing_files:
        _log_with_time(f"  ⚠ Missing files: {', '.join(missing_files)}")
    
    # Check modules.json structure
    modules_json = labse_path / "modules.json"
    if modules_json.exists():
        _log_with_time("\nChecking modules.json structure...")
        try:
            import json
            with open(modules_json, 'r', encoding='utf-8') as f:
                modules_data = json.load(f)
            _log_with_time(f"  Found {len(modules_data)} modules")
            for i, module in enumerate(modules_data):
                module_path = module.get('path', '')
                module_type = module.get('type', '')
                _log_with_time(f"    Module {i}: path='{module_path}', type={module_type}")
                # Check if module expects files in subdirectory
                if module_path:
                    subdir = labse_path / module_path
                    if subdir.exists():
                        subdir_files = list(subdir.glob("*"))
                        _log_with_time(f"      → Subdirectory exists with {len(subdir_files)} files")
        except Exception as e:
            _log_with_time(f"  ⚠ Could not read modules.json: {e}")
    
    # Check for subdirectories that might contain model files
    _log_with_time("\nChecking for subdirectories...")
    subdirs = [d for d in labse_path.iterdir() if d.is_dir()]
    for subdir in subdirs:
        subdir_files = list(subdir.glob("*.safetensors")) + list(subdir.glob("*.bin"))
        if subdir_files:
            _log_with_time(f"  ⚠ Found model files in subdirectory '{subdir.name}':")
            for f in subdir_files:
                _log_with_time(f"    - {f.name}")
    
    # Try loading with SentenceTransformer
    _log_with_time("\nAttempting to load with SentenceTransformer...")
    try:
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
        
        _log_with_time(f"  sentence-transformers version: {sentence_transformers.__version__}")
        
        # Try different path formats
        path_variants = [
            (str(labse_path), "Absolute path (string)"),
            (str(labse_path.resolve()), "Absolute resolved path (string)"),
            (labse_path, "Path object"),
            (labse_path.resolve(), "Resolved Path object"),
        ]
        
        all_errors = []
        
        for path_var, description in path_variants:
            _log_with_time(f"\n  Attempt with {description}: {path_var}")
            
            # Try with local_files_only=True
            try:
                model = SentenceTransformer(path_var, local_files_only=True)
                _log_with_time(f"  ✓ Successfully loaded with {description} (local_files_only=True)")
                return True, f"LaBSE loaded successfully with {description}"
            except Exception as e1:
                error_msg = f"{type(e1).__name__}: {str(e1)[:200]}"
                _log_with_time(f"  ✗ Failed (local_files_only=True): {error_msg}")
                all_errors.append(f"{description} (local_files_only=True): {error_msg}")
                
                # Try without local_files_only
                try:
                    model = SentenceTransformer(path_var, local_files_only=False)
                    _log_with_time(f"  ✓ Successfully loaded with {description} (local_files_only=False)")
                    return True, f"LaBSE loaded successfully with {description} (without local_files_only)"
                except Exception as e2:
                    error_msg = f"{type(e2).__name__}: {str(e2)[:200]}"
                    _log_with_time(f"  ✗ Failed (local_files_only=False): {error_msg}")
                    all_errors.append(f"{description} (local_files_only=False): {error_msg}")
        
        # Try loading underlying transformers model directly
        _log_with_time("\n  Attempting to load underlying transformers model directly...")
        try:
            from transformers import AutoModel, AutoTokenizer
            _log_with_time("    Loading AutoModel...")
            # Try to find the actual model directory (might be in a subdirectory)
            # Check if there's a "0" subdirectory (module 0)
            module_0_dir = labse_path / "0"
            if module_0_dir.exists():
                _log_with_time(f"    Found '0' subdirectory, trying to load from there...")
                try:
                    model = AutoModel.from_pretrained(str(module_0_dir), local_files_only=True)
                    _log_with_time("    ✓ Successfully loaded from '0' subdirectory")
                    return True, "LaBSE loaded successfully from '0' subdirectory"
                except Exception as e:
                    _log_with_time(f"    ✗ Failed to load from '0' subdirectory: {e}")
            
            # Try loading from root
            try:
                model = AutoModel.from_pretrained(str(labse_path), local_files_only=True)
                _log_with_time("    ✓ Successfully loaded with AutoModel.from_pretrained")
                return True, "LaBSE loaded successfully with AutoModel"
            except Exception as e:
                _log_with_time(f"    ✗ Failed with AutoModel: {e}")
                all_errors.append(f"AutoModel: {e}")
        except ImportError as e:
            _log_with_time(f"    ✗ Could not import transformers: {e}")
        
        # Try by model name (will fail offline, but shows what it expects)
        _log_with_time("\n  Attempting to load by model name (will fail offline)...")
        try:
            model = SentenceTransformer('sentence-transformers/LaBSE', local_files_only=True)
            _log_with_time("  ✓ Successfully loaded by model name")
            return True, "LaBSE loaded successfully (by model name)"
        except Exception as e3:
            _log_with_time(f"  ✗ Failed: {type(e3).__name__}: {e3}")
            all_errors.append(f"Model name: {e3}")
        
        return False, f"All loading attempts failed:\n  " + "\n  ".join(all_errors)
    
    except ImportError as e:
        return False, f"Could not import sentence_transformers: {e}"


def test_awesome_align_model() -> Tuple[bool, str]:
    """Test loading awesome-align-with-co model for term alignment."""
    _log_with_time("\n" + "="*80)
    _log_with_time("Testing awesome-align-with-co Model")
    _log_with_time("="*80)
    
    # Possible paths for awesome-align
    possible_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/awesome-align-with-co"),
        Path.home() / "user-default-efs" / "HF_models" / "awesome-align-with-co",
        Path("/mnt/custom-file-systems/efs") / "HF_models" / "awesome-align-with-co",
    ]
    
    awesome_align_path = find_model_path("awesome-align-with-co", possible_paths)
    
    if awesome_align_path is None:
        return False, f"awesome-align-with-co model not found in any of the expected paths:\n  " + "\n  ".join(str(p) for p in possible_paths)
    
    _log_with_time(f"Found awesome-align-with-co at: {awesome_align_path}")
    
    # Check directory structure
    _log_with_time("Checking directory structure...")
    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]
    
    found_files = []
    missing_files = []
    for file in required_files:
        if (awesome_align_path / file).exists():
            found_files.append(file)
            _log_with_time(f"  ✓ Found {file}")
        else:
            missing_files.append(file)
    
    model_file_found = False
    for file in model_files:
        if (awesome_align_path / file).exists():
            found_files.append(file)
            model_file_found = True
            _log_with_time(f"  ✓ Found {file}")
            size_mb = (awesome_align_path / file).stat().st_size / (1024*1024)
            _log_with_time(f"    Size: {size_mb:.1f} MB")
            readable = os.access(awesome_align_path / file, os.R_OK)
            _log_with_time(f"    Readable: {readable}")
    
    if not model_file_found:
        return False, f"No model weight files found ({', '.join(model_files)})"
    
    if missing_files:
        _log_with_time(f"  ⚠ Missing files: {', '.join(missing_files)}")
    
    # Try loading with transformers
    _log_with_time("\nAttempting to load with transformers...")
    try:
        from transformers import AutoModel, AutoTokenizer
        
        _log_with_time("  Loading model...")
        try:
            model = AutoModel.from_pretrained(str(awesome_align_path), local_files_only=True)
            tokenizer = AutoTokenizer.from_pretrained(str(awesome_align_path), local_files_only=True)
            _log_with_time("  ✓ Successfully loaded model and tokenizer")
            
            # Check GPU availability
            import torch
            if torch.cuda.is_available():
                _log_with_time(f"  GPU available: {torch.cuda.get_device_name(0)}")
                _log_with_time("  Testing GPU move...")
                model = model.to('cuda')
                _log_with_time("  ✓ Model moved to GPU successfully")
            else:
                _log_with_time("  No GPU available, using CPU")
            
            return True, "awesome-align-with-co loaded successfully"
        except Exception as e:
            _log_with_time(f"  ✗ Failed: {type(e).__name__}: {e}")
            return False, f"Could not load awesome-align-with-co: {e}"
    
    except ImportError as e:
        return False, f"Could not import transformers: {e}"


def test_comet_model() -> Tuple[bool, str]:
    """Test loading wmt22-comet-da model for COMET evaluation."""
    _log_with_time("\n" + "="*80)
    _log_with_time("Testing wmt22-comet-da Model")
    _log_with_time("="*80)
    
    # Possible paths for COMET model
    possible_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/wmt22-comet-da"),
        Path.home() / "user-default-efs" / "HF_models" / "wmt22-comet-da",
        Path("/mnt/custom-file-systems/efs") / "HF_models" / "wmt22-comet-da",
    ]
    
    comet_model_path = find_model_path("wmt22-comet-da", possible_paths)
    
    if comet_model_path is None:
        return False, f"wmt22-comet-da model not found in any of the expected paths:\n  " + "\n  ".join(str(p) for p in possible_paths)
    
    _log_with_time(f"Found wmt22-comet-da at: {comet_model_path}")
    
    # Check directory structure
    _log_with_time("Checking directory structure...")
    # COMET models typically have checkpoint files
    checkpoint_files = list(comet_model_path.glob("*.ckpt"))
    pytorch_files = list(comet_model_path.glob("*.bin"))
    safetensors_files = list(comet_model_path.glob("*.safetensors"))
    
    if checkpoint_files:
        _log_with_time(f"  ✓ Found {len(checkpoint_files)} checkpoint file(s)")
        for ckpt in checkpoint_files[:3]:  # Show first 3
            size_mb = ckpt.stat().st_size / (1024*1024)
            _log_with_time(f"    - {ckpt.name} ({size_mb:.1f} MB)")
    if pytorch_files:
        _log_with_time(f"  ✓ Found {len(pytorch_files)} pytorch file(s)")
    if safetensors_files:
        _log_with_time(f"  ✓ Found {len(safetensors_files)} safetensors file(s)")
    
    if not checkpoint_files and not pytorch_files and not safetensors_files:
        return False, "No model checkpoint or weight files found"
    
    # Try loading with COMET
    _log_with_time("\nAttempting to load with COMET...")
    try:
        # Find COMET repo
        comet_repo_paths = [
            Path("other_repos/COMET"),
            Path.home() / "user-default-efs" / "tools" / "COMET",
            Path("/mnt/custom-file-systems/efs") / "tools" / "COMET",
        ]
        
        comet_repo = find_model_path("COMET", comet_repo_paths)
        
        if comet_repo is None:
            return False, f"COMET repository not found in any of the expected paths:\n  " + "\n  ".join(str(p) for p in comet_repo_paths)
        
        _log_with_time(f"Found COMET repository at: {comet_repo}")
        
        # Add to path
        comet_repo_str = str(comet_repo.resolve())
        if comet_repo_str not in sys.path:
            sys.path.insert(0, comet_repo_str)
        
        # Try importing
        try:
            from comet import load_from_checkpoint, download_model
            _log_with_time("  ✓ Successfully imported COMET module")
        except ImportError:
            try:
                from comet.models import load_from_checkpoint, download_model
                _log_with_time("  ✓ Successfully imported COMET module (from comet.models)")
            except ImportError as e:
                return False, f"Could not import COMET module: {e}"
        
        # Try loading model
        _log_with_time("  Loading model from checkpoint...")
        try:
            model = load_from_checkpoint(str(comet_model_path))
            _log_with_time("  ✓ Successfully loaded COMET model")
            
            # Check GPU availability
            import torch
            if torch.cuda.is_available():
                _log_with_time(f"  GPU available: {torch.cuda.get_device_name(0)}")
                _log_with_time("  Testing GPU move...")
                model.to('cuda')
                _log_with_time("  ✓ Model moved to GPU successfully")
            else:
                _log_with_time("  No GPU available, using CPU")
            
            return True, "wmt22-comet-da loaded successfully"
        except Exception as e:
            _log_with_time(f"  ✗ Failed: {type(e).__name__}: {e}")
            return False, f"Could not load COMET model: {e}"
    
    except Exception as e:
        return False, f"Error testing COMET: {e}"


def test_comet_repository() -> Tuple[bool, str]:
    """Test that COMET repository is accessible."""
    _log_with_time("\n" + "="*80)
    _log_with_time("Testing COMET Repository")
    _log_with_time("="*80)
    
    # Possible paths for COMET repo
    possible_paths = [
        Path("other_repos/COMET"),
        Path.home() / "user-default-efs" / "tools" / "COMET",
        Path("/mnt/custom-file-systems/efs") / "tools" / "COMET",
    ]
    
    comet_repo = find_model_path("COMET", possible_paths)
    
    if comet_repo is None:
        return False, f"COMET repository not found in any of the expected paths:\n  " + "\n  ".join(str(p) for p in possible_paths)
    
    _log_with_time(f"Found COMET repository at: {comet_repo}")
    
    # Check for key files
    _log_with_time("Checking repository structure...")
    key_files = [
        "comet/__init__.py",
        "comet/models.py",
        "setup.py",
        "requirements.txt",
    ]
    
    found_files = []
    missing_files = []
    for file in key_files:
        if (comet_repo / file).exists():
            found_files.append(file)
            _log_with_time(f"  ✓ Found {file}")
        else:
            missing_files.append(file)
    
    if missing_files:
        _log_with_time(f"  ⚠ Missing files: {', '.join(missing_files)}")
    
    # Try importing
    _log_with_time("\nAttempting to import COMET module...")
    comet_repo_str = str(comet_repo.resolve())
    if comet_repo_str not in sys.path:
        sys.path.insert(0, comet_repo_str)
    
    try:
        from comet import load_from_checkpoint, download_model
        _log_with_time("  ✓ Successfully imported from comet")
        return True, "COMET repository is accessible"
    except ImportError:
        try:
            from comet.models import load_from_checkpoint, download_model
            _log_with_time("  ✓ Successfully imported from comet.models")
            return True, "COMET repository is accessible (from comet.models)"
        except ImportError as e:
            return False, f"Could not import COMET module: {e}"


def main():
    """Run all model tests."""
    _log_with_time("="*80)
    _log_with_time("HF Models and Tools Test Suite")
    _log_with_time("="*80)
    _log_with_time("This script tests loading of all models needed for metrics evaluation")
    _log_with_time("")
    
    results = []
    
    # Test 1: LaBSE
    success, message = test_labse_model()
    results.append(("LaBSE", success, message))
    
    # Test 2: awesome-align-with-co
    success, message = test_awesome_align_model()
    results.append(("awesome-align-with-co", success, message))
    
    # Test 3: COMET repository
    success, message = test_comet_repository()
    results.append(("COMET Repository", success, message))
    
    # Test 4: wmt22-comet-da model
    success, message = test_comet_model()
    results.append(("wmt22-comet-da", success, message))
    
    # Print summary
    _log_with_time("\n" + "="*80)
    _log_with_time("Test Summary")
    _log_with_time("="*80)
    
    all_passed = True
    for name, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        _log_with_time(f"{status}: {name}")
        if not success:
            all_passed = False
            _log_with_time(f"  Error: {message}")
    
    _log_with_time("")
    if all_passed:
        _log_with_time("✓ All tests passed!")
        return 0
    else:
        _log_with_time("✗ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

