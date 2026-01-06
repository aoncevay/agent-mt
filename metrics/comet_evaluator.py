"""
COMET evaluator for computing reference-based quality scores per segment.

Uses COMET-DA model for reference-based evaluation.
Works with cloned COMET repository (no pip install needed).
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import sys
import yaml


def _find_comet_repo() -> Optional[Path]:
    """
    Find the cloned COMET repository.
    Checks both local (other_repos/COMET) and SageMaker paths (EFS mount and home directory).
    
    Returns:
        Path to COMET repository, or None if not found
    """
    # Try local path first (for development)
    local_path = Path(__file__).parent.parent.parent / "other_repos" / "COMET"
    if local_path.exists():
        return local_path
    
    # Try EFS mount path (SageMaker)
    efs_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/tools/COMET"),
        Path("/mnt/custom-file-systems/efs/tools/COMET"),
    ]
    for efs_path in efs_paths:
        if efs_path.exists():
            return efs_path
    
    # Try SageMaker home directory path
    sagemaker_path = Path.home() / "user-default-efs" / "tools" / "COMET"
    if sagemaker_path.exists():
        return sagemaker_path
    
    # Try alternative SageMaker path (if cloned with different name)
    sagemaker_path_alt = Path.home() / "user-default-efs" / "tools" / "comet"
    if sagemaker_path_alt.exists():
        return sagemaker_path_alt
    
    return None


def _load_comet_module():
    """
    Load COMET module from cloned repository.
    Adds the repository to sys.path if needed.
    
    Returns:
        Tuple of (load_from_checkpoint, download_model) functions
    """
    comet_repo = _find_comet_repo()
    
    if comet_repo is None:
        raise ImportError(
            "COMET repository not found. Please clone it to one of:\n"
            "  - other_repos/COMET (local development)\n"
            "  - ~/user-default-efs/tools/COMET (SageMaker)\n"
            "Or install with: pip install unbabel-comet"
        )
    
    # Add COMET repo to path if not already there
    comet_repo_str = str(comet_repo.resolve())
    if comet_repo_str not in sys.path:
        sys.path.insert(0, comet_repo_str)
    
    try:
        # Try the main import path (from comet/__init__.py)
        from comet import load_from_checkpoint, download_model
        return load_from_checkpoint, download_model
    except ImportError as e:
        # Check if it's a missing dependency issue
        error_msg = str(e)
        if "pytorch_lightning" in error_msg or "pytorch-lightning" in error_msg:
            raise ImportError(
                f"COMET requires 'pytorch_lightning' but it's not installed.\n"
                f"Please install it with: pip install pytorch-lightning\n"
                f"Or add it to your requirements.txt.\n"
                f"COMET repo found at: {comet_repo_str}\n"
                f"Original error: {error_msg}"
            )
        # Try alternative import path (from comet.models)
        try:
            from comet.models import load_from_checkpoint, download_model
            return load_from_checkpoint, download_model
        except ImportError as e2:
            error_msg2 = str(e2)
            if "pytorch_lightning" in error_msg2 or "pytorch-lightning" in error_msg2:
                raise ImportError(
                    f"COMET requires 'pytorch_lightning' but it's not installed.\n"
                    f"Please install it with: pip install pytorch-lightning\n"
                    f"Or add it to your requirements.txt.\n"
                    f"COMET repo found at: {comet_repo_str}\n"
                    f"Original error: {error_msg2}"
                )
            raise ImportError(
                f"Could not import COMET from {comet_repo}. "
                f"Please ensure the repository is properly cloned and dependencies are installed.\n"
                f"Path checked: {comet_repo_str}\n"
                f"Original error: {error_msg2}\n"
                f"Required dependencies: pytorch-lightning, torch, transformers, pandas, numpy, etc.\n"
                f"See other_repos/COMET/pyproject.toml for full list."
            )


def compute_comet_scores(
    segments: List[Tuple[str, str, str]],
    comet_model_path: Optional[Path] = None,
    comet_repo_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute COMET scores for aligned segments.
    
    Args:
        segments: List of (source, translation, reference) tuples
        comet_model_path: Path to COMET-DA model directory (default: ~/user-default-efs/HF_models/wmt22-comet-da)
        comet_repo_path: Optional path to cloned COMET repository (auto-detected if not provided)
    
    Returns:
        Dictionary with:
        - 'avg_comet': Average COMET score
        - 'min_comet': Minimum COMET score
        - 'max_comet': Maximum COMET score
        - 'scores': List of per-segment scores
    """
    if comet_model_path is None:
        # Try multiple possible paths for EFS mount
        possible_paths = [
            # EFS mount path (SageMaker)
            Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/wmt22-comet-da"),
            # Home directory path
            Path.home() / "user-default-efs" / "HF_models" / "wmt22-comet-da",
            # Alternative EFS path pattern
            Path("/mnt/custom-file-systems/efs") / "HF_models" / "wmt22-comet-da",
        ]
        
        # Find the first path that exists
        comet_model_path = None
        for path in possible_paths:
            if path.exists():
                comet_model_path = path
                break
        
        # If none found, use the first one as default (will raise error below)
        if comet_model_path is None:
            comet_model_path = possible_paths[0]
    
    # Convert to absolute path and check if model exists
    comet_model_path = Path(comet_model_path).resolve()
    
    # COMET's load_from_checkpoint expects a checkpoint FILE, not a directory
    # If the path is a directory, look for the checkpoint file inside it
    if comet_model_path.is_dir():
        # Try common checkpoint file locations
        possible_checkpoint_files = [
            comet_model_path / "checkpoints" / "model.ckpt",  # Standard COMET structure
            comet_model_path / "model.ckpt",  # Alternative location
            comet_model_path / "checkpoint.ckpt",  # Another alternative
        ]
        
        checkpoint_found = False
        for checkpoint_file in possible_checkpoint_files:
            if checkpoint_file.exists() and checkpoint_file.is_file():
                comet_model_path = checkpoint_file
                checkpoint_found = True
                break
        
        if not checkpoint_found:
            raise FileNotFoundError(
                f"COMET-DA model directory found at {comet_model_path}, but no checkpoint file found.\n"
                f"Looked for:\n"
                f"  - {comet_model_path / 'checkpoints' / 'model.ckpt'}\n"
                f"  - {comet_model_path / 'model.ckpt'}\n"
                f"  - {comet_model_path / 'checkpoint.ckpt'}\n"
                f"Please ensure the checkpoint file exists in one of these locations."
            )
    elif not comet_model_path.exists():
        raise FileNotFoundError(
            f"COMET-DA model not found at {comet_model_path}\n"
            f"Please ensure the model is downloaded and available.\n"
            f"Tried paths:\n"
            f"  - /mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/wmt22-comet-da\n"
            f"  - ~/user-default-efs/HF_models/wmt22-comet-da\n"
            f"  - /mnt/custom-file-systems/efs/HF_models/wmt22-comet-da"
        )
    
    # Load COMET module from cloned repo
    if comet_repo_path:
        # Temporarily add to path
        comet_repo_str = str(Path(comet_repo_path).resolve())
        if comet_repo_str not in sys.path:
            sys.path.insert(0, comet_repo_str)
    
    load_from_checkpoint, download_model = _load_comet_module()
    
    # Load COMET model
    print(f"  Loading COMET-DA model from {comet_model_path}...")
    try:
        # Check if model uses sparsemax (which requires entmax)
        # If entmax is not available, we'll use softmax as fallback
        # hparams.yaml is located at checkpoint_path.parent.parent (two levels up from checkpoint file)
        # If checkpoint is at wmt22-comet-da/checkpoints/model.ckpt, hparams is at wmt22-comet-da/hparams.yaml
        # If checkpoint is at wmt22-comet-da/model.ckpt, hparams is at wmt22-comet-da/hparams.yaml
        checkpoint_path_obj = Path(comet_model_path)
        if checkpoint_path_obj.is_file():
            # Checkpoint is a file, hparams is two levels up
            hparams_file = checkpoint_path_obj.parent.parent / "hparams.yaml"
        else:
            # Checkpoint is a directory, hparams is in the same directory
            hparams_file = checkpoint_path_obj / "hparams.yaml"
        
        use_softmax_fallback = False
        
        if hparams_file.exists():
            try:
                with open(hparams_file, encoding='utf-8') as f:
                    hparams = yaml.safe_load(f)
                if hparams.get("layer_transformation") == "sparsemax":
                    # Check if entmax is available
                    try:
                        import entmax  # noqa: F401
                        # entmax is available, no fallback needed
                    except ImportError:
                        # entmax not available, use softmax fallback
                        print(f"  ⚠ Model uses sparsemax but entmax not available, using softmax fallback")
                        use_softmax_fallback = True
            except Exception:
                # If we can't read hparams, proceed normally
                pass
        
        if use_softmax_fallback:
            # Load with softmax override
            from comet.models import str2model
            model_class = str2model[hparams["class_identifier"]]
            model = model_class.load_from_checkpoint(
                str(comet_model_path),
                layer_transformation="softmax",
                load_pretrained_weights=False,
                strict=False,
                local_files_only=True
            )
        else:
            # Normal loading (model uses softmax or entmax is available)
            model = load_from_checkpoint(str(comet_model_path))
        
        # Check GPU availability and inform user
        import torch
        if torch.cuda.is_available():
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  COMET will use GPU (gpus=1 in predict call)")
        else:
            print(f"  No GPU available, using CPU")
    except Exception as e:
        # Fallback: try to download (shouldn't happen if model is local)
        print(f"  ⚠ Warning: Could not load from {comet_model_path}: {e}")
        print(f"  Trying to download model...")
        try:
            model_path = download_model("Unbabel/wmt22-comet-da")
            model = load_from_checkpoint(model_path)
        except Exception as e2:
            raise RuntimeError(
                f"Could not load COMET model: {e2}\n"
                f"Please ensure the model is available at {comet_model_path}"
            )
    
    # Prepare data for COMET
    sources = [seg[0] for seg in segments]
    translations = [seg[1] for seg in segments]
    references = [seg[2] for seg in segments]
    
    # Compute scores
    import time
    print(f"  Computing COMET scores for {len(segments)} segments...")
    _log_with_time = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    
    _log_with_time("  Preparing data for COMET...")
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, translations, references)
    ]
    
    _log_with_time(f"  Running COMET prediction (batch_size=8, gpus=1)...")
    predict_start = time.time()
    scores, _ = model.predict(data, batch_size=8, gpus=1)
    _log_with_time(f"  ✓ COMET prediction completed in {time.time() - predict_start:.2f}s")
    
    # Convert to list if needed (COMET may return numpy array)
    if hasattr(scores, 'tolist'):
        scores = scores.tolist()
    else:
        scores = list(scores)
    
    # Compute statistics
    if not scores:
        return {
            'avg_comet': None,
            'min_comet': None,
            'max_comet': None,
            'scores': []
        }
    
    avg_comet = sum(scores) / len(scores)
    min_comet = min(scores)
    max_comet = max(scores)
    
    return {
        'avg_comet': avg_comet,
        'min_comet': min_comet,
        'max_comet': max_comet,
        'scores': scores
    }


def compute_comet_scores_from_dataframe(
    df: 'pd.DataFrame',
    src_col: str,
    tgt_col: str,
    ref_texts: List[str],
    comet_model_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute COMET scores from a DataFrame with aligned segments.
    
    Args:
        df: DataFrame with aligned segments (from docpreprocessor)
        src_col: Column name for source text
        tgt_col: Column name for target/translation text
        ref_texts: List of reference texts (one per document, will be split to match segments)
        comet_model_path: Path to COMET-DA model
    
    Returns:
        Dictionary with COMET statistics
    """
    # Prepare segments: (source, translation, reference)
    # Note: references need to be aligned to segments - for now, use document-level reference
    # TODO: Implement proper reference alignment if needed
    
    segments = []
    for idx, row in df.iterrows():
        src = row[src_col]
        tgt = row[tgt_col]
        # For now, use the first reference (document-level)
        # In the future, we might need to align references to segments
        ref = ref_texts[0] if ref_texts else ""
        segments.append((src, tgt, ref))
    
    return compute_comet_scores(segments, comet_model_path)

