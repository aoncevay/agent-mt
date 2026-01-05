"""
COMET evaluator for computing reference-based quality scores per segment.

Uses COMET-DA model for reference-based evaluation.
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json


def compute_comet_scores(
    segments: List[Tuple[str, str, str]],
    comet_model_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Compute COMET scores for aligned segments.
    
    Args:
        segments: List of (source, translation, reference) tuples
        comet_model_path: Path to COMET-DA model directory (default: ~/user-default-efs/HF_models/wmt22-comet-da)
    
    Returns:
        Dictionary with:
        - 'avg_comet': Average COMET score
        - 'min_comet': Minimum COMET score
        - 'max_comet': Maximum COMET score
        - 'scores': List of per-segment scores
    """
    if comet_model_path is None:
        # Default path from user's configuration
        comet_model_path = Path.home() / "user-default-efs" / "HF_models" / "wmt22-comet-da"
    
    if not comet_model_path.exists():
        raise FileNotFoundError(
            f"COMET-DA model not found at {comet_model_path}\n"
            f"Please ensure the model is downloaded and available."
        )
    
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        raise ImportError(
            "COMET library not found. Please install with: pip install unbabel-comet"
        )
    
    # Load COMET model
    print(f"  Loading COMET-DA model from {comet_model_path}...")
    try:
        # Try to load from local path
        model = load_from_checkpoint(str(comet_model_path))
    except Exception:
        # Fallback: try to download (shouldn't happen if model is local)
        print(f"  ⚠ Warning: Could not load from {comet_model_path}, trying download...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        model = load_from_checkpoint(model_path)
    
    # Prepare data for COMET
    sources = [seg[0] for seg in segments]
    translations = [seg[1] for seg in segments]
    references = [seg[2] for seg in segments]
    
    # Compute scores
    print(f"  Computing COMET scores for {len(segments)} segments...")
    data = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(sources, translations, references)
    ]
    
    scores, _ = model.predict(data, batch_size=8, gpus=1)
    
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

