"""
Utility functions for saving and loading intermediate alignment results.

Used to split the evaluation pipeline into stages to reduce GPU memory usage.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import json
import pickle


TMP_DIR = Path(__file__).parent / "tmp"


def get_tmp_file_path(output_dir: Path, dataset: str, lang_pair: str) -> Path:
    """
    Get the path to save/load temporary alignment results.
    
    Args:
        output_dir: Original experiment output directory
        dataset: Dataset name
        lang_pair: Language pair
    
    Returns:
        Path to temporary file
    """
    # Extract workflow and model from output_dir
    # Structure: outputs/{dataset}/{lang_pair}/{workflow_dir}/{model}/
    parts = output_dir.parts
    try:
        # Find the index of 'outputs' or 'outputs_qwen3'
        base_idx = -1
        for i, part in enumerate(parts):
            if part in ['outputs', 'outputs_qwen3']:
                base_idx = i
                break
        
        if base_idx == -1:
            raise ValueError(f"Could not find 'outputs' in path: {output_dir}")
        
        # Extract: workflow_dir, model
        workflow_dir = parts[base_idx + 3]  # e.g., "IRB.term" or "IRB"
        model = parts[base_idx + 4]
        
        # Create tmp directory structure: tmp/{dataset}/{lang_pair}/{workflow_dir}/{model}/
        tmp_file = TMP_DIR / dataset / lang_pair / workflow_dir / model / "aligned_df.pkl"
        return tmp_file
    except (IndexError, ValueError) as e:
        # Fallback: use a hash of the output_dir
        import hashlib
        dir_hash = hashlib.md5(str(output_dir).encode()).hexdigest()[:8]
        return TMP_DIR / f"aligned_{dir_hash}.pkl"


def save_aligned_df(
    aligned_df: pd.DataFrame,
    output_dir: Path,
    dataset: str,
    lang_pair: str,
    sample_data: list
) -> Path:
    """
    Save aligned DataFrame and sample data to temporary file.
    
    Args:
        aligned_df: DataFrame with aligned segments
        output_dir: Original experiment output directory
        dataset: Dataset name
        lang_pair: Language pair
        sample_data: List of sample data dictionaries
    
    Returns:
        Path to saved file
    """
    tmp_file = get_tmp_file_path(output_dir, dataset, lang_pair)
    tmp_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save both aligned_df and sample_data
    data = {
        'aligned_df': aligned_df,
        'sample_data': sample_data
    }
    
    with open(tmp_file, 'wb') as f:
        pickle.dump(data, f)
    
    return tmp_file


def load_aligned_df(
    output_dir: Path,
    dataset: str,
    lang_pair: str
) -> Optional[Tuple[pd.DataFrame, list]]:
    """
    Load aligned DataFrame and sample data from temporary file.
    
    Args:
        output_dir: Original experiment output directory
        dataset: Dataset name
        lang_pair: Language pair
    
    Returns:
        Tuple of (aligned_df, sample_data) or None if file doesn't exist
    """
    tmp_file = get_tmp_file_path(output_dir, dataset, lang_pair)
    
    if not tmp_file.exists():
        return None
    
    try:
        with open(tmp_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['aligned_df'], data['sample_data']
    except Exception as e:
        print(f"  ⚠ Warning: Could not load tmp file {tmp_file}: {e}")
        return None


def has_aligned_df(output_dir: Path, dataset: str, lang_pair: str) -> bool:
    """
    Check if temporary alignment file exists.
    
    Args:
        output_dir: Original experiment output directory
        dataset: Dataset name
        lang_pair: Language pair
    
    Returns:
        True if file exists, False otherwise
    """
    tmp_file = get_tmp_file_path(output_dir, dataset, lang_pair)
    return tmp_file.exists()

