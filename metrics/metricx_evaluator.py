"""
MetricX-24 evaluator for computing reference-based quality scores per segment.

Uses MetricX-24-Hybrid-Large model for reference-based evaluation.
Based on mT5, so requires mT5 tokenizer files.

Uses the same approach as SEGALE: calls metricx24.predict module.
If metricx24 package is not available, falls back to direct transformers usage.

Reference: https://huggingface.co/google/metricx-24-hybrid-large-v2p6-bfloat16
"""

from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import sys
import os
import torch
import json
import tempfile
import subprocess
import importlib.util

# Import centralized configuration
from metrics.config import get_config


def _find_metricx_model() -> Optional[Path]:
    """
    Find the MetricX-24 model directory using centralized config.
    
    Returns:
        Path to MetricX-24 model directory, or None if not found
    """
    # Use centralized config first
    config = get_config()
    metricx_path = config.get_metricx_path()
    if metricx_path and metricx_path.exists():
        return metricx_path
    
    # Fallback to hardcoded paths for backward compatibility
    fallback_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/metricx-24-hybrid-large-v2p6-bfloat16"),
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/metricx-24-hybrid-large-v2p6"),
        Path.home() / "user-default-efs" / "HF_models" / "metricx-24-hybrid-large-v2p6-bfloat16",
        Path.home() / "user-default-efs" / "HF_models" / "metricx-24-hybrid-large-v2p6",
        Path.home() / "HF_models" / "metricx-24-hybrid-large-v2p6-bfloat16",
    ]
    
    for path in fallback_paths:
        if path.exists():
            return path
    
    return None


def _find_mt5_tokenizer() -> Optional[Path]:
    """
    Find the mT5 tokenizer files using centralized config.
    MetricX-24 is based on mT5, so it needs mT5 tokenizer files.
    
    Returns:
        Path to mT5 tokenizer directory, or None if not found
    """
    # Use centralized config first
    config = get_config()
    mt5_path = config.get_mt5_tokenizer_path()
    if mt5_path and mt5_path.exists():
        return mt5_path
    
    # Fallback to hardcoded paths for backward compatibility
    fallback_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/mt5-base"),
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/mt5-large"),
        Path.home() / "user-default-efs" / "HF_models" / "mt5-base",
        Path.home() / "user-default-efs" / "HF_models" / "mt5-large",
        Path.home() / "HF_models" / "mt5-base",
        Path("/mnt/custom-file-systems/efs") / "HF_models" / "mt5-base",
        Path.home() / ".cache" / "huggingface" / "hub" / "models--google--mt5-base",
    ]
    
    for path in fallback_paths:
        if path.exists():
            # Check for tokenizer files
            # mT5 uses 'spiece.model' (not 'sentencepiece.model')
            tokenizer_files = [
                path / "spiece.model",  # mT5 uses this name
                path / "tokenizer_config.json",
            ]
            # Also check in snapshots subdirectory (HF cache structure)
            if (path / "snapshots").exists():
                for snapshot_dir in (path / "snapshots").iterdir():
                    if snapshot_dir.is_dir():
                        snapshot_tokenizer_files = [
                            snapshot_dir / "spiece.model",  # mT5 uses this name
                            snapshot_dir / "tokenizer_config.json",
                        ]
                        if any(f.exists() for f in snapshot_tokenizer_files):
                            return snapshot_dir
            else:
                # Direct model directory
                if any(f.exists() for f in tokenizer_files):
                    return path
    
    return None


def compute_metricx_scores(
    segments: List[Tuple[str, str, str]],
    metricx_model_path: Optional[Path] = None,
    mt5_tokenizer_path: Optional[Path] = None,
    metric_model=None,  # Optional pre-loaded model
    tokenizer=None  # Optional pre-loaded tokenizer
) -> Dict[str, Any]:
    """
    Compute MetricX-24 scores for aligned segments.
    
    Args:
        segments: List of (source, translation, reference) tuples
        metricx_model_path: Path to MetricX-24 model directory (auto-detected if None, ignored if metric_model provided)
        mt5_tokenizer_path: Path to mT5 tokenizer directory (auto-detected if None, ignored if tokenizer provided)
        metric_model: Optional pre-loaded AutoModelForSeq2SeqLM model (reuses if provided)
        tokenizer: Optional pre-loaded AutoTokenizer (reuses if provided)
    
    Returns:
        Dictionary with:
        - 'avg_metricx': Average MetricX score
        - 'min_metricx': Minimum MetricX score
        - 'max_metricx': Maximum MetricX score
        - 'scores': List of per-segment scores
    """
    print(f"  Computing MetricX-24 scores for {len(segments)} segments...")
    
    # Use pre-loaded model and tokenizer if provided
    if metric_model is not None and tokenizer is not None:
        print(f"  Using pre-loaded MetricX-24 model and tokenizer...")
        model = metric_model
        # Tokenizer is already loaded
    else:
        # Find model path
        if metricx_model_path is None:
            metricx_model_path = _find_metricx_model()
        
        if metricx_model_path is None:
            raise FileNotFoundError(
                "MetricX-24 model not found. Please ensure the model is available.\n"
                "Expected locations:\n"
                "  - /mnt/custom-file-systems/efs/.../HF_models/metricx-24-hybrid-large-v2p6-bfloat16\n"
                "  - ~/user-default-efs/HF_models/metricx-24-hybrid-large-v2p6-bfloat16"
            )
        
        metricx_model_path = Path(metricx_model_path).resolve()
        
        if not metricx_model_path.exists():
            raise FileNotFoundError(f"MetricX-24 model not found at {metricx_model_path}")
        
        # Find mT5 tokenizer path
        if mt5_tokenizer_path is None:
            mt5_tokenizer_path = _find_mt5_tokenizer()
        
        if mt5_tokenizer_path is None:
            raise FileNotFoundError(
                "mT5 tokenizer not found. MetricX-24 requires mT5 tokenizer files.\n"
                "Please download mT5 tokenizer files (spiece.model, tokenizer_config.json)\n"
                "Expected locations:\n"
                "  - /mnt/custom-file-systems/efs/.../HF_models/mt5-base\n"
                "  - ~/user-default-efs/HF_models/mt5-base"
            )
        
        mt5_tokenizer_path = Path(mt5_tokenizer_path).resolve()
        
        # Set environment variables for offline mode
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        
        # Use transformers directly (default approach)
        # We don't install SEGALE/metricx24 package, so we use transformers
        print(f"  Loading MetricX-24 model from {metricx_model_path}...")
        
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        # Load tokenizer from mT5 path
        print(f"  Loading mT5 tokenizer from {mt5_tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(
            str(mt5_tokenizer_path),
            local_files_only=True
        )
        
        # Load MetricX-24 model
        print(f"  Loading MetricX-24 model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(metricx_model_path),
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.to('cuda')
            print(f"  GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  MetricX-24 will use GPU")
        else:
            print(f"  No GPU available, using CPU")
    
    # Prepare data for MetricX-24
    try:
        # MetricX-24 expects input format: "source: {src} target: {tgt} reference: {ref}"
        sources = [seg[0] for seg in segments]
        translations = [seg[1] for seg in segments]
        references = [seg[2] for seg in segments]
        
        # Format inputs for MetricX-24 (same format as metricx24.predict expects)
        # Handle over-translation (empty source): assign penalty score of 0.0
        # Handle under-translation (empty target): MetricX will naturally penalize (low score)
        inputs = []
        over_translation_indices = []  # Track indices with empty source (over-translation)
        
        for idx, (src, tgt, ref) in enumerate(zip(sources, translations, references)):
            if not src or not src.strip():
                # Over-translation: empty source, target exists
                # Per SEGALE paper: assign worst possible score (25 for MetricX, 0 for COMET)
                # MetricX range is [0, 25] where 25 is worst (higher = worse quality)
                over_translation_indices.append(idx)
                # Still create input for consistency, but we'll replace the score
                inputs.append(f"source:  target: {tgt} reference: {ref}")
            else:
                inputs.append(f"source: {src} target: {tgt} reference: {ref}")
        
        # Process in batches to avoid GPU OOM
        # MetricX-24 can handle large batches, but we'll use smaller batches for safety
        batch_size = 32  # Adjust based on GPU memory
        all_scores = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            
            # Tokenize batch
            encoded = tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1536,  # Same as SEGALE
            )
            
            if torch.cuda.is_available():
                encoded = {k: v.to('cuda') for k, v in encoded.items()}
            
            # Generate scores (MetricX-24 outputs a score as text)
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=10,
                    num_beams=1,  # Greedy decoding for speed
                    do_sample=False,
                )
            
            # Decode the generated tokens to get score strings
            generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Parse scores from generated text
            import re
            batch_start_idx = i
            for batch_idx, text in enumerate(generated_texts):
                global_idx = batch_start_idx + batch_idx
                
                # Check if this is an over-translation segment (empty source)
                if global_idx in over_translation_indices:
                    # Over-translation: assign worst possible score (25 for MetricX)
                    # Per SEGALE paper: "For each null alignment, the worst possible score is assigned:
                    # 0 for COMET, 25 for MetricX"
                    # MetricX range is [0, 25] where 25 is worst (higher = worse quality)
                    all_scores.append(25.0)
                    continue
                
                try:
                    # Extract the first number from the generated text
                    numbers = re.findall(r'\d+\.?\d*', text.strip())
                    if numbers:
                        score = float(numbers[0])
                        # Clip to [0, 25] range as per MetricX-24 documentation
                        score = max(0.0, min(25.0, score))
                        all_scores.append(score)
                    else:
                        print(f"  ⚠ Warning: Could not parse score from: '{text}'")
                        all_scores.append(0.0)
                except Exception as e:
                    print(f"  ⚠ Warning: Error parsing score from '{text}': {e}")
                    all_scores.append(0.0)
            
            # Clear GPU cache after each batch to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        scores = all_scores
        
        # Compute statistics
        if not scores:
            return {
                'avg_metricx': None,
                'min_metricx': None,
                'max_metricx': None,
                'scores': []
            }
        
        avg_metricx = sum(scores) / len(scores)
        min_metricx = min(scores)
        max_metricx = max(scores)
        
        return {
            'avg_metricx': avg_metricx,
            'min_metricx': min_metricx,
            'max_metricx': max_metricx,
            'scores': scores
        }
        
    except Exception as e:
        raise RuntimeError(
            f"Could not load or run MetricX-24 model: {e}\n"
            f"Model path: {metricx_model_path}\n"
            f"Tokenizer path: {mt5_tokenizer_path}"
        )

