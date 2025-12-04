"""
Evaluation functions for translation quality.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional


def compute_chrf(
    hypothesis: str,
    reference: str,
    chrf_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute chrF++ score using sacrebleu or sacreBLEU Python package.
    
    Args:
        hypothesis: Translated text
        reference: Reference translation
        chrf_path: Path to chrF++ script (optional, uses sacrebleu if available)
    
    Returns:
        Dictionary with 'score', 'precision', 'recall', 'fscore' keys
    """
    try:
        # Try using sacrebleu Python package
        import sacrebleu
        
        # chrF++ with default settings
        chrf = sacrebleu.corpus_chrf(
            [hypothesis],
            [[reference]],
            word_order=2
        )
        
        return {
            "score": chrf.score
        }
    except ImportError:
        # Fallback to sacrebleu CLI if available
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as hyp_file, \
                 tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', encoding='utf-8') as ref_file:
                
                hyp_file.write(hypothesis)
                hyp_file.flush()
                ref_file.write(reference)
                ref_file.flush()
                
                result = subprocess.run(
                    ["sacrebleu", ref_file.name, "-i", hyp_file.name, "-m", "chrf", "--chrf-word-order", "6"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Parse output (format: "chrF2 = XX.XX")
                score_line = result.stdout.strip().split('\n')[-1]
                score = float(score_line.split('=')[1].strip())
                
                return {
                    "score": score  
                }
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            # If sacrebleu is not available, return None
            return {
                "score": None
            }


def format_chrf_result(chrf_result: Dict[str, float]) -> str:
    """Format chrF++ result as a string."""
    if chrf_result["score"] is None:
        return "N/A"
    return f"{chrf_result['score']:.2f}"

