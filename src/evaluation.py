"""
Evaluation functions for translation quality.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List


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
        
        # chrF++ with char_order=6, word_order=2 (matching WMT25 evaluation)
        chrf = sacrebleu.corpus_chrf(
            [hypothesis],
            [[reference]],
            char_order=6,
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


def compute_bleu(
    hypothesis: str,
    reference: str,
    target_lang: str,
    max_ngram_order: int = 4
) -> Dict[str, float]:
    """
    Compute BLEU score using sacrebleu.
    
    Args:
        hypothesis: Translated text
        reference: Reference translation
        target_lang: Target language code (e.g., "en", "zht")
        max_ngram_order: Maximum n-gram order (default: 4 for BLEU-4)
    
    Returns:
        Dictionary with 'score' key
    """
    try:
        import sacrebleu
        from sacrebleu.metrics.bleu import BLEU
        
        # Select tokenizer based on target language
        if target_lang in ["zht", "zh"]:
            bleu_tokenizer = "zh"  # Chinese (both traditional and simplified)
        elif target_lang == "ko":
            bleu_tokenizer = "ko-mecab"  # Korean
        else:
            bleu_tokenizer = "13a"  # Default for other languages
        
        bleu_metric = BLEU(max_ngram_order=max_ngram_order, tokenize=bleu_tokenizer)
        bleu_score = bleu_metric.corpus_score([hypothesis], [[reference]])
        
        return {
            "score": bleu_score.score
        }
    except ImportError:
        # If sacrebleu is not available, return None
        return {
            "score": None
        }
    except Exception as e:
        # Handle any other errors gracefully
        return {
            "score": None
        }


def compute_term_success_rate(
    source_text: str,
    hypothesis: str,
    reference: str,
    terminology: Dict[str, List[str]],
    lowercase: bool = True
) -> float:
    """
    Compute term success rate based on WMT25 terminology track evaluation.
    
    Args:
        source_text: Source text
        hypothesis: Translated text (hypothesis)
        reference: Reference translation
        terminology: Terminology dictionary mapping source terms to target term lists
        lowercase: Whether to lowercase strings for matching (default: True)
    
    Returns:
        Term success rate (0.0 to 1.0), or -1.0 if no valid terms found
    """
    if not terminology:
        return -1.0
    
    valid_src_terms = 0.0
    aggregated_success_rate = 0.0
    
    # Prepare strings for matching
    src_str = source_text.lower() if lowercase else source_text
    hyp_str = hypothesis.lower() if lowercase else hypothesis
    ref_str = reference.lower() if lowercase else reference
    
    for src_term, trg_terms in terminology.items():
        src_term = src_term.strip()
        if not src_term:
            continue
        
        # Prepare target terms
        trg_terms = [t.strip() for t in trg_terms if t.strip()]
        if not trg_terms:
            continue
        
        # Only compute success rate for a source term if:
        # 1. It appears in the source sentence
        # 2. Any of the corresponding target terms appear in the reference
        src_term_lower = src_term.lower() if lowercase else src_term
        trg_terms_lower = [t.lower() if lowercase else t for t in trg_terms]
        
        if src_term_lower in src_str and any(t in ref_str for t in trg_terms_lower):
            valid_src_terms += 1
            
            # Count occurrences
            input_term_count = src_str.count(src_term_lower)
            output_terms_count = sum(hyp_str.count(t) for t in trg_terms_lower)
            
            # Success rate is min(1.0, output_count / input_count)
            success_rate = min(1.0, output_terms_count / input_term_count) if input_term_count > 0 else 0.0
            aggregated_success_rate += success_rate
    
    # Return average success rate, or -1.0 if no valid terms
    if valid_src_terms > 0:
        return aggregated_success_rate / valid_src_terms
    else:
        return -1.0

