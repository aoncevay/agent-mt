"""
Alignment module for document-level MT evaluation.

Supports two alignment methods:
1. VecAlign (default): The original algorithm from Thompson & Koehn (2019)
   - Used by SEGALE for MT evaluation
   - Handles many-to-many alignment with overlaps
   - More rigorous, better for reproducibility

2. Simple DP: A simplified implementation for cases where VecAlign isn't available
   - Similar concept but without the overlap complexity
   - Faster but may be less accurate

Reference:
- VecAlign: https://github.com/thompsonb/vecalign
- SEGALE: https://github.com/NVIDIA/SEGALE
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer

# Import centralized configuration
from metrics.config import get_config, get_vecalign_path, get_labse_path

# Try to import VecAlign using config
VECALIGN_AVAILABLE = False
try:
    # Get vecalign path from config
    vecalign_path = get_vecalign_path()
    if vecalign_path and vecalign_path.exists():
        sys.path.insert(0, str(vecalign_path))
        from dp_utils import (
            yield_overlaps, 
            make_doc_embedding, 
            vecalign as vecalign_dp,
            make_alignment_types
        )
        VECALIGN_AVAILABLE = True
except ImportError as e:
    print(f"VecAlign not available: {e}")


def compute_similarity_matrix(
    src_embeddings: np.ndarray,
    tgt_embeddings: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity matrix between source and target embeddings.
    
    Args:
        src_embeddings: (N, dim) array of source embeddings
        tgt_embeddings: (M, dim) array of target embeddings
    
    Returns:
        (N, M) similarity matrix where [i,j] = cosine_sim(src[i], tgt[j])
    """
    # Normalize embeddings
    src_norm = src_embeddings / (np.linalg.norm(src_embeddings, axis=1, keepdims=True) + 1e-9)
    tgt_norm = tgt_embeddings / (np.linalg.norm(tgt_embeddings, axis=1, keepdims=True) + 1e-9)
    
    # Compute cosine similarity
    return np.dot(src_norm, tgt_norm.T)


# ============================================================================
# VecAlign-based alignment (default, recommended)
# ============================================================================

def _embed_overlaps(
    sentences: List[str],
    model: SentenceTransformer,
    num_overlaps: int = 4
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Create overlaps and embed them for VecAlign.
    
    Args:
        sentences: List of sentences/segments
        model: SentenceTransformer for embeddings
        num_overlaps: Maximum overlap size (default: 4)
    
    Returns:
        (sent2line, embeddings): Mapping from text to index, and embedding array
    """
    # Generate all overlaps
    overlaps = list(yield_overlaps(sentences, num_overlaps))
    
    # Create mapping
    sent2line = {text: idx for idx, text in enumerate(overlaps)}
    
    # Embed all overlaps
    embeddings = model.encode(overlaps, convert_to_numpy=True)
    
    return sent2line, embeddings


def align_vecalign(
    src_segments: List[str],
    tgt_segments: List[str],
    model: SentenceTransformer,
    alignment_max_size: int = 4,
    del_percentile_frac: float = 0.2,
    num_overlaps: int = 4
) -> List[Tuple[List[int], List[int], float]]:
    """
    Align segments using VecAlign algorithm.
    
    Args:
        src_segments: List of source segments
        tgt_segments: List of target segments
        model: SentenceTransformer model
        alignment_max_size: Maximum alignment size (N+M <= this value)
        del_percentile_frac: Deletion penalty percentile (0-1)
        num_overlaps: Number of overlaps to compute
    
    Returns:
        List of (src_indices, tgt_indices, cost) tuples
    """
    if not VECALIGN_AVAILABLE:
        raise RuntimeError("VecAlign is not available. Please clone vecalign to other_repos/vecalign")
    
    if not src_segments or not tgt_segments:
        return []
    
    # Embed overlaps for source and target
    src_sent2line, src_embeddings = _embed_overlaps(src_segments, model, num_overlaps)
    tgt_sent2line, tgt_embeddings = _embed_overlaps(tgt_segments, model, num_overlaps)
    
    # Make document embeddings (3D: num_overlaps x num_sentences x dim)
    src_vecs = make_doc_embedding(src_sent2line, src_embeddings, src_segments, num_overlaps)
    tgt_vecs = make_doc_embedding(tgt_sent2line, tgt_embeddings, tgt_segments, num_overlaps)
    
    # Run VecAlign
    alignment_types = make_alignment_types(alignment_max_size)
    
    stack = vecalign_dp(
        vecs0=src_vecs,
        vecs1=tgt_vecs,
        final_alignment_types=alignment_types,
        del_percentile_frac=del_percentile_frac,
        width_over2=5,  # Search buffer size
        max_size_full_dp=300,
        costs_sample_size=20000,
        num_samps_for_norm=100
    )
    
    # Extract alignments
    alignments = stack[0].get('final_alignments', stack[0].get('alignments', []))
    scores = stack[0].get('alignment_scores', [0.0] * len(alignments))
    
    # Convert to our format
    results = []
    for (src_indices, tgt_indices), score in zip(alignments, scores):
        results.append((src_indices, tgt_indices, score))
    
    return results


# ============================================================================
# Simple DP alignment (fallback)
# ============================================================================

def dp_align(
    similarity_matrix: np.ndarray,
    max_alignment_size: int = 3,
    deletion_cost: float = 0.3,
    insertion_cost: float = 0.3
) -> List[Tuple[List[int], List[int], float]]:
    """
    Dynamic programming alignment to find optimal alignment between source and target.
    
    Based on VecAlign's approach: finds the best path through the similarity matrix
    that allows for:
    - 1-to-1 alignments
    - many-to-1 alignments (up to max_alignment_size source segments to 1 target)
    - 1-to-many alignments (1 source to up to max_alignment_size target segments)
    - deletions (source segment with no target - under-translation)
    - insertions (target segment with no source - over-translation)
    
    Args:
        similarity_matrix: (N, M) matrix of similarities
        max_alignment_size: Maximum number of segments to align together
        deletion_cost: Cost penalty for unaligned source (under-translation)
        insertion_cost: Cost penalty for unaligned target (over-translation)
    
    Returns:
        List of (src_indices, tgt_indices, cost) tuples representing alignments
    """
    n_src, n_tgt = similarity_matrix.shape
    
    if n_src == 0 or n_tgt == 0:
        return []
    
    # DP cost matrix: cost[i][j] = minimum cost to align src[0:i] with tgt[0:j]
    INF = float('inf')
    cost = np.full((n_src + 1, n_tgt + 1), INF)
    backtrack = {}
    
    cost[0][0] = 0.0
    
    # Fill the DP table
    for i in range(n_src + 1):
        for j in range(n_tgt + 1):
            if cost[i][j] == INF:
                continue
            
            # 1. Deletion: align source segment(s) with nothing (under-translation)
            for di in range(1, min(max_alignment_size, n_src - i) + 1):
                new_cost = cost[i][j] + deletion_cost * di
                if new_cost < cost[i + di][j]:
                    cost[i + di][j] = new_cost
                    backtrack[(i + di, j)] = (i, j, list(range(i, i + di)), [])
            
            # 2. Insertion: align nothing with target segment(s) (over-translation)
            for dj in range(1, min(max_alignment_size, n_tgt - j) + 1):
                new_cost = cost[i][j] + insertion_cost * dj
                if new_cost < cost[i][j + dj]:
                    cost[i][j + dj] = new_cost
                    backtrack[(i, j + dj)] = (i, j, [], list(range(j, j + dj)))
            
            # 3. Alignment: align source segment(s) with target segment(s)
            for di in range(1, min(max_alignment_size, n_src - i) + 1):
                for dj in range(1, min(max_alignment_size, n_tgt - j) + 1):
                    # Compute average similarity for this alignment
                    sim_sum = 0.0
                    for si in range(i, i + di):
                        for sj in range(j, j + dj):
                            sim_sum += similarity_matrix[si][sj]
                    avg_sim = sim_sum / (di * dj)
                    
                    # Cost is inverse of similarity
                    align_cost = 1.0 - avg_sim
                    
                    # Penalty for many-to-many alignments
                    if di > 1 or dj > 1:
                        align_cost += 0.1 * (di + dj - 2)
                    
                    new_cost = cost[i][j] + align_cost
                    if new_cost < cost[i + di][j + dj]:
                        cost[i + di][j + dj] = new_cost
                        backtrack[(i + di, j + dj)] = (i, j, list(range(i, i + di)), list(range(j, j + dj)))
    
    # Backtrack
    alignments = []
    i, j = n_src, n_tgt
    
    while (i, j) != (0, 0):
        if (i, j) not in backtrack:
            break
        prev_i, prev_j, src_indices, tgt_indices = backtrack[(i, j)]
        
        # Compute step cost
        if src_indices and tgt_indices:
            sim_sum = sum(similarity_matrix[si][sj] 
                         for si in src_indices for sj in tgt_indices)
            step_cost = 1.0 - sim_sum / (len(src_indices) * len(tgt_indices))
        elif src_indices:
            step_cost = deletion_cost * len(src_indices)
        else:
            step_cost = insertion_cost * len(tgt_indices)
        
        alignments.append((src_indices, tgt_indices, step_cost))
        i, j = prev_i, prev_j
    
    alignments.reverse()
    return alignments


def align_simple_dp(
    src_segments: List[str],
    tgt_segments: List[str],
    model: SentenceTransformer,
    max_alignment_size: int = 3,
    deletion_cost: float = 0.3,
    insertion_cost: float = 0.3
) -> List[Tuple[List[int], List[int], float]]:
    """
    Align segments using simple DP with LaBSE similarity matrix.
    
    This is a fallback when VecAlign is not available.
    
    Args:
        src_segments: List of source segments
        tgt_segments: List of target segments
        model: SentenceTransformer model
        max_alignment_size: Max segments to align together
        deletion_cost: Cost for under-translation
        insertion_cost: Cost for over-translation
    
    Returns:
        List of (src_indices, tgt_indices, cost) tuples
    """
    if not src_segments or not tgt_segments:
        if src_segments:
            return [(list(range(len(src_segments))), [], 0.0)]
        elif tgt_segments:
            return [([], list(range(len(tgt_segments))), 0.0)]
        return []
    
    # Compute embeddings
    src_embeddings = model.encode(src_segments, convert_to_numpy=True)
    tgt_embeddings = model.encode(tgt_segments, convert_to_numpy=True)
    
    # Compute similarity matrix
    sim_matrix = compute_similarity_matrix(src_embeddings, tgt_embeddings)
    
    # Run DP
    return dp_align(
        sim_matrix,
        max_alignment_size=max_alignment_size,
        deletion_cost=deletion_cost,
        insertion_cost=insertion_cost
    )


# ============================================================================
# Main alignment interface
# ============================================================================

def align_segments(
    src_segments: List[str],
    tgt_segments: List[str],
    model: SentenceTransformer,
    aligner: str = "vecalign",
    max_alignment_size: int = 4,
    deletion_cost: float = 0.3,
    insertion_cost: float = 0.3
) -> List[Tuple[str, str, float, List[int], List[int]]]:
    """
    Align source and target segments using the specified aligner.
    
    Args:
        src_segments: List of source segments (paragraphs)
        tgt_segments: List of target segments (paragraphs)
        model: SentenceTransformer model for embeddings
        aligner: "vecalign" (default) or "dp"
        max_alignment_size: Maximum segments to align together
        deletion_cost: Cost for unaligned source
        insertion_cost: Cost for unaligned target
    
    Returns:
        List of (src_text, tgt_text, score, src_indices, tgt_indices) tuples
    """
    # Choose aligner
    if aligner == "vecalign":
        if VECALIGN_AVAILABLE:
            alignments = align_vecalign(
                src_segments, tgt_segments, model,
                alignment_max_size=max_alignment_size,
                del_percentile_frac=deletion_cost  # Use deletion_cost as percentile
            )
        else:
            print("  ⚠ Warning: VecAlign not available, falling back to simple DP")
            alignments = align_simple_dp(
                src_segments, tgt_segments, model,
                max_alignment_size=max_alignment_size,
                deletion_cost=deletion_cost,
                insertion_cost=insertion_cost
            )
    elif aligner == "dp":
        alignments = align_simple_dp(
            src_segments, tgt_segments, model,
            max_alignment_size=max_alignment_size,
            deletion_cost=deletion_cost,
            insertion_cost=insertion_cost
        )
    else:
        raise ValueError(f"Unknown aligner: {aligner}. Use 'vecalign' or 'dp'")
    
    # Convert to output format
    results = []
    for src_indices, tgt_indices, cost in alignments:
        src_text = " ".join([src_segments[i] for i in src_indices]) if src_indices else ""
        tgt_text = " ".join([tgt_segments[j] for j in tgt_indices]) if tgt_indices else ""
        score = max(0.0, min(1.0, 1.0 - cost))
        results.append((src_text, tgt_text, score, src_indices, tgt_indices))
    
    return results


def align_with_reference(
    src_segments: List[str],
    tgt_segments: List[str],
    ref_segments: List[str],
    model: SentenceTransformer,
    aligner: str = "vecalign",
    max_alignment_size: int = 4,
    deletion_cost: float = 0.3,
    insertion_cost: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Align source with target AND source with reference.
    
    This is the main alignment function for MT evaluation:
    1. Align source → target (to find what was translated)
    2. Use source indices to get corresponding reference segments
    
    Args:
        src_segments: List of source segments
        tgt_segments: List of target segments (translation)
        ref_segments: List of reference segments
        model: SentenceTransformer model
        aligner: "vecalign" (default) or "dp"
        max_alignment_size: Maximum segments to align together
        deletion_cost: Cost for under-translation
        insertion_cost: Cost for over-translation
    
    Returns:
        List of alignment dictionaries with keys:
        - src: Source text
        - tgt: Target text (translation)
        - ref: Reference text
        - score: Alignment score
        - src_indices: Source segment indices
        - tgt_indices: Target segment indices
        - alignment_type: 'aligned', 'under_translated', 'over_translated'
    """
    # First, align source with reference (if different lengths)
    if len(src_segments) == len(ref_segments):
        # Same structure: 1-to-1 correspondence
        src_to_ref = {i: i for i in range(len(src_segments))}
    else:
        # Different structure: need to align with embeddings
        src_embeddings = model.encode(src_segments, convert_to_numpy=True)
        ref_embeddings = model.encode(ref_segments, convert_to_numpy=True)
        sim_matrix = compute_similarity_matrix(src_embeddings, ref_embeddings)
        
        # For src-ref alignment, use 1-to-1 greedy matching
        src_to_ref = {}
        used_ref = set()
        for i in range(len(src_segments)):
            best_j = -1
            best_sim = -1
            for j in range(len(ref_segments)):
                if j not in used_ref and sim_matrix[i][j] > best_sim:
                    best_sim = sim_matrix[i][j]
                    best_j = j
            if best_j >= 0 and best_sim > 0.3:
                src_to_ref[i] = best_j
                used_ref.add(best_j)
    
    # Now align source with target
    src_tgt_alignments = align_segments(
        src_segments, tgt_segments, model,
        aligner=aligner,
        max_alignment_size=max_alignment_size,
        deletion_cost=deletion_cost,
        insertion_cost=insertion_cost
    )
    
    # Build output with reference
    results = []
    for src_text, tgt_text, score, src_indices, tgt_indices in src_tgt_alignments:
        # Get reference text for aligned source segments
        ref_texts = []
        for src_idx in src_indices:
            if src_idx in src_to_ref:
                ref_idx = src_to_ref[src_idx]
                if ref_idx < len(ref_segments):
                    ref_texts.append(ref_segments[ref_idx])
        ref_text = " ".join(ref_texts) if ref_texts else ""
        
        # Determine alignment type
        if src_indices and tgt_indices:
            alignment_type = 'aligned'
        elif src_indices and not tgt_indices:
            alignment_type = 'under_translated'
        else:
            alignment_type = 'over_translated'
        
        results.append({
            'src': src_text,
            'tgt': tgt_text,
            'ref': ref_text,
            'score': score,
            'src_indices': src_indices,
            'tgt_indices': tgt_indices,
            'alignment_type': alignment_type
        })
    
    return results


# For backward compatibility
def align_segments_dp(
    src_segments: List[str],
    tgt_segments: List[str],
    model: SentenceTransformer,
    max_alignment_size: int = 3,
    deletion_cost: float = 0.3,
    insertion_cost: float = 0.3
) -> List[Tuple[str, str, float, List[int], List[int]]]:
    """Backward compatible wrapper - uses align_segments with dp aligner."""
    return align_segments(
        src_segments, tgt_segments, model,
        aligner="dp",
        max_alignment_size=max_alignment_size,
        deletion_cost=deletion_cost,
        insertion_cost=insertion_cost
    )


def is_vecalign_available() -> bool:
    """Check if VecAlign is available."""
    return VECALIGN_AVAILABLE


def load_labse_model() -> SentenceTransformer:
    """
    Load LaBSE model from local path (no internet connection).
    
    Returns:
        SentenceTransformer model
    
    Raises:
        FileNotFoundError: If LaBSE model is not found locally
    """
    # Set offline mode
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    labse_path = get_labse_path()
    
    if labse_path and labse_path.exists():
        print(f"  Loading LaBSE from: {labse_path}")
        return SentenceTransformer(str(labse_path), local_files_only=True)
    
    # Try fallback paths from config
    config = get_config()
    hf_dir = config.get_hf_models_dir()
    
    if hf_dir:
        labse_in_hf = hf_dir / "LaBSE"
        if labse_in_hf.exists():
            print(f"  Loading LaBSE from: {labse_in_hf}")
            return SentenceTransformer(str(labse_in_hf), local_files_only=True)
    
    raise FileNotFoundError(
        "LaBSE model not found. Please:\n"
        "1. Download LaBSE locally\n"
        "2. Set HF_MODELS_DIR or LABSE_MODEL_PATH in metrics/.env\n"
        f"   Current search paths:\n"
        f"   - {labse_path or '(not configured)'}\n"
        f"   - {hf_dir / 'LaBSE' if hf_dir else '(HF_MODELS_DIR not set)'}"
    )


# Test function
if __name__ == "__main__":
    # Print configuration
    config = get_config()
    config.print_config()
    
    print(f"\nVecAlign available: {VECALIGN_AVAILABLE}")
    
    if VECALIGN_AVAILABLE:
        print("\nTesting VecAlign alignment...")
        # Simple test
        src = ["Hello world.", "How are you?", "This is a test."]
        tgt = ["Hola mundo.", "¿Cómo estás?", "Esta es una prueba."]
        
        try:
            # Load model from local path
            model = load_labse_model()
            
            results = align_segments(src, tgt, model, aligner="vecalign")
            for r in results:
                print(f"  {r[0]} -> {r[1]} (score: {r[2]:.4f})")
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
    else:
        print("\nTesting simple DP alignment...")
        try:
            model = load_labse_model()
            src = ["Hello world.", "How are you?"]
            tgt = ["Hola mundo.", "¿Cómo estás?"]
            
            results = align_segments(src, tgt, model, aligner="dp")
            for r in results:
                print(f"  {r[0]} -> {r[1]} (score: {r[2]:.4f})")
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
