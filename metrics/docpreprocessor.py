"""
Document preprocessor for splitting and aligning documents using LaBSE embeddings.

Adapted from WMT25-Term term-consistency approach to work with our data format.
"""

import pandas as pd
import json
import re
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from sentence_transformers import SentenceTransformer
from polyfuzz import PolyFuzz
from polyfuzz.models import Embeddings
from flair.embeddings import TransformerWordEmbeddings
from flair.embeddings import SentenceTransformerDocumentEmbeddings
import spacy
from nltk.tokenize import word_tokenize

# Set environment variables to prevent HuggingFace connections
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Map language codes to spaCy model names
SPACY_MODEL_MAP = {
    'en': 'en_core_web_sm',
    'de': 'de_core_news_sm',
    'es': 'es_core_news_sm',
    'fr': 'fr_core_news_sm',
    'it': 'it_core_news_sm',
    'zh': 'zh_core_web_sm',
    'zht': 'zh_core_web_sm',  # Traditional Chinese uses same model
}

def find_spacy_model(model_name: str) -> Optional[Path]:
    """
    Find spaCy model in local directories (metrics/models/spacy/) or default location.
    
    Checks in order:
    1. metrics/models/spacy/{model_name} (local repo)
    2. Default spaCy location
    """
    # Check local metrics/models directory first
    local_paths = [
        Path(__file__).parent / "models" / "spacy" / model_name,
        Path(__file__).parent.parent / "metrics" / "models" / "spacy" / model_name,
    ]
    
    for path in local_paths:
        if path.exists():
            return path
    
    # Try default spaCy location
    try:
        import spacy.util
        default_path = spacy.util.find_model(model_name)
        if default_path and Path(default_path).exists():
            return Path(default_path)
    except Exception:
        pass
    
    return None

# Global cache for loaded models and embeddings (loaded once, reused everywhere)
_loaded_models_cache = {}
_loaded_embeddings_cache = {}
_loaded_polyfuzz_cache = {}


def _log_with_time(message: str):
    """Log message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def find_labse_model_path(labse_model_path: Optional[Path] = None) -> Path:
    """Find LaBSE model path from common locations."""
    if labse_model_path is not None:
        return Path(labse_model_path).resolve()
    
    possible_paths = [
        Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/LaBSE"),
        Path.home() / "user-default-efs" / "HF_models" / "LaBSE",
        Path("/mnt/custom-file-systems/efs") / "HF_models" / "LaBSE",
        Path.home() / "Documents" / "Code" / "HF_models" / "LaBSE",  # Local path
    ]
    
    for path in possible_paths:
        if path.exists():
            return Path(path).resolve()
    
    raise FileNotFoundError(
        f"LaBSE model not found. Tried paths:\n" + "\n".join(f"  - {p}" for p in possible_paths)
    )


def load_labse_model_once(
    labse_model_path: Optional[Path] = None,
    use_gpu: bool = True
) -> SentenceTransformer:
    """
    Load LaBSE model once and cache it. Reuses the same model instance.
    
    Args:
        labse_model_path: Path to LaBSE model directory
        use_gpu: Whether to use GPU if available
        
    Returns:
        SentenceTransformer instance
    """
    # Check cache first
    cache_key = str(labse_model_path) if labse_model_path else "default"
    if cache_key in _loaded_models_cache:
        _log_with_time(f"  Using cached LaBSE model from {cache_key}")
        return _loaded_models_cache[cache_key]
    
    # Find model path
    labse_model_path = find_labse_model_path(labse_model_path)
    _log_with_time(f"Loading LaBSE model from: {labse_model_path}")
    
    # Check sentence-transformers version
    try:
        import sentence_transformers
        _log_with_time(f"  sentence-transformers version: {sentence_transformers.__version__}")
    except:
        _log_with_time("  ⚠ Could not determine sentence-transformers version")
    
    # Load SentenceTransformer from local path
    _log_with_time("  Loading SentenceTransformer...")
    try:
        # Try with local_files_only=False first (offline env vars should prevent downloads)
        labse_model = SentenceTransformer(str(labse_model_path), local_files_only=False)
        _log_with_time("  ✓ SentenceTransformer loaded successfully")
    except Exception as e1:
        _log_with_time(f"  First attempt failed: {type(e1).__name__}: {e1}")
        # Try with local_files_only=True as fallback
        try:
            labse_model = SentenceTransformer(str(labse_model_path), local_files_only=True)
            _log_with_time("  ✓ SentenceTransformer loaded successfully (with local_files_only=True)")
        except Exception as e2:
            _log_with_time(f"  Second attempt also failed: {type(e2).__name__}: {e2}")
            raise RuntimeError(
                f"Could not load SentenceTransformer from {labse_model_path}\n"
                f"First error ({type(e1).__name__}): {e1}\n"
                f"Second error ({type(e2).__name__}): {e2}"
            ) from e2
    
    # Move to GPU if available and requested
    if use_gpu:
        import torch
        if torch.cuda.is_available():
            _log_with_time(f"  Moving to GPU: {torch.cuda.get_device_name(0)}")
            labse_model = labse_model.to('cuda')
            _log_with_time("  ✓ LaBSE on GPU")
        else:
            _log_with_time("  Using CPU (GPU not available)")
    else:
        _log_with_time("  Using CPU")
    
    # Cache it
    _loaded_models_cache[cache_key] = labse_model
    _log_with_time("  ✓ Model cached for reuse")
    
    return labse_model


def load_embeddings_wrapper_once(
    labse_model_path: Optional[Path] = None,
    labse_model: Optional[SentenceTransformer] = None
) -> SentenceTransformerDocumentEmbeddings:
    """
    Load embeddings wrapper once and cache it. Reuses the same instance.
    
    Args:
        labse_model_path: Path to LaBSE model directory (used if labse_model not provided)
        labse_model: Optional pre-loaded SentenceTransformer model
        
    Returns:
        SentenceTransformerDocumentEmbeddings instance
    """
    # Check cache first
    cache_key = str(labse_model_path) if labse_model_path else "default"
    if cache_key in _loaded_embeddings_cache:
        _log_with_time(f"  Using cached embeddings wrapper from {cache_key}")
        return _loaded_embeddings_cache[cache_key]
    
    # Get model path (needed for SentenceTransformerDocumentEmbeddings)
    if labse_model_path is None:
        labse_model_path = find_labse_model_path()
    
    labse_model_path = Path(labse_model_path).resolve()
    
    _log_with_time("  Creating embeddings wrapper for PolyFuzz...")
    
    # SentenceTransformerDocumentEmbeddings needs a path, not a model object
    # So we use the path even if we have the model loaded
    try:
        embeddings = SentenceTransformerDocumentEmbeddings(str(labse_model_path))
        _log_with_time("  ✓ Created SentenceTransformerDocumentEmbeddings from path")
    except Exception as e:
        _log_with_time(f"  ✗ Could not create embeddings wrapper: {e}")
        raise RuntimeError(
            f"Could not create SentenceTransformerDocumentEmbeddings from {labse_model_path}\n"
            f"Error: {e}"
        ) from e
    
    # Cache it
    _loaded_embeddings_cache[cache_key] = embeddings
    _log_with_time("  ✓ Embeddings wrapper cached for reuse")
    
    return embeddings


def load_polyfuzz_model_once(
    labse_model_path: Optional[Path] = None,
    embeddings: Optional[SentenceTransformerDocumentEmbeddings] = None
) -> PolyFuzz:
    """
    Load PolyFuzz model once and cache it. Reuses the same instance.
    
    Args:
        labse_model_path: Path to LaBSE model directory (used if embeddings not provided)
        embeddings: Optional pre-loaded SentenceTransformerDocumentEmbeddings
        
    Returns:
        PolyFuzz instance
    """
    # Check cache first
    cache_key = str(labse_model_path) if labse_model_path else "default"
    if cache_key in _loaded_polyfuzz_cache:
        _log_with_time(f"  Using cached PolyFuzz model from {cache_key}")
        return _loaded_polyfuzz_cache[cache_key]
    
    # Load embeddings if not provided
    if embeddings is None:
        embeddings = load_embeddings_wrapper_once(labse_model_path)
    
    _log_with_time("  Creating PolyFuzz model...")
    LaBSE_embeddings = Embeddings(embeddings, min_similarity=0, model_id="LaBSE")
    polyfuzz_model = PolyFuzz([LaBSE_embeddings])
    _log_with_time("  ✓ PolyFuzz model created")
    
    # Cache it
    _loaded_polyfuzz_cache[cache_key] = polyfuzz_model
    _log_with_time("  ✓ PolyFuzz model cached for reuse")
    
    return polyfuzz_model


class DocPreprocessor:
    """
    Document preprocessor that splits and aligns documents using LaBSE embeddings.
    
    Adapted to work with our data format (accepts documents directly, not from files).
    """

    def __init__(
        self, 
        src_lang: str, 
        tgt_lang: str, 
        labse_model_path: Optional[Path] = None,
        labse_model: Optional[SentenceTransformer] = None,
        embeddings: Optional[SentenceTransformerDocumentEmbeddings] = None,
        polyfuzz_model: Optional[PolyFuzz] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the document preprocessor.
        
        Args:
            src_lang: Source language code (e.g., 'en', 'zht')
            tgt_lang: Target language code (e.g., 'zht', 'es')
            labse_model_path: Optional path to local LaBSE model (default: auto-detect)
            labse_model: Optional pre-loaded SentenceTransformer model (reuses if provided)
            embeddings: Optional pre-loaded SentenceTransformerDocumentEmbeddings (reuses if provided)
            polyfuzz_model: Optional pre-loaded PolyFuzz model (reuses if provided)
            use_gpu: Whether to use GPU if available
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Load or reuse LaBSE model (use centralized loader to avoid multiple loads)
        if labse_model is not None:
            _log_with_time("  Using provided LaBSE model (reusing existing, no reload)")
            self.labse_model = labse_model
        else:
            # Load once using the centralized loader (cached, reused across instances)
            self.labse_model = load_labse_model_once(labse_model_path, use_gpu=use_gpu)
        
        # Load or reuse PolyFuzz model (use centralized loader)
        if polyfuzz_model is not None:
            _log_with_time("  Using provided PolyFuzz model (reusing existing, no reload)")
            self.model = polyfuzz_model
        else:
            # Load embeddings wrapper if not provided
            if embeddings is None:
                # Get model path for embeddings wrapper (needs path, not model object)
                if labse_model_path is None:
                    labse_model_path = find_labse_model_path()
                embeddings = load_embeddings_wrapper_once(labse_model_path, self.labse_model)
            
            # Load PolyFuzz model using centralized loader
            self.model = load_polyfuzz_model_once(labse_model_path, embeddings)
        
        _log_with_time("  ✓ DocPreprocessor initialized with cached models")
        
        # Initialize spaCy for English normalization (if needed)
        self.spacy_en = None
        if src_lang == 'en':
            try:
                # Try to find model in local directory first
                model_path = find_spacy_model('en_core_web_sm')
                if model_path:
                    self.spacy_en = spacy.load(str(model_path), disable=['parser', 'ner'])
                else:
                    self.spacy_en = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            except (OSError, IOError) as e:
                _log_with_time(f"  ⚠ Warning: Could not load spaCy English model: {e}")
                _log_with_time(f"  ⚠ English normalization will use lowercase fallback")
            except Exception as e:
                _log_with_time(f"  ⚠ Warning: Error loading spaCy English model: {e}")
                _log_with_time(f"  ⚠ English normalization will use lowercase fallback")

    def _filter_empty_lines(self, text: str) -> str:
        """
        Filter out empty lines from text (for markdown documents).
        Empty lines are lines that contain only whitespace.
        
        Args:
            text: Input text (may contain empty lines)
        
        Returns:
            Text with empty lines removed
        """
        if not text:
            return text
        
        lines = text.split('\n')
        # Filter out lines that are empty or contain only whitespace
        filtered_lines = [line for line in lines if line.strip()]
        return '\n'.join(filtered_lines)
    
    def process_documents(
        self,
        documents: List[Tuple[str, str]],
        references: Optional[List[str]] = None,
        terminology: Optional[Dict[str, list]] = None,
        similarity_threshold: float = 0.4,
        separator: str = '\n'  # Single newline after filtering empty lines
    ) -> pd.DataFrame:
        """
        Process a list of document pairs (source, target) and return aligned segments.
        Optionally also aligns references if provided.
        
        Args:
            documents: List of (source_text, target_text) tuples
            references: Optional list of reference_text strings (same order as documents)
            terminology: Optional terminology dictionary (for WMT25-Term)
            similarity_threshold: Threshold for LaBSE similarity (default: 0.4)
            separator: Paragraph separator (default: '\n\n')
        
        Returns:
            DataFrame with columns: [paragraph, sentence, alignment, src_segment, tgt_segment, ref_segment, score, terms]
            (ref_segment is None if references not provided)
        """
        df_data = []
        skipped_segments_count = 0  # Track segments skipped due to missing src-ref alignment
        
        _log_with_time(f"  Processing {len(documents)} document(s)...")
        start_time = time.time()
        
        # Preprocess: filter empty lines from all documents (for markdown format)
        _log_with_time("  Preprocessing: filtering empty lines from documents...")
        documents = [(self._filter_empty_lines(src), self._filter_empty_lines(tgt)) for src, tgt in documents]
        if references:
            references = [self._filter_empty_lines(ref) if ref else ref for ref in references]
        
        # Add progress bar for document processing
        try:
            from tqdm import tqdm
            doc_iterator = tqdm(enumerate(documents), total=len(documents), desc="  Aligning documents")
        except ImportError:
            doc_iterator = enumerate(documents)
        
        for doc_idx, (src_text, tgt_text) in doc_iterator:
            doc_start = time.time()
            # Get reference text if provided (should have same structure as source)
            ref_text = references[doc_idx] if references and doc_idx < len(references) else None
            
            # Split into paragraphs
            _log_with_time(f"    Document {doc_idx+1}/{len(documents)}: Splitting paragraphs...")
            src_paragraphs, tgt_paragraphs = self._paragraph_aligner(
                src_text, tgt_text, separator=separator
            )
            
            # Filter out any empty paragraphs (shouldn't happen after filtering, but safety check)
            # This prevents "empty Sentence" warnings from Flair/PolyFuzz
            src_paragraphs = [p for p in src_paragraphs if p and p.strip()]
            tgt_paragraphs = [p for p in tgt_paragraphs if p and p.strip()]
            
            # Align reference paragraphs with source paragraphs using LaBSE
            # Default: If same count, use 1-to-1. Else: align using LaBSE
            ref_paragraphs = None
            ref_para_alignment = None  # Track which ref para aligns with which src para
            if ref_text:
                ref_paragraphs_raw = [p.strip() for p in ref_text.split(separator) if p.strip()]
                # Filter out empty paragraphs
                ref_paragraphs_raw = [p for p in ref_paragraphs_raw if p and p.strip()]
                if ref_paragraphs_raw:
                    if len(ref_paragraphs_raw) == len(src_paragraphs):
                        # Same count: use 1-to-1 correspondence (no alignment needed)
                        ref_paragraphs = ref_paragraphs_raw
                        ref_para_alignment = list(range(len(ref_paragraphs_raw)))
                        _log_with_time(f"      Source and reference have same paragraph count ({len(src_paragraphs)}), using 1-to-1 correspondence")
                    else:
                        # Different counts: align using LaBSE
                        _log_with_time(f"      Aligning {len(ref_paragraphs_raw)} reference paragraphs with {len(src_paragraphs)} source paragraphs (counts differ)...")
                        ref_paragraphs, ref_para_alignment = self._align_reference_paragraphs(
                            src_paragraphs, ref_paragraphs_raw
                        )
                        # Check alignment quality (only warn if counts differed and alignment failed)
                        unaligned_src = sum(1 for align in ref_para_alignment if align is None)
                        if unaligned_src > 0:
                            _log_with_time(f"      ⚠ WARNING: {unaligned_src} source paragraphs have no aligned reference paragraph")
                else:
                    _log_with_time(f"      ⚠ WARNING: Reference text has no paragraphs after splitting")
                    ref_paragraphs = []
                    ref_para_alignment = [None] * len(src_paragraphs)
            
            _log_with_time(f"      Source: {len(src_paragraphs)} paragraphs, Target: {len(tgt_paragraphs)} paragraphs" + 
                          (f", Reference: {len(ref_paragraphs)} paragraphs" if ref_paragraphs else ""))
            
            # Align paragraphs (source -> translation) using LaBSE
            # No sentence-level segmentation - work at paragraph level only
            paragraph_alignments = self._align_paragraphs(
                src_paragraphs, tgt_paragraphs, similarity_threshold
            )
            
            # Extract terms if terminology is provided (at paragraph level)
            terms_dict = {}
            if terminology:
                for para_idx, (src_para, tgt_para) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                    para_terms = self._extract_terms(src_para, tgt_para, terminology)
                    if para_terms:
                        terms_dict[para_idx] = para_terms
            
            # Process aligned paragraph pairs
            for align_idx, (src_seg, tgt_seg, score, src_para_idx) in enumerate(paragraph_alignments):
                # Get reference segment for this source paragraph
                ref_seg = ""
                has_ref_seg = False
                
                if ref_paragraphs and ref_para_alignment and src_para_idx is not None:
                    # Find the reference paragraph that aligns with this source paragraph
                    if src_para_idx < len(ref_para_alignment):
                        ref_para_idx = ref_para_alignment[src_para_idx]
                        if ref_para_idx is not None and ref_para_idx < len(ref_paragraphs):
                            ref_seg = ref_paragraphs[ref_para_idx]
                            has_ref_seg = True
                
                # If src_para_idx is None AND src_seg is empty, this is an unaligned target paragraph (over-translation)
                # For over-translation, we don't have a source, so we can't get a reference
                # This is correct - over-translated segments will be counted but won't have reference
                
                # IMPORTANT: Skip segments without src-ref correspondence (can't evaluate)
                # BUT: Always include src-tgt segments (even if tgt is empty) to penalize under-translation
                if src_seg and not has_ref_seg:
                    # Source segment without reference alignment - skip this segment
                    # (We'll count these and report them)
                    skipped_segments_count += 1
                    continue
                
                # Get terms for this paragraph if available
                terms = terms_dict.get(src_para_idx) if src_para_idx is not None else None
                
                df_data.append({
                    'document': doc_idx,  # Document/sample index
                    'paragraph': align_idx,  # Alignment index (replaces paragraph/sentence hierarchy)
                    'sentence': 0,  # Not used anymore, kept for compatibility
                    'alignment': 'labse',
                    'src_segment': src_seg,
                    'tgt_segment': tgt_seg,  # Can be empty (under-translation) - this is OK, will be penalized
                    'ref_segment': ref_seg,  # Reference segment (aligned with source)
                    'has_ref_alignment': has_ref_seg,  # Flag: True if reference was successfully aligned
                    'score': score,
                    'terms': terms
                })
            
            doc_time = time.time() - doc_start
            _log_with_time(f"    Document {doc_idx+1} processed in {doc_time:.2f}s")
        
        total_time = time.time() - start_time
        _log_with_time(f"  Processed {len(documents)} document(s) in {total_time:.2f}s")
        
        if skipped_segments_count > 0:
            _log_with_time(f"  ⚠ WARNING: Skipped {skipped_segments_count} segments due to missing src-ref alignment")
        
        # Create DataFrame
        self.df = pd.DataFrame(df_data)
        # Store skipped count as attribute for reporting
        self.skipped_segments_count = skipped_segments_count
        return self.df
    
    def _paragraph_aligner(self, src_text: str, tgt_text: str, separator: str = '\n\n') -> Tuple[List[str], List[str]]:
        """Align paragraphs between source and target using LaBSE embeddings."""
        src_paragraphs = [p.strip() for p in src_text.split(separator) if p.strip()]
        tgt_paragraphs = [p.strip() for p in tgt_text.split(separator) if p.strip()]
        
        if len(src_paragraphs) == len(tgt_paragraphs):
            # Simple 1-to-1 alignment
            return src_paragraphs, tgt_paragraphs
        
        # LaBSE-based alignment (many-to-many)
        _log_with_time(f"      Using LaBSE alignment (src: {len(src_paragraphs)}, tgt: {len(tgt_paragraphs)})")
        alignment = 'labse'
        
        # Use PolyFuzz to align paragraphs
        try:
            matches = self.model.match(src_paragraphs, tgt_paragraphs)
            
            # Extract aligned pairs
            aligned_src = []
            aligned_tgt = []
            
            # Simple greedy alignment based on similarity scores
            used_tgt_indices = set()
            for src_idx, src_para in enumerate(src_paragraphs):
                best_tgt_idx = None
                best_score = 0.0
                
                for tgt_idx, tgt_para in enumerate(tgt_paragraphs):
                    if tgt_idx in used_tgt_indices:
                        continue
                    
                    # Get similarity score
                    score = matches['Similarity'].iloc[src_idx * len(tgt_paragraphs) + tgt_idx] if hasattr(matches, 'iloc') else 0.0
                    
                    if score > best_score and score >= 0.4:  # similarity_threshold
                        best_score = score
                        best_tgt_idx = tgt_idx
                
                if best_tgt_idx is not None:
                    aligned_src.append(src_para)
                    aligned_tgt.append(tgt_paragraphs[best_tgt_idx])
                    used_tgt_indices.add(best_tgt_idx)
                else:
                    # Unaligned source paragraph
                    aligned_src.append(src_para)
                    aligned_tgt.append("")  # Empty target
            
            # Add unaligned target paragraphs
            for tgt_idx, tgt_para in enumerate(tgt_paragraphs):
                if tgt_idx not in used_tgt_indices:
                    aligned_src.append("")  # Empty source
                    aligned_tgt.append(tgt_para)
            
            return aligned_src, aligned_tgt
            
        except Exception as e:
            _log_with_time(f"      ⚠ LaBSE alignment failed: {e}, using simple 1-to-1")
            # Fallback: simple 1-to-1 alignment
            min_len = min(len(src_paragraphs), len(tgt_paragraphs))
            return src_paragraphs[:min_len], tgt_paragraphs[:min_len]
    
    def _align_reference_paragraphs(
        self, 
        src_paragraphs: List[str], 
        ref_paragraphs: List[str],
        similarity_threshold: float = 0.4
    ) -> Tuple[List[str], List[Optional[int]]]:
        """
        Align reference paragraphs with source paragraphs using LaBSE.
        
        Returns:
            Tuple of (aligned_ref_paragraphs, alignment_mapping)
            - aligned_ref_paragraphs: List of reference paragraphs in source order (empty string if unaligned)
            - alignment_mapping: List mapping src_para_idx -> ref_para_idx (None if unaligned)
        """
        # Filter out any empty paragraphs before passing to PolyFuzz (prevents "empty Sentence" warnings)
        src_paragraphs = [p for p in src_paragraphs if p and p.strip()]
        ref_paragraphs = [p for p in ref_paragraphs if p and p.strip()]
        
        if len(src_paragraphs) == len(ref_paragraphs):
            # Simple 1-to-1 alignment (assume same structure)
            return ref_paragraphs, list(range(len(ref_paragraphs)))
        
        # LaBSE-based alignment
        try:
            matches = self.model.match(src_paragraphs, ref_paragraphs)
            
            aligned_ref = []
            alignment_mapping = []
            used_ref_indices = set()
            
            for src_idx, src_para in enumerate(src_paragraphs):
                best_ref_idx = None
                best_score = 0.0
                
                for ref_idx, ref_para in enumerate(ref_paragraphs):
                    if ref_idx in used_ref_indices:
                        continue
                    
                    try:
                        score = float(matches['Similarity'].iloc[src_idx * len(ref_paragraphs) + ref_idx]) if hasattr(matches, 'iloc') else 0.0
                    except:
                        score = 0.0
                    
                    if score > best_score and score >= similarity_threshold:
                        best_score = score
                        best_ref_idx = ref_idx
                
                if best_ref_idx is not None:
                    aligned_ref.append(ref_paragraphs[best_ref_idx])
                    alignment_mapping.append(best_ref_idx)
                    used_ref_indices.add(best_ref_idx)
                else:
                    # Unaligned source paragraph
                    aligned_ref.append("")
                    alignment_mapping.append(None)
            
            return aligned_ref, alignment_mapping
            
        except Exception as e:
            _log_with_time(f"      ⚠ Reference paragraph alignment failed: {e}, using simple 1-to-1")
            # Fallback: simple 1-to-1 alignment
            min_len = min(len(src_paragraphs), len(ref_paragraphs))
            aligned_ref = ref_paragraphs[:min_len] + [""] * (len(src_paragraphs) - min_len)
            alignment_mapping = list(range(min_len)) + [None] * (len(src_paragraphs) - min_len)
            return aligned_ref, alignment_mapping
    
    def _align_reference_sentences(
        self,
        src_sentences: List[str],
        ref_sentences: List[str],
        similarity_threshold: float = 0.4
    ) -> Tuple[List[str], List[Optional[int]]]:
        """
        Align reference sentences with source sentences using LaBSE.
        
        Returns:
            Tuple of (aligned_ref_sentences, alignment_mapping)
            - aligned_ref_sentences: List of reference sentences in source order (empty string if unaligned)
            - alignment_mapping: List mapping src_sent_idx -> ref_sent_idx (None if unaligned)
        """
        if len(src_sentences) == len(ref_sentences):
            # Simple 1-to-1 alignment (assume same structure)
            return ref_sentences, list(range(len(ref_sentences)))
        
        # LaBSE-based alignment
        try:
            matches = self.model.match(src_sentences, ref_sentences)
            
            aligned_ref = []
            alignment_mapping = []
            used_ref_indices = set()
            
            for src_idx, src_sent in enumerate(src_sentences):
                best_ref_idx = None
                best_score = 0.0
                
                for ref_idx, ref_sent in enumerate(ref_sentences):
                    if ref_idx in used_ref_indices:
                        continue
                    
                    try:
                        score = float(matches['Similarity'].iloc[src_idx * len(ref_sentences) + ref_idx]) if hasattr(matches, 'iloc') else 0.0
                    except:
                        score = 0.0
                    
                    if score > best_score and score >= similarity_threshold:
                        best_score = score
                        best_ref_idx = ref_idx
                
                if best_ref_idx is not None:
                    aligned_ref.append(ref_sentences[best_ref_idx])
                    alignment_mapping.append(best_ref_idx)
                    used_ref_indices.add(best_ref_idx)
                else:
                    # Unaligned source sentence
                    aligned_ref.append("")
                    alignment_mapping.append(None)
            
            return aligned_ref, alignment_mapping
            
        except Exception as e:
            _log_with_time(f"      ⚠ Reference sentence alignment failed: {e}, using simple 1-to-1")
            # Fallback: simple 1-to-1 alignment
            min_len = min(len(src_sentences), len(ref_sentences))
            aligned_ref = ref_sentences[:min_len] + [""] * (len(src_sentences) - min_len)
            alignment_mapping = list(range(min_len)) + [None] * (len(src_sentences) - min_len)
            return aligned_ref, alignment_mapping
    
    def _split_sentences(self, text: str, lang: str) -> List[str]:
        """
        Split text into sentences using spaCy if available, otherwise fallback to pysbd.
        Filters out empty sentences (whitespace-only).
        
        Args:
            text: Text to split (empty lines already filtered in preprocessing)
            lang: Language code (e.g., 'en', 'fr', 'it', 'zh')
        
        Returns:
            List of sentences (non-empty strings, filtered)
        """
        # Filter empty lines first (for markdown documents)
        text = self._filter_empty_lines(text)
        
        if not text or not text.strip():
            return []
        
        # Try spaCy first (as used in SEGALE)
        try:
            # Check if spaCy model is available for this language
            spacy_model_name = SPACY_MODEL_MAP.get(lang)
            if spacy_model_name:
                spacy_model = _load_spacy_model_once(lang)
                if spacy_model:
                    doc = spacy_model(text)
                    sentences = [sent.text.strip() for sent in doc.sents]
                    # Filter out empty sentences
                    return [s for s in sentences if s.strip()]
        except Exception:
            # spaCy not available or failed, try pysbd
            pass
        
        # Fallback to pysbd for sentence splitting
        try:
            import pysbd
            seg = pysbd.Segmenter(language=lang, clean=False)
            sentences = seg.segment(text)
            # Filter out empty sentences
            return [s.strip() for s in sentences if s.strip()]
        except Exception:
            # Final fallback: simple splitting
            sentences = re.split(r'[.!?]\s+', text)
            # Filter out empty sentences
            return [s.strip() for s in sentences if s.strip()]
    
    def _align_paragraphs(self, src_paragraphs: List[str], tgt_paragraphs: List[str], similarity_threshold: float = 0.4) -> List[Tuple[str, str, float, Optional[int]]]:
        """
        Align paragraphs using LaBSE embeddings.
        
        Returns:
            List of tuples: (src_segment, tgt_segment, score, src_para_idx)
            src_para_idx is None for unaligned target paragraphs (empty source)
        """
        if not src_paragraphs or not tgt_paragraphs:
            return []
        
        # Filter out any empty paragraphs before passing to PolyFuzz (prevents "empty Sentence" warnings)
        src_paragraphs = [p for p in src_paragraphs if p and p.strip()]
        tgt_paragraphs = [p for p in tgt_paragraphs if p and p.strip()]
        
        if not src_paragraphs or not tgt_paragraphs:
            return []
        
        # Use PolyFuzz to align paragraphs
        try:
            matches = self.model.match(src_paragraphs, tgt_paragraphs)
            
            # Extract aligned pairs with scores
            alignment = []
            used_tgt_indices = set()
            
            for src_idx, src_para in enumerate(src_paragraphs):
                best_tgt_idx = None
                best_score = 0.0
                
                for tgt_idx, tgt_para in enumerate(tgt_paragraphs):
                    if tgt_idx in used_tgt_indices:
                        continue
                    
                    # Get similarity score from PolyFuzz
                    try:
                        score = float(matches['Similarity'].iloc[src_idx * len(tgt_paragraphs) + tgt_idx]) if hasattr(matches, 'iloc') else 0.0
                    except:
                        score = 0.0
                    
                    if score > best_score and score >= similarity_threshold:
                        best_score = score
                        best_tgt_idx = tgt_idx
                
                if best_tgt_idx is not None:
                    alignment.append((src_para, tgt_paragraphs[best_tgt_idx], best_score, src_idx))
                    used_tgt_indices.add(best_tgt_idx)
                else:
                    # Unaligned source paragraph
                    alignment.append((src_para, "", 0.0, src_idx))
            
            # Add unaligned target paragraphs (no corresponding source)
            for tgt_idx, tgt_para in enumerate(tgt_paragraphs):
                if tgt_idx not in used_tgt_indices:
                    alignment.append(("", tgt_para, 0.0, None))
            
            return alignment
            
        except Exception as e:
            _log_with_time(f"      ⚠ Paragraph alignment failed: {e}")
            # Fallback: simple 1-to-1 alignment
            min_len = min(len(src_paragraphs), len(tgt_paragraphs))
            return [(src_paragraphs[i], tgt_paragraphs[i] if i < len(tgt_paragraphs) else "", 1.0, i) for i in range(min_len)]
    
    def _extract_terms(self, src_text: str, tgt_text: str, terminology: Dict[str, list]) -> Optional[Dict[str, list]]:
        """
        Extract terminology terms from source and target text.
        
        Terminology format: Dict[str, List[str]] where keys are source terms and values are lists of target terms.
        Example: {"source_term": ["target_term1", "target_term2"], ...}
        """
        if not terminology:
            return None
        
        found_terms = {}
        try:
            for src_term, tgt_terms_list in terminology.items():
                # Handle both formats:
                # 1. Dict[str, List[str]]: {"source_term": ["target1", "target2"]}
                # 2. Dict[str, List[Dict]]: {"type": [{"source": "...", "target": "..."}]}
                
                if not isinstance(tgt_terms_list, list):
                    continue
                
                found = []
                for tgt_term_item in tgt_terms_list:
                    # Check if it's a dict format (old format)
                    if isinstance(tgt_term_item, dict):
                        term_src = tgt_term_item.get('source', '')
                        term_tgt = tgt_term_item.get('target', '')
                    # Otherwise, it's a string format (WMT25 format)
                    elif isinstance(tgt_term_item, str):
                        term_src = src_term
                        term_tgt = tgt_term_item
                    else:
                        continue
                    
                    # Case-insensitive search
                    if term_src and term_tgt:
                        if term_src.lower() in src_text.lower() and term_tgt.lower() in tgt_text.lower():
                            # Store in consistent format
                            found.append({
                                'source': term_src,
                                'target': term_tgt
                            })
                
                if found:
                    # Use source term as key (or 'proper' if we want to group by type)
                    found_terms[src_term] = found
        
        except Exception as e:
            # If there's any error in term extraction, log and return None
            _log_with_time(f"      ⚠ Warning: Error extracting terms: {e}")
            return None
        
        return found_terms if found_terms else None
