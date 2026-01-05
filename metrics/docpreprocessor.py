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
import stanza
from nltk.tokenize import word_tokenize

# Set environment variables to prevent HuggingFace connections
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

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
        
        # Initialize stanza for English normalization (if needed)
        if src_lang == 'en':
            self.stanza_en = stanza.Pipeline('en', processors='tokenize,lemma', lemma_pretagged=True, tokenize_pretokenized=False)
        else:
            self.stanza_en = None

    def process_documents(
        self,
        documents: List[Tuple[str, str]],
        terminology: Optional[Dict[str, list]] = None,
        similarity_threshold: float = 0.4,
        separator: str = '\n\n'
    ) -> pd.DataFrame:
        """
        Process a list of document pairs (source, target) and return aligned segments.
        
        Args:
            documents: List of (source_text, target_text) tuples
            terminology: Optional terminology dictionary (for WMT25-Term)
            similarity_threshold: Threshold for LaBSE similarity (default: 0.4)
            separator: Paragraph separator (default: '\n\n')
        
        Returns:
            DataFrame with columns: [paragraph, sentence, alignment, src_segment, tgt_segment, score, terms]
        """
        df_data = []
        
        _log_with_time(f"  Processing {len(documents)} document(s)...")
        start_time = time.time()
        
        # Add progress bar for document processing
        try:
            from tqdm import tqdm
            doc_iterator = tqdm(enumerate(documents), total=len(documents), desc="  Aligning documents")
        except ImportError:
            doc_iterator = enumerate(documents)
        
        for doc_idx, (src_text, tgt_text) in doc_iterator:
            doc_start = time.time()
            # Split into paragraphs
            _log_with_time(f"    Document {doc_idx+1}/{len(documents)}: Splitting paragraphs...")
            src_paragraphs, tgt_paragraphs = self._paragraph_aligner(
                src_text, tgt_text, separator=separator
            )
            
            _log_with_time(f"      Source: {len(src_paragraphs)} paragraphs, Target: {len(tgt_paragraphs)} paragraphs")
            
            # Align paragraphs
            for para_idx, (src_para, tgt_para) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                # Split into sentences
                src_sentences = self._split_sentences(src_para, self.src_lang)
                tgt_sentences = self._split_sentences(tgt_para, self.tgt_lang)
                
                if not src_sentences or not tgt_sentences:
                    continue
                
                # Align sentences using LaBSE
                alignment = self._align_sentences(src_sentences, tgt_sentences, similarity_threshold)
                
                # Extract terms if terminology is provided
                terms = None
                if terminology:
                    terms = self._extract_terms(src_para, tgt_para, terminology)
                
                # Store aligned segments
                for align_idx, (src_seg, tgt_seg, score) in enumerate(alignment):
                    df_data.append({
                        'document': doc_idx,  # Document/sample index
                        'paragraph': para_idx,  # Paragraph index within document
                        'sentence': align_idx,
                        'alignment': 'labse',
                        'src_segment': src_seg,
                        'tgt_segment': tgt_seg,
                        'score': score,
                        'terms': terms
                    })
            
            doc_time = time.time() - doc_start
            _log_with_time(f"    Document {doc_idx+1} processed in {doc_time:.2f}s")
        
        total_time = time.time() - start_time
        _log_with_time(f"  Processed {len(documents)} document(s) in {total_time:.2f}s")
        
        # Create DataFrame
        self.df = pd.DataFrame(df_data)
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
    
    def _split_sentences(self, text: str, lang: str) -> List[str]:
        """Split text into sentences."""
        if not text.strip():
            return []
        
        # Use pysbd for sentence splitting
        try:
            import pysbd
            seg = pysbd.Segmenter(language=lang, clean=False)
            sentences = seg.segment(text)
            return [s.strip() for s in sentences if s.strip()]
        except:
            # Fallback: simple splitting
            sentences = re.split(r'[.!?]\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _align_sentences(self, src_sentences: List[str], tgt_sentences: List[str], similarity_threshold: float = 0.4) -> List[Tuple[str, str, float]]:
        """Align sentences using LaBSE embeddings."""
        if not src_sentences or not tgt_sentences:
            return []
        
        # Use PolyFuzz to align sentences
        try:
            matches = self.model.match(src_sentences, tgt_sentences)
            
            # Extract aligned pairs with scores
            alignment = []
            used_tgt_indices = set()
            
            for src_idx, src_sent in enumerate(src_sentences):
                best_tgt_idx = None
                best_score = 0.0
                
                for tgt_idx, tgt_sent in enumerate(tgt_sentences):
                    if tgt_idx in used_tgt_indices:
                        continue
                    
                    # Get similarity score (simplified - PolyFuzz returns DataFrame)
                    # For now, use a simple approach
                    try:
                        score = float(matches['Similarity'].iloc[src_idx * len(tgt_sentences) + tgt_idx]) if hasattr(matches, 'iloc') else 0.0
                    except:
                        score = 0.0
                    
                    if score > best_score and score >= similarity_threshold:
                        best_score = score
                        best_tgt_idx = tgt_idx
                
                if best_tgt_idx is not None:
                    alignment.append((src_sent, tgt_sentences[best_tgt_idx], best_score))
                    used_tgt_indices.add(best_tgt_idx)
                else:
                    # Unaligned source sentence
                    alignment.append((src_sent, "", 0.0))
            
            # Add unaligned target sentences
            for tgt_idx, tgt_sent in enumerate(tgt_sentences):
                if tgt_idx not in used_tgt_indices:
                    alignment.append(("", tgt_sent, 0.0))
            
            return alignment
            
        except Exception as e:
            _log_with_time(f"      ⚠ Sentence alignment failed: {e}")
            # Fallback: simple 1-to-1 alignment
            min_len = min(len(src_sentences), len(tgt_sentences))
            return [(src_sentences[i], tgt_sentences[i] if i < len(tgt_sentences) else "", 1.0) for i in range(min_len)]
    
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
