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

# Global cache for loaded models (loaded once, reused everywhere)
_loaded_models_cache = {}


def _log_with_time(message: str):
    """Log message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


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
    if labse_model_path is None:
        possible_paths = [
            Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/LaBSE"),
            Path.home() / "user-default-efs" / "HF_models" / "LaBSE",
            Path("/mnt/custom-file-systems/efs") / "HF_models" / "LaBSE",
            Path.home() / "Documents" / "Code" / "HF_models" / "LaBSE",  # Local path
        ]
        
        for path in possible_paths:
            if path.exists():
                labse_model_path = path
                break
        
        if labse_model_path is None:
            raise FileNotFoundError(
                f"LaBSE model not found. Tried paths:\n" + "\n".join(f"  - {p}" for p in possible_paths)
            )
    
    labse_model_path = Path(labse_model_path).resolve()
    _log_with_time(f"Loading LaBSE model from: {labse_model_path}")
    
    # Check sentence-transformers version
    try:
        import sentence_transformers
        _log_with_time(f"  sentence-transformers version: {sentence_transformers.__version__}")
    except:
        _log_with_time("  ⚠ Could not determine sentence-transformers version")
    
    # Strategy: Use AutoModel to load, then construct SentenceTransformer
    _log_with_time("  Loading with AutoModel (known to work)...")
    try:
        from transformers import AutoModel, AutoTokenizer
        from sentence_transformers.models import Transformer, Pooling, Dense, Normalize
        import torch
        
        # Load transformer model ONCE
        transformer_model = AutoModel.from_pretrained(str(labse_model_path), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(str(labse_model_path), local_files_only=True)
        _log_with_time("  ✓ AutoModel loaded")
        
        # Move to GPU if available and requested
        if use_gpu and torch.cuda.is_available():
            _log_with_time(f"  Moving to GPU: {torch.cuda.get_device_name(0)}")
            transformer_model = transformer_model.to('cuda')
        else:
            _log_with_time("  Using CPU")
        
        # Construct modules
        modules = []
        
        # Module 0: Transformer
        modules.append(Transformer(transformer_model, tokenizer))
        _log_with_time("  ✓ Transformer module created")
        
        # Module 1: Pooling
        pooling_dir = labse_model_path / "1_Pooling"
        if pooling_dir.exists() and (pooling_dir / "config.json").exists():
            with open(pooling_dir / "config.json", 'r', encoding='utf-8') as f:
                pooling_config = json.load(f)
            modules.append(Pooling(**pooling_config))
        else:
            modules.append(Pooling())  # Default pooling
        _log_with_time("  ✓ Pooling module created")
        
        # Module 2: Dense
        dense_dir = labse_model_path / "2_Dense"
        if dense_dir.exists() and (dense_dir / "config.json").exists():
            with open(dense_dir / "config.json", 'r', encoding='utf-8') as f:
                dense_config = json.load(f)
            if 'in_features' not in dense_config:
                dense_config['in_features'] = transformer_model.config.hidden_size
            modules.append(Dense(**dense_config))
        else:
            # Default Dense with transformer's hidden size
            in_features = transformer_model.config.hidden_size
            modules.append(Dense(in_features=in_features))
        _log_with_time("  ✓ Dense module created")
        
        # Module 3: Normalize
        modules.append(Normalize())
        _log_with_time("  ✓ Normalize module created")
        
        # Create SentenceTransformer
        labse_model = SentenceTransformer(modules=modules)
        _log_with_time("  ✓ SentenceTransformer created from AutoModel")
        
        # Cache it
        _loaded_models_cache[cache_key] = labse_model
        _log_with_time("  ✓ Model cached for reuse")
        
        return labse_model
        
    except Exception as e:
        _log_with_time(f"  ✗ AutoModel workaround failed: {e}")
        # Fallback: try SentenceTransformer directly (might work in some cases)
        _log_with_time("  Trying SentenceTransformer directly as fallback...")
        try:
            labse_model = SentenceTransformer(str(labse_model_path), local_files_only=False)
            if use_gpu:
                import torch
                if torch.cuda.is_available():
                    labse_model = labse_model.to('cuda')
            _loaded_models_cache[cache_key] = labse_model
            return labse_model
        except Exception as e2:
            raise RuntimeError(
                f"Could not load LaBSE model from {labse_model_path}\n"
                f"AutoModel approach: {e}\n"
                f"SentenceTransformer fallback: {e2}"
            ) from e2


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
        use_gpu: bool = True
    ):
        """
        Initialize the document preprocessor.
        
        Args:
            src_lang: Source language code (e.g., 'en', 'zht')
            tgt_lang: Target language code (e.g., 'zht', 'es')
            labse_model_path: Optional path to local LaBSE model (default: auto-detect)
            labse_model: Optional pre-loaded SentenceTransformer model (reuses if provided)
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
        
        # Create embeddings wrapper for PolyFuzz using the loaded model
        # Try passing the SentenceTransformer instance directly
        try:
            _log_with_time("  Creating embeddings wrapper for PolyFuzz...")
            # Pass the SentenceTransformer instance directly
            self.embeddings = SentenceTransformerDocumentEmbeddings(self.labse_model)
            _log_with_time("  ✓ Created SentenceTransformerDocumentEmbeddings from model instance")
        except (TypeError, ValueError) as e:
            _log_with_time(f"  ⚠ Could not pass model instance: {e}")
            _log_with_time("  Trying with model path as fallback...")
            # Fallback: try with path (but this might reload, which we want to avoid)
            # Get the path from the model if possible
            try:
                # Try to get path from cache key
                model_path = None
                for key, model in _loaded_models_cache.items():
                    if model is self.labse_model:
                        model_path = key
                        break
                
                if model_path and model_path != "default":
                    self.embeddings = SentenceTransformerDocumentEmbeddings(str(model_path))
                    _log_with_time("  ✓ Created SentenceTransformerDocumentEmbeddings from path")
                else:
                    # Last resort: try to use the model's encode method directly
                    # We'll need to wrap it for PolyFuzz compatibility
                    _log_with_time("  ⚠ Could not determine model path, using direct encoding")
                    raise ValueError("Could not create embeddings wrapper - need model path")
            except Exception as e2:
                _log_with_time(f"  ✗ Fallback also failed: {e2}")
                raise RuntimeError(
                    f"Could not create embeddings wrapper. "
                    f"Model loaded successfully but cannot create PolyFuzz embeddings.\n"
                    f"Original error: {e}\n"
                    f"Fallback error: {e2}\n"
                    f"Consider using the model's encode() method directly."
                ) from e2
        
        self.LaBSE = Embeddings(self.embeddings, min_similarity=0, model_id="LaBSE")
        self.model = PolyFuzz([self.LaBSE])
        _log_with_time("  ✓ PolyFuzz model initialized")
        
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
            if len(src_paragraphs) == len(tgt_paragraphs):
                # Naive alignment (1-to-1)
                _log_with_time(f"      Using naive 1-to-1 alignment ({len(src_paragraphs)} pairs)")
                alignment = 'naive'
                for sent_idx, (src, tgt) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                    score = self._one_one_aligner(src, tgt)
                    df_data.append([doc_idx, sent_idx, alignment, src, tgt, score])
                _log_with_time(f"      ✓ Document {doc_idx+1} aligned ({len(src_paragraphs)} segments) in {time.time() - doc_start:.2f}s")
            else:
                # LaBSE-based alignment (many-to-many)
                _log_with_time(f"      Using LaBSE alignment (src: {len(src_paragraphs)}, tgt: {len(tgt_paragraphs)})")
                alignment = 'labse'
                aligned_count = 0
                for sent_idx, (src, tgt) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                    score = self._one_one_aligner(src, tgt)
                    if score < similarity_threshold:
                        # Need many-to-many alignment
                        _log_with_time(f"      Low similarity at {sent_idx}, switching to many-to-many alignment...")
                        src_left, tgt_left = src_paragraphs[sent_idx:], tgt_paragraphs[sent_idx:]
                        break
                    else:
                        df_data.append([doc_idx, sent_idx, 'naive', src, tgt, score])
                        aligned_count += 1
                        src_left, tgt_left = None, None
                
                if src_left is not None and tgt_left is not None:
                    _log_with_time(f"      Running many-to-many alignment ({len(src_left)} src, {len(tgt_left)} tgt)...")
                    many_to_many_start = time.time()
                    aligned_triplets = self._many_to_many_aligner(src_left, tgt_left)
                    _log_with_time(f"      ✓ Many-to-many alignment completed ({len(aligned_triplets)} pairs) in {time.time() - many_to_many_start:.2f}s")
                    for s, t, score in aligned_triplets:
                        df_data.append([doc_idx, -1, alignment, s, t, score])
                    aligned_count += len(aligned_triplets)
                
                _log_with_time(f"      ✓ Document {doc_idx+1} aligned ({aligned_count} segments) in {time.time() - doc_start:.2f}s")
        
        # Create DataFrame
        _log_with_time(f"  Creating DataFrame from {len(df_data)} aligned segments...")
        self.df = pd.DataFrame(
            df_data,
            columns=['paragraph', 'sentence', 'alignment', self.src_lang, self.tgt_lang, 'score']
        )
        
        # Add terminology if provided
        if terminology:
            _log_with_time("  Assigning terminology to segments...")
            self.df['terms'] = self._assign_terms_to_segments(terminology)
        else:
            self.df['terms'] = [{}] * len(self.df)
        
        total_time = time.time() - start_time
        _log_with_time(f"  ✓ Document processing complete: {len(self.df)} segments in {total_time:.2f}s ({total_time/len(documents):.2f}s per document)")
        
        return self.df

    def _paragraph_aligner(
        self,
        src_text: str,
        tgt_text: str,
        separator: str = '\n\n'
    ) -> Tuple[List[str], List[str]]:
        """
        Split source and target texts into paragraphs.
        
        Args:
            src_text: Source text
            tgt_text: Target text
            separator: Paragraph separator
        
        Returns:
            (src_paragraphs, tgt_paragraphs) - Lists of paragraph strings
        """
        src_paragraphs = [p.strip() for p in src_text.split(separator) if p.strip()]
        tgt_paragraphs = [p.strip() for p in tgt_text.split(separator) if p.strip()]
        
        return src_paragraphs, tgt_paragraphs

    def _many_to_many_aligner(
        self,
        src_paragraphs: List[str],
        tgt_paragraphs: List[str]
    ) -> List[Tuple[str, str, float]]:
        """
        Align multiple source paragraphs to multiple target paragraphs using LaBSE.
        
        Args:
            src_paragraphs: List of source paragraphs
            tgt_paragraphs: List of target paragraphs
        
        Returns:
            List of (src, tgt, score) tuples
        """
        output = self.model.match(src_paragraphs, tgt_paragraphs)
        dfx = self.model.get_matches()
        return dfx.values.tolist()

    def _one_one_aligner(self, src_sent: str, tgt_sent: str) -> float:
        """
        Compute similarity score between two sentences using LaBSE.
        
        Args:
            src_sent: Source sentence
            tgt_sent: Target sentence
        
        Returns:
            Similarity score (0.0 to 1.0)
        """
        output = self.model.match([src_sent], [tgt_sent])
        score = output.matches['LaBSE']['Similarity'][0]
        return score

    def _assign_terms_to_segments(
        self,
        terminology: Dict[str, list]
    ) -> List[Dict[str, list]]:
        """
        Assign terminology to segments based on term occurrence in source text.
        
        Args:
            terminology: Global terminology dictionary {term: [translations]}
        
        Returns:
            List of term dictionaries (one per segment)
        """
        term_assignments = []
        
        for _, row in self.df.iterrows():
            src_segment = row[self.src_lang]
            segment_terms = {}
            
            # Find terms that occur in this segment
            for term, translations in terminology.items():
                # Simple substring matching (case-insensitive)
                if term.lower() in src_segment.lower():
                    segment_terms[term] = translations
            
            term_assignments.append(segment_terms)
        
        return term_assignments

    def _normalize_en_paragraph(self, paragraph: str) -> str:
        """
        Normalize an English paragraph using stanza.
        
        Args:
            paragraph: English paragraph text
        
        Returns:
            Normalized paragraph (lemmatized, lowercase)
        """
        if not self.stanza_en:
            return paragraph.lower()
        
        doc = self.stanza_en(paragraph)
        lemmas = [word.lemma.lower() for sent in doc.sentences for word in sent.words]
        return ' '.join(lemmas)

