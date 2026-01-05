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


def _log_with_time(message: str):
    """Log message with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


class DocPreprocessor:
    """
    Document preprocessor that splits and aligns documents using LaBSE embeddings.
    
    Adapted to work with our data format (accepts documents directly, not from files).
    """

    def __init__(self, src_lang: str, tgt_lang: str, labse_model_path: Optional[Path] = None):
        """
        Initialize the document preprocessor.
        
        Args:
            src_lang: Source language code (e.g., 'en', 'zht')
            tgt_lang: Target language code (e.g., 'zht', 'es')
            labse_model_path: Optional path to local LaBSE model (default: ~/user-default-efs/HF_models/LaBSE)
        """
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Find LaBSE model path
        if labse_model_path is None:
            # Try multiple possible paths for EFS mount
            possible_paths = [
                # EFS mount path (SageMaker)
                Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/LaBSE"),
                # Home directory path
                Path.home() / "user-default-efs" / "HF_models" / "LaBSE",
                # Alternative EFS path pattern
                Path("/mnt/custom-file-systems/efs") / "HF_models" / "LaBSE",
            ]
            
            # Find the first path that exists
            labse_model_path = None
            for path in possible_paths:
                if path.exists():
                    labse_model_path = path
                    break
            
            # If none found, use the first one as default (will raise error below)
            if labse_model_path is None:
                labse_model_path = possible_paths[0]
        
        # Convert to absolute path and check if local model exists
        labse_model_path = Path(labse_model_path).resolve()
        
        if not labse_model_path.exists():
            raise FileNotFoundError(
                f"LaBSE model not found at {labse_model_path}\n"
                f"Please ensure the model is downloaded and available locally.\n"
                f"Tried paths:\n"
                f"  - /mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models/LaBSE\n"
                f"  - ~/user-default-efs/HF_models/LaBSE\n"
                f"  - /mnt/custom-file-systems/efs/HF_models/LaBSE"
            )
        
        # Debug: Check files in directory
        _log_with_time(f"  Loading LaBSE from local path: {labse_model_path}")
        _log_with_time(f"  Debug: Checking directory contents...")
        
        try:
            files_in_dir = list(labse_model_path.iterdir())
            _log_with_time(f"  Found {len(files_in_dir)} items in directory")
            
            # Check for required files
            model_safetensors = labse_model_path / "model.safetensors"
            model_bin = labse_model_path / "pytorch_model.bin"
            config_json = labse_model_path / "config.json"
            
            _log_with_time(f"  Checking model.safetensors: exists={model_safetensors.exists()}, readable={os.access(model_safetensors, os.R_OK) if model_safetensors.exists() else False}")
            _log_with_time(f"  Checking pytorch_model.bin: exists={model_bin.exists()}, readable={os.access(model_bin, os.R_OK) if model_bin.exists() else False}")
            _log_with_time(f"  Checking config.json: exists={config_json.exists()}, readable={os.access(config_json, os.R_OK) if config_json.exists() else False}")
            
            # List all files for debugging
            _log_with_time(f"  Directory contents:")
            for item in sorted(files_in_dir):
                if item.is_file():
                    size = item.stat().st_size / (1024*1024)  # Size in MB
                    readable = os.access(item, os.R_OK)
                    _log_with_time(f"    - {item.name} ({size:.1f} MB, readable={readable})")
                elif item.is_dir():
                    _log_with_time(f"    - {item.name}/ (directory)")
            
            # Check if files are actually readable
            if model_safetensors.exists() and not os.access(model_safetensors, os.R_OK):
                _log_with_time(f"  ⚠ WARNING: model.safetensors exists but is not readable!")
                _log_with_time(f"  Trying to fix permissions...")
                try:
                    os.chmod(model_safetensors, 0o644)
                    _log_with_time(f"  ✓ Changed permissions on model.safetensors")
                except Exception as e:
                    _log_with_time(f"  ✗ Could not change permissions: {e}")
            
            if model_bin.exists() and not os.access(model_bin, os.R_OK):
                _log_with_time(f"  ⚠ WARNING: pytorch_model.bin exists but is not readable!")
                _log_with_time(f"  Trying to fix permissions...")
                try:
                    os.chmod(model_bin, 0o644)
                    _log_with_time(f"  ✓ Changed permissions on pytorch_model.bin")
                except Exception as e:
                    _log_with_time(f"  ✗ Could not change permissions: {e}")
            
        except Exception as e:
            _log_with_time(f"  ⚠ Warning: Could not debug directory: {e}")
        
        # Initialize LaBSE embeddings from local path (no HF connection)
        try:
            # Set environment variables to prevent HF connections (set at module level too)
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # Load SentenceTransformer from local path with local_files_only=True
            # This prevents any HuggingFace connection attempts
            _log_with_time("  Initializing SentenceTransformer...")
            _log_with_time(f"  Using path: {labse_model_path}")
            _log_with_time(f"  Path exists: {labse_model_path.exists()}")
            _log_with_time(f"  Path is directory: {labse_model_path.is_dir()}")
            
            # Check modules.json to understand the expected structure
            modules_json = labse_model_path / "modules.json"
            if modules_json.exists():
                try:
                    with open(modules_json, 'r') as f:
                        modules_data = json.load(f)
                    _log_with_time(f"  modules.json structure: {json.dumps(modules_data, indent=2)}")
                except Exception as e:
                    _log_with_time(f"  Could not read modules.json: {e}")
            
            # Check if module 0 (Transformer) expects files in a subdirectory
            # Sometimes SentenceTransformer looks for model files in "0/" subdirectory
            module_0_dir = labse_model_path / "0"
            if module_0_dir.exists():
                module_0_files = list(module_0_dir.glob("*"))
                _log_with_time(f"  Module 0 directory exists with {len(module_0_files)} files")
                # Check if model files are in module 0 subdirectory
                module_0_safetensors = module_0_dir / "model.safetensors"
                module_0_bin = module_0_dir / "pytorch_model.bin"
                if module_0_safetensors.exists() or module_0_bin.exists():
                    _log_with_time(f"  ⚠ Model files found in '0/' subdirectory!")
                    _log_with_time(f"     This might be the issue - SentenceTransformer may expect files in root")
            
            # Workaround: Since AutoModel can load the model, try SentenceTransformer with local_files_only=False
            # The offline env vars should still prevent downloads, but this allows more flexible path resolution
            _log_with_time("  Attempting to load SentenceTransformer...")
            _log_with_time("  Note: AutoModel successfully loads this model, so files are accessible")
            
            try:
                # Try without local_files_only first (since we know files are accessible via AutoModel)
                # The TRANSFORMERS_OFFLINE and HF_HUB_OFFLINE env vars should prevent downloads
                labse_model = SentenceTransformer(str(labse_model_path), local_files_only=False)
                _log_with_time("  ✓ SentenceTransformer loaded successfully (without local_files_only)")
            except Exception as e1:
                _log_with_time(f"  First attempt failed: {type(e1).__name__}: {e1}")
                
                # Try with local_files_only=True as fallback
                _log_with_time("  Trying with local_files_only=True...")
                try:
                    labse_model = SentenceTransformer(str(labse_model_path), local_files_only=True)
                    _log_with_time("  ✓ SentenceTransformer loaded successfully (with local_files_only=True)")
                except Exception as e2:
                    _log_with_time(f"  Second attempt also failed: {type(e2).__name__}: {e2}")
                    
                    # Workaround: Load with AutoModel ONCE and construct SentenceTransformer manually
                    _log_with_time("  Using workaround: Loading with AutoModel and constructing SentenceTransformer...")
                    try:
                        from transformers import AutoModel, AutoTokenizer
                        from sentence_transformers.models import Transformer, Pooling, Dense, Normalize
                        # json is already imported at module level
                        
                        # Load the base transformer model ONCE (which we know works) - reuse everywhere
                        _log_with_time("    Loading AutoModel and AutoTokenizer (once, will be reused)...")
                        transformer_model = AutoModel.from_pretrained(str(labse_model_path), local_files_only=True)
                        tokenizer = AutoTokenizer.from_pretrained(str(labse_model_path), local_files_only=True)
                        _log_with_time("    ✓ AutoModel and tokenizer loaded")
                        
                        # Construct SentenceTransformer from modules
                        _log_with_time("    Constructing SentenceTransformer from modules...")
                        modules = []
                        
                        # Module 0: Transformer - use the loaded model directly (no reload)
                        modules.append(Transformer(transformer_model, tokenizer))
                        _log_with_time("    ✓ Created Transformer module from loaded model")
                        
                        # Load other modules from their config.json files (not using .load() which looks for model files)
                        # Pooling module
                        pooling_dir = labse_model_path / "1_Pooling"
                        if pooling_dir.exists():
                            pooling_config_path = pooling_dir / "config.json"
                            if pooling_config_path.exists():
                                with open(pooling_config_path, 'r', encoding='utf-8') as f:
                                    pooling_config = json.load(f)
                                # Create Pooling from config (no model files needed for pooling)
                                pooling = Pooling(**pooling_config)
                                modules.append(pooling)
                                _log_with_time("    ✓ Created Pooling module from config")
                            else:
                                _log_with_time("    ⚠ Pooling directory exists but no config.json, using defaults")
                                modules.append(Pooling())  # Use defaults
                        
                        # Dense module
                        dense_dir = labse_model_path / "2_Dense"
                        if dense_dir.exists():
                            dense_config_path = dense_dir / "config.json"
                            if dense_config_path.exists():
                                with open(dense_config_path, 'r', encoding='utf-8') as f:
                                    dense_config = json.load(f)
                                # Dense module needs in_features - get from transformer model
                                if 'in_features' not in dense_config:
                                    try:
                                        dense_config['in_features'] = transformer_model.config.hidden_size
                                    except:
                                        dense_config['in_features'] = 768  # Default for LaBSE
                                dense = Dense(**dense_config)
                                modules.append(dense)
                                _log_with_time("    ✓ Created Dense module from config")
                            else:
                                _log_with_time("    ⚠ Dense directory exists but no config.json, using defaults")
                                # Use default Dense with transformer's hidden size
                                try:
                                    in_features = transformer_model.config.hidden_size
                                except:
                                    in_features = 768
                                modules.append(Dense(in_features=in_features))
                        
                        # Normalize module (typically doesn't need config)
                        normalize_dir = labse_model_path / "3_Normalize"
                        if normalize_dir.exists():
                            normalize = Normalize()
                            modules.append(normalize)
                            _log_with_time("    ✓ Created Normalize module")
                        
                        # Create SentenceTransformer with modules (using loaded model, no reloads)
                        labse_model = SentenceTransformer(modules=modules)
                        _log_with_time("  ✓ SentenceTransformer created from AutoModel (workaround, single load)")
                    except Exception as e3:
                        _log_with_time(f"  Workaround also failed: {type(e3).__name__}: {e3}")
                        raise RuntimeError(
                            f"Could not load SentenceTransformer from {labse_model_path}\n"
                            f"Even AutoModel workaround failed.\n"
                            f"First error ({type(e1).__name__}): {e1}\n"
                            f"Second error ({type(e2).__name__}): {e2}\n"
                            f"Workaround error ({type(e3).__name__}): {e3}\n"
                            f"Files exist: model.safetensors={model_safetensors.exists()}, "
                            f"pytorch_model.bin={model_bin.exists()}\n"
                            f"Path: {labse_model_path.resolve()}"
                        ) from e3
            
            _log_with_time("  ✓ SentenceTransformer loaded")
            
            # Move model to GPU if available
            import torch
            if torch.cuda.is_available():
                _log_with_time(f"  Moving LaBSE to GPU: {torch.cuda.get_device_name(0)}")
                labse_model = labse_model.to('cuda')
                _log_with_time("  ✓ LaBSE on GPU")
            else:
                _log_with_time("  Using CPU for LaBSE embeddings")
            
            # Create embeddings wrapper for PolyFuzz
            # Try passing the SentenceTransformer instance directly if supported,
            # otherwise use the path (but ensure offline mode is set)
            try:
                # Some versions of flair support passing the model directly
                self.embeddings = SentenceTransformerDocumentEmbeddings(labse_model)
            except (TypeError, ValueError):
                # Fallback: use path string, but ensure offline mode
                # Note: This might still try to connect, but with offline env vars it should fail gracefully
                self.embeddings = SentenceTransformerDocumentEmbeddings(str(labse_model_path))
            
            self.LaBSE = Embeddings(self.embeddings, min_similarity=0, model_id="LaBSE")
            self.model = PolyFuzz([self.LaBSE])
        except Exception as e:
            raise RuntimeError(
                f"Could not load LaBSE model from {labse_model_path}: {e}\n"
                f"Please ensure the model is properly downloaded and the path is correct.\n"
                f"Expected path: ~/user-default-efs/HF_models/LaBSE"
            )
        
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

