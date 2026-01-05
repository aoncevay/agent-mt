"""
Document preprocessor for splitting and aligning documents using LaBSE embeddings.

Adapted from WMT25-Term term-consistency approach to work with our data format.
"""

import pandas as pd
import json
import re
import os
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
        
        # Initialize LaBSE embeddings from local path (no HF connection)
        print(f"  Loading LaBSE from local path: {labse_model_path}")
        try:
            # Set environment variables to prevent HF connections (set at module level too)
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # Load SentenceTransformer from local path with local_files_only=True
            # This prevents any HuggingFace connection attempts
            labse_model = SentenceTransformer(str(labse_model_path), local_files_only=True)
            
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
        
        for doc_idx, (src_text, tgt_text) in enumerate(documents):
            # Split into paragraphs
            src_paragraphs, tgt_paragraphs = self._paragraph_aligner(
                src_text, tgt_text, separator=separator
            )
            
            # Align paragraphs
            if len(src_paragraphs) == len(tgt_paragraphs):
                # Naive alignment (1-to-1)
                alignment = 'naive'
                for sent_idx, (src, tgt) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                    score = self._one_one_aligner(src, tgt)
                    df_data.append([doc_idx, sent_idx, alignment, src, tgt, score])
            else:
                # LaBSE-based alignment (many-to-many)
                alignment = 'labse'
                for sent_idx, (src, tgt) in enumerate(zip(src_paragraphs, tgt_paragraphs)):
                    score = self._one_one_aligner(src, tgt)
                    if score < similarity_threshold:
                        # Need many-to-many alignment
                        src_left, tgt_left = src_paragraphs[sent_idx:], tgt_paragraphs[sent_idx:]
                        break
                    else:
                        df_data.append([doc_idx, sent_idx, 'naive', src, tgt, score])
                        src_left, tgt_left = None, None
                
                if src_left is not None and tgt_left is not None:
                    aligned_triplets = self._many_to_many_aligner(src_left, tgt_left)
                    for s, t, score in aligned_triplets:
                        df_data.append([doc_idx, -1, alignment, s, t, score])
        
        # Create DataFrame
        self.df = pd.DataFrame(
            df_data,
            columns=['paragraph', 'sentence', 'alignment', self.src_lang, self.tgt_lang, 'score']
        )
        
        # Add terminology if provided
        if terminology:
            self.df['terms'] = self._assign_terms_to_segments(terminology)
        else:
            self.df['terms'] = [{}] * len(self.df)
        
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

