"""
Data loader abstraction for different datasets.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    """Base class for dataset loaders."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
    
    @abstractmethod
    def load_samples(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load samples from the dataset."""
        pass
    
    @abstractmethod
    def get_translation_direction(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Get source and target language codes for a sample."""
        pass
    
    @abstractmethod
    def extract_texts(self, sample: Dict[str, Any], source_lang: str, target_lang: str) -> Tuple[str, str, Optional[Dict[str, list]]]:
        """Extract source text, reference text, and optional terminology."""
        pass
    
    @abstractmethod
    def get_dataset_name(self) -> str:
        """Get the dataset name."""
        pass


class WMT25DataLoader(BaseDataLoader):
    """Loader for WMT25 terminology track2 dataset."""
    
    def __init__(self, data_dir: Path, target_languages: Optional[List[str]] = None):
        """
        Initialize WMT25 loader.
        
        Args:
            data_dir: Directory containing WMT25 files
            target_languages: List of target languages to filter (e.g., ["zht"] for en->zht, ["en"] for zht->en)
                             If None, processes both directions. Note: filtering happens in run.py, not here.
        """
        super().__init__(data_dir)
        self.years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        self.target_languages = target_languages  # Kept for API compatibility, but filtering happens in run.py
    
    def load_samples(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load samples from all year files."""
        all_samples = []
        
        for year in self.years:
            file_path = self.data_dir / f"full_data_{year}.jsonl"
            if not file_path.exists():
                continue
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        sample["_year"] = year  # Add year metadata
                        all_samples.append(sample)
                        
                        if max_samples and len(all_samples) >= max_samples:
                            return all_samples
        
        return all_samples
    
    def get_translation_direction(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """Determine direction based on year: odd=en->zht, even=zht->en."""
        year = sample.get("_year", 2015)
        if year % 2 == 1:  # Odd year
            return "en", "zht"
        else:  # Even year
            return "zht", "en"
    
    def extract_texts(self, sample: Dict[str, Any], source_lang: str, target_lang: str) -> Tuple[str, str, Optional[Dict[str, list]]]:
        """Extract texts and terminology from WMT25 sample."""
        # Map language codes to keys in data
        lang_key_map = {
            "en": "en",
            "zht": "zh",  # Files use "zh" but we refer to it as "zht"
            "zh": "zh"
        }
        
        source_key = lang_key_map.get(source_lang, source_lang)
        target_key = lang_key_map.get(target_lang, target_lang)
        
        source_text = sample.get(source_key, "")
        target_text = sample.get(target_key, "")
        terminology = sample.get("proper", {})
        
        return source_text, target_text, terminology
    
    def get_dataset_name(self) -> str:
        return "wmt25"


class DOLFINDataLoader(BaseDataLoader):
    """Loader for DOLFIN dataset."""
    
    def __init__(self, data_dir: Path, lang_pair: str = "en_es"):
        """
        Initialize DOLFIN loader.
        
        Args:
            data_dir: Directory containing DOLFIN files
            lang_pair: Language pair (e.g., "en_es", "en_de", "en_fr", "en_it")
        """
        super().__init__(data_dir)
        self.lang_pair = lang_pair
        self.source_lang, self.target_lang = lang_pair.split("_")
    
    def load_samples(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load samples from DOLFIN file."""
        file_path = self.data_dir / f"dolfin_test_{self.lang_pair}.jsonl"
        
        if not file_path.exists():
            raise FileNotFoundError(f"DOLFIN file not found: {file_path}")
        
        samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
                    if max_samples and len(samples) >= max_samples:
                        break
        
        return samples
    
    def get_translation_direction(self, sample: Dict[str, Any]) -> Tuple[str, str]:
        """DOLFIN always uses the same direction for a given file."""
        return self.source_lang, self.target_lang
    
    def extract_texts(self, sample: Dict[str, Any], source_lang: str, target_lang: str) -> Tuple[str, str, Optional[Dict[str, list]]]:
        """Extract texts from DOLFIN sample (no terminology)."""
        source_text = sample.get(source_lang, "")
        target_text = sample.get(target_lang, "")
        # DOLFIN has no terminology
        terminology = None
        
        return source_text, target_text, terminology
    
    def get_dataset_name(self) -> str:
        return f"dolfin_{self.lang_pair}"


def get_data_loader(dataset_name: str, data_dir: Path, target_languages: Optional[List[str]] = None) -> BaseDataLoader:
    """
    Factory function to get the appropriate data loader.
    
    Args:
        dataset_name: Name of dataset ("wmt25" or "dolfin")
        data_dir: Base data directory
        target_languages: List of target languages to filter (optional)
    
    Returns:
        Appropriate data loader instance
    """
    if dataset_name == "wmt25":
        return WMT25DataLoader(data_dir / "wmt25-terminology-track2", target_languages=target_languages)
    elif dataset_name == "dolfin":
        # For dolfin, we'll handle multiple language pairs in run.py
        # This is a placeholder - actual filtering happens in run.py
        return None  # Will be handled differently
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: 'wmt25', 'dolfin'")


def get_available_dolfin_lang_pairs(data_dir: Path) -> List[str]:
    """Get list of available DOLFIN language pairs."""
    dolfin_dir = data_dir / "dolfin"
    available_pairs = []
    
    for file_path in dolfin_dir.glob("dolfin_test_*.jsonl"):
        # Extract lang pair from filename: dolfin_test_en_es.jsonl -> en_es
        lang_pair = file_path.stem.replace("dolfin_test_", "")
        available_pairs.append(lang_pair)
    
    return sorted(available_pairs)

