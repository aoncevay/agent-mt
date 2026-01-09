"""
Centralized configuration for metrics evaluation.

This module provides a clean interface for accessing model paths and tool locations.
All HuggingFace models and tools must be loaded locally (no internet connections).

Configuration is loaded from:
1. Environment variables (highest priority)
2. .env file in metrics/ directory
3. Default fallback paths (for common setups)

Usage:
    from metrics.config import get_config
    
    config = get_config()
    labse_path = config.get_labse_path()
    metricx_path = config.get_metricx_path()
"""

import os
from pathlib import Path
from typing import Optional, List
from functools import lru_cache

# Try to load python-dotenv if available
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False


class MetricsConfig:
    """
    Centralized configuration for metrics evaluation paths.
    
    Supports environment variable overrides and .env file configuration.
    """
    
    def __init__(self):
        # Load .env file if available
        self._load_dotenv()
        
        # Cache for resolved paths
        self._path_cache = {}
    
    def _load_dotenv(self):
        """Load .env file from metrics directory."""
        if DOTENV_AVAILABLE:
            # Try multiple locations for .env
            env_paths = [
                Path(__file__).parent / ".env",  # metrics/.env
                Path(__file__).parent.parent / ".env",  # project root/.env
            ]
            for env_path in env_paths:
                if env_path.exists():
                    load_dotenv(env_path)
                    break
    
    def _expand_path(self, path: str) -> Path:
        """Expand ~ and environment variables in path."""
        return Path(os.path.expandvars(os.path.expanduser(path)))
    
    def _find_existing_path(self, paths: List[Path]) -> Optional[Path]:
        """Return the first existing path from a list of candidates."""
        for path in paths:
            expanded = self._expand_path(str(path)) if isinstance(path, Path) else self._expand_path(path)
            if expanded.exists():
                return expanded
        return None
    
    # =========================================================================
    # Base Directories
    # =========================================================================
    
    def get_hf_models_dir(self) -> Optional[Path]:
        """
        Get the base directory for HuggingFace models.
        
        Environment variable: HF_MODELS_DIR
        Default search paths:
        - ~/HF_models
        - ~/user-default-efs/HF_models
        - /mnt/custom-file-systems/efs/*/HF_models
        """
        env_value = os.environ.get("HF_MODELS_DIR")
        if env_value:
            return self._expand_path(env_value)
        
        candidates = [
            Path.home() / "HF_models",
            Path.home() / "user-default-efs" / "HF_models",
            Path("/mnt/custom-file-systems/efs/fs-0ab0971a17be333d6_fsap-0266e37db01d3e76f/HF_models"),
            Path("/mnt/custom-file-systems/efs") / "HF_models",
        ]
        return self._find_existing_path(candidates)
    
    def get_tools_dir(self) -> Optional[Path]:
        """
        Get the base directory for external tools (vecalign, etc.).
        
        Environment variable: TOOLS_DIR
        Default: {project_root}/other_repos
        """
        env_value = os.environ.get("TOOLS_DIR")
        if env_value:
            return self._expand_path(env_value)
        
        # Default to other_repos in project root
        default = Path(__file__).parent.parent / "other_repos"
        if default.exists():
            return default
        return None
    
    def get_stanza_resources_dir(self) -> Optional[Path]:
        """
        Get the Stanza resources directory.
        
        Environment variable: STANZA_RESOURCES_DIR
        Default search paths:
        - metrics/models/stanza/
        - ~/stanza_resources
        - ~/user-default-efs/stanza_resources
        """
        env_value = os.environ.get("STANZA_RESOURCES_DIR")
        if env_value:
            return self._expand_path(env_value)
        
        candidates = [
            Path(__file__).parent / "models" / "stanza",
            Path.home() / "stanza_resources",
            Path.home() / "user-default-efs" / "stanza_resources",
            Path("/mnt/custom-file-systems/efs") / "stanza_resources",
        ]
        return self._find_existing_path(candidates)
    
    # =========================================================================
    # Individual Model Paths
    # =========================================================================
    
    def get_labse_path(self) -> Optional[Path]:
        """
        Get the LaBSE model path.
        
        Environment variable: LABSE_MODEL_PATH
        Default: {HF_MODELS_DIR}/LaBSE
        """
        env_value = os.environ.get("LABSE_MODEL_PATH")
        if env_value:
            return self._expand_path(env_value)
        
        # Check explicit paths first
        candidates = [
            Path(__file__).parent / "models" / "LaBSE",
        ]
        
        # Then check HF_MODELS_DIR subdirectory
        hf_dir = self.get_hf_models_dir()
        if hf_dir:
            candidates.append(hf_dir / "LaBSE")
        
        # Additional fallback paths
        candidates.extend([
            Path.home() / "HF_models" / "LaBSE",
            Path.home() / "user-default-efs" / "HF_models" / "LaBSE",
            Path.home() / "Documents" / "Code" / "HF_models" / "LaBSE",
        ])
        
        return self._find_existing_path(candidates)
    
    def get_metricx_path(self) -> Optional[Path]:
        """
        Get the MetricX model path.
        
        Environment variable: METRICX_MODEL_PATH
        Default: {HF_MODELS_DIR}/metricx-24-hybrid-large-v2p6-bfloat16
        """
        env_value = os.environ.get("METRICX_MODEL_PATH")
        if env_value:
            return self._expand_path(env_value)
        
        model_names = [
            "metricx-24-hybrid-large-v2p6-bfloat16",
            "metricx-24-hybrid-large-v2p6",
        ]
        
        candidates = []
        
        # Check HF_MODELS_DIR
        hf_dir = self.get_hf_models_dir()
        if hf_dir:
            for name in model_names:
                candidates.append(hf_dir / name)
        
        # Additional fallback paths
        for name in model_names:
            candidates.extend([
                Path.home() / "HF_models" / name,
                Path.home() / "user-default-efs" / "HF_models" / name,
            ])
        
        return self._find_existing_path(candidates)
    
    def get_mt5_tokenizer_path(self) -> Optional[Path]:
        """
        Get the MT5 tokenizer path (used by MetricX).
        
        Environment variable: MT5_TOKENIZER_PATH
        Default: {HF_MODELS_DIR}/mt5-base or mt5-large
        """
        env_value = os.environ.get("MT5_TOKENIZER_PATH")
        if env_value:
            return self._expand_path(env_value)
        
        candidates = []
        hf_dir = self.get_hf_models_dir()
        
        for name in ["mt5-base", "mt5-large"]:
            if hf_dir:
                candidates.append(hf_dir / name)
            candidates.extend([
                Path.home() / "HF_models" / name,
                Path.home() / "user-default-efs" / "HF_models" / name,
            ])
        
        return self._find_existing_path(candidates)
    
    def get_awesome_align_path(self) -> Optional[Path]:
        """
        Get the Awesome-align model path.
        
        Environment variable: AWESOME_ALIGN_MODEL_PATH
        Default: {HF_MODELS_DIR}/awesome-align-with-co
        """
        env_value = os.environ.get("AWESOME_ALIGN_MODEL_PATH")
        if env_value:
            return self._expand_path(env_value)
        
        candidates = []
        hf_dir = self.get_hf_models_dir()
        
        if hf_dir:
            candidates.append(hf_dir / "awesome-align-with-co")
        
        candidates.extend([
            Path.home() / "HF_models" / "awesome-align-with-co",
            Path.home() / "user-default-efs" / "HF_models" / "awesome-align-with-co",
        ])
        
        return self._find_existing_path(candidates)
    
    def get_comet_path(self) -> Optional[Path]:
        """
        Get the COMET model path.
        
        Environment variable: COMET_MODEL_PATH
        Default: {HF_MODELS_DIR}/wmt22-comet-da
        """
        env_value = os.environ.get("COMET_MODEL_PATH")
        if env_value:
            return self._expand_path(env_value)
        
        candidates = []
        hf_dir = self.get_hf_models_dir()
        
        if hf_dir:
            candidates.append(hf_dir / "wmt22-comet-da")
        
        candidates.extend([
            Path.home() / "HF_models" / "wmt22-comet-da",
            Path.home() / "user-default-efs" / "HF_models" / "wmt22-comet-da",
        ])
        
        return self._find_existing_path(candidates)
    
    # =========================================================================
    # Tool Paths
    # =========================================================================
    
    def get_vecalign_path(self) -> Optional[Path]:
        """
        Get the VecAlign repository path.
        
        Environment variable: VECALIGN_PATH
        Default: {TOOLS_DIR}/vecalign or {project_root}/other_repos/vecalign
        """
        env_value = os.environ.get("VECALIGN_PATH")
        if env_value:
            return self._expand_path(env_value)
        
        candidates = [
            Path(__file__).parent.parent / "other_repos" / "vecalign",
        ]
        
        tools_dir = self.get_tools_dir()
        if tools_dir:
            candidates.insert(0, tools_dir / "vecalign")
        
        return self._find_existing_path(candidates)
    
    def get_segale_path(self) -> Optional[Path]:
        """
        Get the SEGALE repository path.
        
        Environment variable: SEGALE_PATH
        Default: {TOOLS_DIR}/SEGALE or {project_root}/other_repos/SEGALE
        """
        env_value = os.environ.get("SEGALE_PATH")
        if env_value:
            return self._expand_path(env_value)
        
        candidates = [
            Path(__file__).parent.parent / "other_repos" / "SEGALE",
        ]
        
        tools_dir = self.get_tools_dir()
        if tools_dir:
            candidates.insert(0, tools_dir / "SEGALE")
        
        return self._find_existing_path(candidates)
    
    # =========================================================================
    # Environment Setup
    # =========================================================================
    
    def setup_offline_environment(self):
        """
        Set environment variables for offline model loading.
        
        This ensures no internet connections are attempted when loading models.
        Call this early in your script before importing transformers/torch/etc.
        """
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # Set Stanza resources dir if found
        stanza_dir = self.get_stanza_resources_dir()
        if stanza_dir:
            os.environ["STANZA_RESOURCES_DIR"] = str(stanza_dir)
    
    def print_config(self):
        """Print current configuration for debugging."""
        print("=" * 60)
        print("Metrics Configuration")
        print("=" * 60)
        
        print("\n📁 Base Directories:")
        print(f"  HF_MODELS_DIR: {self.get_hf_models_dir() or '(not found)'}")
        print(f"  TOOLS_DIR: {self.get_tools_dir() or '(not found)'}")
        print(f"  STANZA_RESOURCES_DIR: {self.get_stanza_resources_dir() or '(not found)'}")
        
        print("\n🤖 Model Paths:")
        print(f"  LaBSE: {self.get_labse_path() or '(not found)'}")
        print(f"  MetricX: {self.get_metricx_path() or '(not found)'}")
        print(f"  MT5 Tokenizer: {self.get_mt5_tokenizer_path() or '(not found)'}")
        print(f"  Awesome-align: {self.get_awesome_align_path() or '(not found)'}")
        print(f"  COMET: {self.get_comet_path() or '(not found)'}")
        
        print("\n🔧 Tool Paths:")
        print(f"  VecAlign: {self.get_vecalign_path() or '(not found)'}")
        print(f"  SEGALE: {self.get_segale_path() or '(not found)'}")
        
        print("=" * 60)


@lru_cache(maxsize=1)
def get_config() -> MetricsConfig:
    """
    Get the singleton configuration instance.
    
    This is cached to avoid repeated .env loading and path resolution.
    """
    return MetricsConfig()


# Convenience functions for common operations
def get_labse_path() -> Optional[Path]:
    """Get LaBSE model path."""
    return get_config().get_labse_path()


def get_metricx_path() -> Optional[Path]:
    """Get MetricX model path."""
    return get_config().get_metricx_path()


def get_vecalign_path() -> Optional[Path]:
    """Get VecAlign repository path."""
    return get_config().get_vecalign_path()


def setup_offline_environment():
    """Set up environment for offline model loading."""
    get_config().setup_offline_environment()


# CLI for testing configuration
if __name__ == "__main__":
    config = get_config()
    config.print_config()
