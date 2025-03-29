"""Configuration handling for LlamaSum."""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any

import torch


@dataclass
class SummarizerConfig:
    """Configuration for the summarization models and processes."""
    
    # Model settings
    model_name: str = "facebook/bart-large-cnn"
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Generation settings
    max_length: int = 150
    min_length: int = 40
    length_penalty: float = 2.0
    num_beams: int = 4
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 50
    
    # Extractive settings
    extractive_ratio: float = 0.3
    use_mmr: bool = True
    diversity_penalty: float = 0.5
    
    # Preprocessing settings
    sentence_splitter: str = "nltk"  # nltk, spacy, or transformers
    min_sentence_length: int = 5
    max_sentence_length: int = 200
    clean_text: bool = True
    remove_stopwords: bool = False
    
    # Advanced settings
    use_coreference_resolution: bool = False
    use_entity_recognition: bool = True
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        """Post initialization processing."""
        # Ensure device is properly set
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU.")
            self.device = "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU.")
            self.device = "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, path: Union[str, Path]):
        """Save configuration to a JSON file."""
        path = Path(path) if isinstance(path, str) else path
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SummarizerConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "SummarizerConfig":
        """Load configuration from a JSON file."""
        path = Path(path) if isinstance(path, str) else path
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


@dataclass
class HierarchicalConfig(SummarizerConfig):
    """Configuration for hierarchical summarization."""
    
    # Hierarchical settings
    levels: int = 3
    level_ratios: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.3])
    segment_size: int = 1000
    segment_stride: int = 500
    segment_strategy: str = "sentence"  # sentence, token, paragraph
    method: str = "recursive"  # recursive, bottom-up, top-down
    

@dataclass
class MultiDocConfig(SummarizerConfig):
    """Configuration for multi-document summarization."""
    
    # Multi-document settings
    clustering_method: str = "kmeans"  # kmeans, agglomerative, spectral
    num_clusters: int = 5
    redundancy_threshold: float = 0.8
    use_temporal_ordering: bool = True
    use_citation_info: bool = False
    cross_document_coreference: bool = False


def load_default_config() -> SummarizerConfig:
    """Load default configuration, potentially from environment variables."""
    config = SummarizerConfig()
    
    # Override from environment variables
    if "LLAMASUM_MODEL" in os.environ:
        config.model_name = os.environ["LLAMASUM_MODEL"]
    
    if "LLAMASUM_DEVICE" in os.environ:
        config.device = os.environ["LLAMASUM_DEVICE"]
    
    return config


def load_config_from_env() -> SummarizerConfig:
    """Load configuration from environment variables."""
    config = SummarizerConfig()
    
    for key, value in os.environ.items():
        if key.startswith("LLAMASUM_"):
            config_key = key[9:].lower()  # Remove LLAMASUM_ prefix
            if hasattr(config, config_key):
                attr_type = type(getattr(config, config_key))
                
                # Convert to appropriate type
                if attr_type == bool:
                    setattr(config, config_key, value.lower() in ("true", "yes", "1"))
                elif attr_type == int:
                    setattr(config, config_key, int(value))
                elif attr_type == float:
                    setattr(config, config_key, float(value))
                elif attr_type == list:
                    setattr(config, config_key, value.split(","))
                else:
                    setattr(config, config_key, value)
    
    return config 