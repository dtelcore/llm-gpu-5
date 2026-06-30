# run_config.py
import json
import os
import shutil
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RunConfig:
    # Model architecture
    vocab_size: int = 4096
    max_len: int = 128
    embedding_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    attention_impl: str = "strided"
    
    # Training hyperparameters
    batch_size: int = 1
    grad_accum: int = 16
    learning_rate: float = 0.015
    total_steps: int = 60
    
    # Dataset
    dataset: str = "fineweb"
    corpus_limit: int = 5000

    # Regularization
    label_smoothing: float = 0.1

    # Validation cadence (decoupled from logging interval)
    val_interval: int = 100
    
    # Run identity
    name: str = "gpt_model"
    preset_name: str = "custom"
    
    # Additional paths
    init_checkpoint_path: str = None

    # Profiling
    profile_gpu_timing: bool = False
    profile_memcpy: bool = False
    
    def __post_init__(self):
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError(f"embedding_dim ({self.embedding_dim}) must be divisible by num_heads ({self.num_heads})")

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # atomic save
        tmp_path = filepath + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=4)
        shutil.move(tmp_path, filepath)

    @classmethod
    def load(cls, filepath: str) -> "RunConfig":
        if not os.path.exists(filepath):
            return cls()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Filter data to only valid keys for the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    @classmethod
    def load_presets(cls, presets_path: str = "config/presets_gt730_v2.json") -> list:
        if not os.path.exists(presets_path):
            return []
        with open(presets_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle both old flat format and new nested format
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "presets_by_category" in data:
            # Flatten nested structure back to list for compatibility
            presets = []
            for category_presets in data["presets_by_category"].values():
                presets.extend(category_presets)
            return presets
        return []
