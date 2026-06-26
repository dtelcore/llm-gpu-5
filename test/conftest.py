"""
Shared pytest fixtures and utilities for all tests.
Handles GPU setup, model initialization, and common test data.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

import pytest
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Add parent directory to path for imports
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import setup_logging, get_logger


# ==============================================================================
# FIXTURES: GPU & CUDA Setup
# ==============================================================================

@pytest.fixture(scope="session")
def cuda_context():
    """Initialize CUDA context for entire test session."""
    # Single GPU (GT730) doesn't need peer access
    # cuda.Context.enable_peer_access() expects Context, not Device
    cuda.Context.synchronize()
    yield
    cuda.Context.synchronize()


@pytest.fixture(scope="session")
def logger():
    """Setup global logging for tests."""
    setup_logging(log_level="INFO", log_dir=str(TEST_DIR / "logs"))
    return get_logger()


@pytest.fixture
def temp_dir():
    """Create temporary directory for test artifacts."""
    tmpdir = tempfile.mkdtemp(prefix="keplgpt_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def temp_checkpoint(temp_dir):
    """Temporary checkpoint file path."""
    return str(Path(temp_dir) / "test_checkpoint.npz")


@pytest.fixture
def temp_config(temp_dir):
    """Temporary config file path."""
    return str(Path(temp_dir) / "test_config.json")


@pytest.fixture
def temp_log_dir(temp_dir):
    """Temporary log directory."""
    log_dir = Path(temp_dir) / "logs"
    log_dir.mkdir(exist_ok=True)
    return str(log_dir)


# ==============================================================================
# FIXTURES: Model Configuration
# ==============================================================================

@pytest.fixture
def tiny_config():
    """Minimal config for fast testing."""
    return {
        "vocab_size": 50,
        "max_len": 8,
        "max_seq_len": 8,  # Alias for backward compatibility
        "embedding_dim": 32,
        "num_heads": 2,
        "num_layers": 1,
        "dropout_prob": 0.1,
        "batch_size": 2,  # Add for tests
    }


@pytest.fixture
def small_config():
    """Small config for reasonable testing."""
    return {
        "vocab_size": 92,
        "max_len": 16,
        "max_seq_len": 16,
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout_prob": 0.1,
        "batch_size": 4,
    }


@pytest.fixture
def medium_config():
    """Medium config."""
    return {
        "vocab_size": 128,
        "max_len": 32,
        "max_seq_len": 32,
        "embedding_dim": 128,
        "num_heads": 8,
        "num_layers": 4,
        "dropout_prob": 0.1,
        "batch_size": 8,
    }


# ==============================================================================
# FIXTURES: Sample Data
# ==============================================================================

@pytest.fixture
def sample_text_corpus():
    """Small corpus for testing."""
    return "the quick brown fox jumps over the lazy dog"


@pytest.fixture
def sample_tokens():
    """Sample token IDs."""
    return np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)


@pytest.fixture
def sample_batch(tiny_config):
    """Create sample batch data."""
    batch_size = 2  # Fixed batch size for testing
    seq_len = tiny_config["max_len"]
    vocab_size = tiny_config["vocab_size"]
    
    input_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    target_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
    
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "batch_size": batch_size,
        "seq_len": seq_len
    }


@pytest.fixture
def sample_weights(tiny_config):
    """Create sample weight matrices."""
    weights = {}
    
    # Embedding weights
    weights["embeddings"] = np.random.randn(
        tiny_config["vocab_size"],
        tiny_config["embedding_dim"]
    ).astype(np.float32) * 0.01
    
    # Linear layer weights
    weights["linear"] = np.random.randn(
        tiny_config["embedding_dim"],
        tiny_config["hidden_dim"]
    ).astype(np.float32) * np.sqrt(2.0 / tiny_config["embedding_dim"])
    
    # Layer norm parameters
    weights["gamma"] = np.ones(tiny_config["embedding_dim"], dtype=np.float32)
    weights["beta"] = np.zeros(tiny_config["embedding_dim"], dtype=np.float32)
    
    return weights


@pytest.fixture
def sample_logits(tiny_config):
    """Sample logits from model."""
    batch_size = 2  # Fixed batch size for testing
    seq_len = tiny_config["max_len"]
    vocab_size = tiny_config["vocab_size"]
    
    return np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)


# ==============================================================================
# FIXTURES: GPU Arrays
# ==============================================================================

@pytest.fixture
def gpu_array_small():
    """Small GPU array for basic operations."""
    arr = np.random.randn(10, 10).astype(np.float32)
    gpu_arr = cuda.mem_alloc(arr.nbytes)
    cuda.memcpy_htod(gpu_arr, arr)
    yield gpu_arr
    gpu_arr.free()


@pytest.fixture
def gpu_array_large():
    """Large GPU array for memory testing."""
    arr = np.random.randn(1000, 1000).astype(np.float32)
    gpu_arr = cuda.mem_alloc(arr.nbytes)
    cuda.memcpy_htod(gpu_arr, arr)
    yield gpu_arr
    gpu_arr.free()


# ==============================================================================
# FIXTURES: Model Instances
# ==============================================================================

@pytest.fixture
def tokenizer(sample_text_corpus):
    """Create tokenizer instance."""
    from tokenizer.tokenizer import CharacterGPTTokenizer
    return CharacterGPTTokenizer(sample_text_corpus)


@pytest.fixture
def tiny_model(tiny_config, cuda_context):
    """Create tiny GPT model for testing."""
    from model.gpt import GPTModel, GPTConfig
    
    config = GPTConfig(**tiny_config)
    model = GPTModel(config)
    yield model
    
    # Cleanup
    try:
        model.free_forward_caches()
    except:
        pass


@pytest.fixture
def small_model(small_config, cuda_context):
    """Create small GPT model for testing."""
    from model.gpt import GPTModel, GPTConfig
    
    config = GPTConfig(**small_config)
    model = GPTModel(config)
    yield model
    
    # Cleanup
    try:
        model.free_forward_caches()
    except:
        pass


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def assert_gpu_memory_freed():
    """Verify no GPU memory leaks after test."""
    cuda.Context.synchronize()
    # Note: Actual memory checking would require detailed CUDA API calls
    # This is a placeholder for memory validation


def save_test_config(config_path, config_data):
    """Save test configuration to JSON."""
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)


def load_test_config(config_path):
    """Load test configuration from JSON."""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_test_checkpoint(checkpoint_path, config, weights_dict):
    """Create test checkpoint file."""
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint_path, **weights_dict)


# ==============================================================================
# PYTEST CONFIGURATION HOOKS
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as end-to-end test"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test module."""
    for item in items:
        if "gpu" in str(item.fspath) or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "end_to_end" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
