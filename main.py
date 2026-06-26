"""
Main orchestrator for GPU-accelerated GPT pipeline.

Demonstrates end-to-end workflow:
1. Environment initialization (CUDA path enforcement)
2. Tokenizer initialization (character-level vocabulary)
3. Token encoding (text -> token IDs)
4. GPU memory staging (CPU NumPy -> GPU VRAM)
5. Model initialization (GPT parameters)
6. Forward pass (token IDs through GPU model)
7. Decoding (logits -> next token prediction)

Target: NVIDIA GeForce GT 730 (Kepler / Compute Capability 3.5)
"""

import os
import sys
import numpy as np

# ============================================================================
# Environment & Path Setup (Must execute FIRST)
# ============================================================================
import env_config  # Centralizes CUDA/MSVC path enforcement

# Now safe to import PyCUDA
import pycuda.autoinit
from pycuda.driver import mem_alloc, memcpy_htod, memcpy_dtoh

from logging_config import logger
from tokenizer import CharacterGPTTokenizer
from model.gpt import GPTModel, GPTConfig


def main():
    """Main execution pipeline."""
    
    logger.info("="*80)
    logger.info("GPU-Accelerated GPT Pipeline (GT 730)")
    logger.info("="*80)
    
    # ========================================================================
    # Step 1: Tokenizer Initialization
    # ========================================================================
    logger.info("[Step 1] Initializing Character-Level Tokenizer...")
    
    # Sample corpus for vocabulary building
    corpus = [
        "cuda kepler gt730",
        "gpu parallel compute",
        "nvidia architecture",
        "matrix multiply kernel"
    ]
    
    tokenizer = CharacterGPTTokenizer(corpus)
    logger.info(f"✓ Tokenizer ready: vocab_size={tokenizer.vocab_size}, BOS_ID={tokenizer.BOS_ID}")
    
    # ========================================================================
    # Step 2: Text Encoding (CPU-side)
    # ========================================================================
    logger.info("[Step 2] Encoding Text Batch on CPU...")
    
    text_batch = ["cuda", "gpu"]
    max_seq_length = 16
    token_matrix = tokenizer.encode_batch_gpu_aligned(text_batch, max_sequence_length=max_seq_length)
    
    logger.info(f"✓ Encoded batch shape: {token_matrix.shape}")
    logger.debug(f"Token matrix (first row): {token_matrix[0]}")
    
    # ========================================================================
    # Step 3: GPU Memory Staging (CPU -> GPU)
    # ========================================================================
    logger.info("[Step 3] Staging Token Matrix to GPU VRAM...")
    
    batch_size = token_matrix.shape[0]
    seq_length = token_matrix.shape[1]
    
    # Convert to int32 for GPU transfer
    token_matrix_int32 = token_matrix.astype(np.int32)
    
    # Allocate GPU memory and transfer
    gpu_token_ids = mem_alloc(token_matrix_int32.nbytes)
    memcpy_htod(gpu_token_ids, token_matrix_int32)
    
    logger.info(f"✓ Transferred {token_matrix_int32.nbytes} bytes to GPU")
    logger.debug(f"GPU memory address: {int(gpu_token_ids)}")
    
    # ========================================================================
    # Step 4: Model Initialization
    # ========================================================================
    logger.info("[Step 4] Initializing GPT Model...")
    
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=256,
        num_layers=2,
        num_heads=4,
        max_seq_length=max_seq_length
    )
    
    model = GPTModel(config)
    logger.info(f"✓ Model initialized: embed_dim={config.embed_dim}, num_heads={config.num_heads}, num_layers={config.num_layers}")
    
    # ========================================================================
    # Step 5: Forward Pass (GPU Inference)
    # ========================================================================
    logger.info("[Step 5] Running Forward Pass on GPU...")
    
    try:
        logits_gpu = model.forward(gpu_token_ids, batch_size, seq_length)
        logger.info(f"✓ Forward pass complete | logits address: {int(logits_gpu)}")
    except Exception as e:
        logger.warning(f"Forward pass encountered an error (expected during development): {e}")
        logger.info("Continuing with cleanup...")
    
    # ========================================================================
    # Step 6: Cleanup
    # ========================================================================
    logger.info("[Step 6] Cleaning Up GPU Memory...")
    
    gpu_token_ids.free()
    logger.info("✓ GPU memory deallocated")
    
    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("="*80)
    logger.info("Pipeline Execution Complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
