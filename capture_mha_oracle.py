"""
One-shot fixture capture: runs the CURRENT (pre-GPU-residency-rewrite)
MultiHeadAttention.forward()/backward() implementation and saves its outputs,
forward caches, and parameter gradients to disk as a parity oracle.

This script must be run BEFORE rewriting MultiHeadAttention.forward() in
model/gpt.py. test_mha_integration_parity.py then reconstructs the identical
scenario (same seeds, same shapes) against the rewritten implementation and
compares against this captured oracle.

Run with the project's CUDA-enabled virtualenv:
    .\\venv\\Scripts\\python.exe capture_mha_oracle.py
"""

import numpy as np
import pycuda.driver as cuda

import env_config  # noqa: F401  (bootstraps MSVC/CUDA environment)
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

from model.gpt import GPTConfig, MultiHeadAttention

SEED = 4242
B, T, C, NH = 2, 5, 8, 2
ORACLE_PATH = "test_mha_integration_oracle.npz"


def main():
    np.random.seed(SEED)
    config = GPTConfig(vocab_size=50, embedding_dim=C, num_heads=NH, max_len=T)
    attn = MultiHeadAttention(config)

    rng = np.random.RandomState(SEED + 1)
    host_x = rng.normal(0.0, 1.0, size=(B * T, C)).astype(np.float32)
    gpu_x = cuda.mem_alloc(host_x.nbytes)
    cuda.memcpy_htod(gpu_x, host_x)

    gpu_out = attn.forward(gpu_x, B, T)
    host_out = np.empty((B * T, C), dtype=np.float32)
    cuda.memcpy_dtoh(host_out, gpu_out)

    host_dout = rng.normal(0.0, 1.0, size=(B * T, C)).astype(np.float32)
    gpu_dout = cuda.mem_alloc(host_dout.nbytes)
    cuda.memcpy_htod(gpu_dout, host_dout)

    gpu_din = attn.backward(gpu_dout, B, T)
    host_din = np.empty((B * T, C), dtype=np.float32)
    cuda.memcpy_dtoh(host_din, gpu_din)

    host_d_attn_w = np.empty(attn.c_attn_w.shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_d_attn_w, attn.c_attn_w.gpu_grads)
    host_d_attn_b = np.empty(attn.c_attn_b.shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_d_attn_b, attn.c_attn_b.gpu_grads)
    host_d_proj_w = np.empty(attn.c_proj_w.shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_d_proj_w, attn.c_proj_w.gpu_grads)
    host_d_proj_b = np.empty(attn.c_proj_b.shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_d_proj_b, attn.c_proj_b.gpu_grads)

    np.savez(
        ORACLE_PATH,
        out=host_out,
        din=host_din,
        cache_q=attn.cache_q,
        cache_k=attn.cache_k,
        cache_v=attn.cache_v,
        cache_attn_weights=attn.cache_attn_weights,
        cache_context=attn.cache_context,
        d_attn_w=host_d_attn_w,
        d_attn_b=host_d_attn_b,
        d_proj_w=host_d_proj_w,
        d_proj_b=host_d_proj_b,
        B=B, T=T, C=C, NH=NH,
    )
    print(f"Oracle captured to {ORACLE_PATH}")
    print(f"out shape={host_out.shape}, cache_q shape={attn.cache_q.shape}, "
          f"cache_attn_weights shape={attn.cache_attn_weights.shape}, "
          f"cache_context shape={attn.cache_context.shape}")


if __name__ == "__main__":
    main()
