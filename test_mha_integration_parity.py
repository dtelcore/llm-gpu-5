"""
Phase 3 Integration Validation: GPU-Resident MultiHeadAttention vs. Pre-Rewrite Oracle.

capture_mha_oracle.py captured the exact outputs, forward caches, and parameter
gradients of the ORIGINAL (CPU-roundtrip-heavy) MultiHeadAttention.forward()/
backward() implementation, using a fixed seed/shape scenario, before this
rewrite landed.

This script reconstructs the identical scenario (same seed -> identical initial
weights and identical input/upstream-gradient data) against the REWRITTEN
GPU-resident implementation and asserts numerical equivalence against the
saved oracle for:
    - gpu_output (forward)
    - cache_q, cache_k, cache_v, cache_attn_weights, cache_context (forward caches
      consumed by backward())
    - gpu_dIn (backward)
    - parameter gradients: d(c_attn_w), d(c_attn_b), d(c_proj_w), d(c_proj_b)

A pass here proves the GPU-resident rewrite did not change any value that
MultiHeadAttention.backward() depends on, and that backward() (left untouched)
still produces correct gradients fed by the new forward's caches.

Run with the project's CUDA-enabled virtualenv:
    .\\venv\\Scripts\\python.exe test_mha_integration_parity.py
"""

import numpy as np
import pycuda.driver as cuda

import env_config  # noqa: F401  (bootstraps MSVC/CUDA environment)
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

from model.gpt import GPTConfig, MultiHeadAttention

SEED = 4242
ORACLE_PATH = "test_mha_integration_oracle.npz"
TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def _check(name, ref, actual, all_passed_flag):
    try:
        np.testing.assert_allclose(actual, ref, **TOLERANCE)
        max_abs_diff = float(np.max(np.abs(actual - ref)))
        print(f"[PASS] {name}: max_abs_diff={max_abs_diff:.3e}")
        return all_passed_flag
    except AssertionError as exc:
        print(f"[FAIL] {name}: {exc}")
        return False


def main():
    oracle = np.load(ORACLE_PATH)
    B, T, C, NH = (int(oracle[k]) for k in ("B", "T", "C", "NH"))

    np.random.seed(SEED)
    config = GPTConfig(vocab_size=50, embedding_dim=C, num_heads=NH, max_len=T)
    attn = MultiHeadAttention(config)

    rng = np.random.RandomState(SEED + 1)
    host_x = rng.normal(0.0, 1.0, size=(B * T, C)).astype(np.float32)
    gpu_x = cuda.mem_alloc(host_x.nbytes)
    cuda.memcpy_htod(gpu_x, host_x)

    print(f"--- Phase 3 integration parity (B={B}, T={T}, C={C}, NH={NH}) ---")

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

    all_passed = True
    all_passed = _check("forward output", oracle["out"], host_out, all_passed)
    all_passed = _check("cache_q", oracle["cache_q"], attn.cache_q, all_passed)
    all_passed = _check("cache_k", oracle["cache_k"], attn.cache_k, all_passed)
    all_passed = _check("cache_v", oracle["cache_v"], attn.cache_v, all_passed)
    all_passed = _check("cache_attn_weights", oracle["cache_attn_weights"], attn.cache_attn_weights, all_passed)
    all_passed = _check("cache_context", oracle["cache_context"], attn.cache_context, all_passed)
    all_passed = _check("backward dIn", oracle["din"], host_din, all_passed)
    all_passed = _check("grad c_attn_w", oracle["d_attn_w"], host_d_attn_w, all_passed)
    all_passed = _check("grad c_attn_b", oracle["d_attn_b"], host_d_attn_b, all_passed)
    all_passed = _check("grad c_proj_w", oracle["d_proj_w"], host_d_proj_w, all_passed)
    all_passed = _check("grad c_proj_b", oracle["d_proj_b"], host_d_proj_b, all_passed)

    print()
    if all_passed:
        print("ALL INTEGRATION PARITY CHECKS PASSED: GPU-resident MultiHeadAttention.forward() "
              "is numerically equivalent to the pre-rewrite CPU-roundtrip implementation, and "
              "backward() gradients are unaffected.")
    else:
        print("INTEGRATION PARITY FAILED: see [FAIL] lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
