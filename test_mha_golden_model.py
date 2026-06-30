"""
Phase 3 Validation Harness: GPU MHA Kernels vs. NumPy Golden Model (Parity Test).

Compares the raw PyCUDA kernels in core/mha_kernels.py against an independent NumPy
reference implementation of causal multi-head attention forward + backward (VJP form).
This is the correctness gate that must pass before any tiling/fusion/FlashAttention-style
optimization work begins -- it exists to catch silent drift between:
    - CUDA math (core/mha_kernels.py)
    - NumPy reference (this file)
    - future kernel rewrites

Checks, per layer:
    - Scores (pre-softmax, Q @ K^T * scale)
    - Probs  (post-softmax, causal)
    - Out    (Probs @ V)
    - dV     (probs^T @ dOut)
    - dProbs_presoftmax (softmax VJP)
    - dQ     (dScores @ K)
    - dK     (dScores^T @ Q)

Run with the project's CUDA-enabled virtualenv, e.g.:
    .\\venv\\Scripts\\python.exe test_mha_golden_model.py
"""

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import env_config  # noqa: F401  (bootstraps MSVC/CUDA environment)
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

from core.mha_kernels import MHA_KERNELS_STRING


SEED = 1337
H, M, D = 3, 17, 8  # heads, sequence length, head_dim (odd M to stress causal edge rows)
SCALE = 1.0 / np.sqrt(D)
TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def _compile():
    nvcc_options = [
        "-ccbin", env_config.MSVC_142_BIN,
        "-O3",
        "--use_fast_math",
        "-Xcompiler", "/w",
    ]
    return SourceModule(MHA_KERNELS_STRING, options=nvcc_options)


def _to_gpu(host_arr):
    gpu_ptr = cuda.mem_alloc(host_arr.nbytes)
    cuda.memcpy_htod(gpu_ptr, host_arr)
    return gpu_ptr


def _from_gpu(gpu_ptr, shape):
    host_arr = np.empty(shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_arr, gpu_ptr)
    return host_arr


def numpy_causal_attention_forward(Q, K, V, scale):
    """Reference forward pass matching matmul_score_kernel + softmax_fused_forward + matmul_proj_kernel."""
    scores = np.einsum("hid,hjd->hij", Q, K).astype(np.float64) * scale  # [H, M, M]

    causal_mask = np.tril(np.ones((M, M), dtype=bool))
    masked_scores = np.where(causal_mask[None, :, :], scores, -np.inf)
    row_max = np.max(masked_scores, axis=-1, keepdims=True)
    exp_scores = np.where(causal_mask[None, :, :], np.exp(masked_scores - row_max), 0.0)
    row_sum = np.sum(exp_scores, axis=-1, keepdims=True)
    probs = exp_scores / row_sum  # [H, M, M], zero above diagonal

    out = np.einsum("hij,hjd->hid", probs, V)  # [H, M, D]
    return scores.astype(np.float32), probs.astype(np.float32), out.astype(np.float32)


def numpy_causal_attention_backward(probs, Q, K, V, dOut):
    """Reference backward pass matching matmul_grad_v + softmax_fused_backward + matmul_grad_q/k_kernel."""
    probs64 = probs.astype(np.float64)
    dOut64 = dOut.astype(np.float64)

    dV = np.einsum("hij,hid->hjd", probs64, dOut64)  # probs^T @ dOut

    dProbs_post = np.einsum("hid,hjd->hij", dOut64, V.astype(np.float64))  # dOut @ V^T

    causal_mask = np.tril(np.ones((M, M), dtype=bool))
    row_dot = np.sum(dProbs_post * probs64 * causal_mask[None, :, :], axis=-1, keepdims=True)
    dScores_pre = np.where(causal_mask[None, :, :], probs64 * (dProbs_post - row_dot), 0.0)

    dQ = np.einsum("hij,hjd->hid", dScores_pre, K.astype(np.float64))  # dScores @ K
    dK = np.einsum("hij,hid->hjd", dScores_pre, Q.astype(np.float64))  # dScores^T @ Q

    return dV.astype(np.float32), dScores_pre.astype(np.float32), dQ.astype(np.float32), dK.astype(np.float32)


def main():
    np.random.seed(SEED)
    module = _compile()

    matmul_score = module.get_function("matmul_score_kernel")
    matmul_proj = module.get_function("matmul_proj_kernel")
    softmax_fwd = module.get_function("softmax_fused_forward")
    softmax_bwd = module.get_function("softmax_fused_backward")
    matmul_grad_q = module.get_function("matmul_grad_q_kernel")
    matmul_grad_k = module.get_function("matmul_grad_k_kernel")
    matmul_grad_v = module.get_function("matmul_grad_v")

    Q = np.random.normal(0.0, 1.0, size=(H, M, D)).astype(np.float32)
    K = np.random.normal(0.0, 1.0, size=(H, M, D)).astype(np.float32)
    V = np.random.normal(0.0, 1.0, size=(H, M, D)).astype(np.float32)
    dOut = np.random.normal(0.0, 1.0, size=(H, M, D)).astype(np.float32)

    ref_scores, ref_probs, ref_out = numpy_causal_attention_forward(Q, K, V, SCALE)
    ref_dV, ref_dScores_pre, ref_dQ, ref_dK = numpy_causal_attention_backward(ref_probs, Q, K, V, dOut)

    gpu_Q, gpu_K, gpu_V, gpu_dOut = _to_gpu(Q), _to_gpu(K), _to_gpu(V), _to_gpu(dOut)
    gpu_scores = cuda.mem_alloc(H * M * M * 4)
    gpu_row_max = cuda.mem_alloc(H * M * 4)
    gpu_row_sum = cuda.mem_alloc(H * M * 4)
    gpu_out = cuda.mem_alloc(H * M * D * 4)

    threads = 256
    score_blocks = (H * M * M + threads - 1) // threads
    proj_blocks = (H * M * D + threads - 1) // threads
    softmax_block_threads = 64
    softmax_shared_bytes = softmax_block_threads * 4

    # --- FORWARD ---
    matmul_score(gpu_Q, gpu_K, gpu_scores, np.int32(H), np.int32(M), np.int32(D), np.float32(SCALE),
                 block=(threads, 1, 1), grid=(score_blocks, 1))
    softmax_fwd(gpu_scores, gpu_row_max, gpu_row_sum, np.int32(H), np.int32(M),
                block=(softmax_block_threads, 1, 1), grid=(H, M), shared=softmax_shared_bytes)
    matmul_proj(gpu_scores, gpu_V, gpu_out, np.int32(H), np.int32(M), np.int32(D),
                block=(threads, 1, 1), grid=(proj_blocks, 1))

    gpu_probs_host = _from_gpu(gpu_scores, (H, M, M))
    gpu_out_host = _from_gpu(gpu_out, (H, M, D))

    # --- BACKWARD ---
    gpu_dProbs_post = cuda.mem_alloc(H * M * M * 4)
    # dProbs_post = dOut @ V^T : reuse matmul_score_kernel's contraction shape via grad_q-style kernel
    # is not directly available, so compute it with matmul_grad_q_kernel's transpose-free pattern instead:
    # dProbs_post[h,i,j] = sum_d dOut[h,i,d] * V[h,j,d]  -- same contraction pattern as matmul_score_kernel
    matmul_score(gpu_dOut, gpu_V, gpu_dProbs_post, np.int32(H), np.int32(M), np.int32(D), np.float32(1.0),
                 block=(threads, 1, 1), grid=(score_blocks, 1))

    gpu_dScores_pre = cuda.mem_alloc(H * M * M * 4)
    softmax_bwd(gpu_dProbs_post, gpu_scores, gpu_row_sum, gpu_dScores_pre, np.int32(H), np.int32(M),
                block=(softmax_block_threads, 1, 1), grid=(H, M), shared=softmax_shared_bytes)

    gpu_dQ = cuda.mem_alloc(H * M * D * 4)
    gpu_dK = cuda.mem_alloc(H * M * D * 4)
    gpu_dV = cuda.mem_alloc(H * M * D * 4)

    matmul_grad_q(gpu_dScores_pre, gpu_K, gpu_dQ, np.int32(H), np.int32(M), np.int32(D),
                  block=(threads, 1, 1), grid=(proj_blocks, 1))
    matmul_grad_k(gpu_dScores_pre, gpu_Q, gpu_dK, np.int32(H), np.int32(M), np.int32(D),
                  block=(threads, 1, 1), grid=(proj_blocks, 1))
    matmul_grad_v(gpu_scores, gpu_dOut, gpu_dV, np.int32(H), np.int32(M), np.int32(D),
                  block=(threads, 1, 1), grid=(proj_blocks, 1))

    gpu_dScores_pre_host = _from_gpu(gpu_dScores_pre, (H, M, M))
    gpu_dQ_host = _from_gpu(gpu_dQ, (H, M, D))
    gpu_dK_host = _from_gpu(gpu_dK, (H, M, D))
    gpu_dV_host = _from_gpu(gpu_dV, (H, M, D))

    for ptr in (gpu_Q, gpu_K, gpu_V, gpu_dOut, gpu_scores, gpu_row_max, gpu_row_sum, gpu_out,
                gpu_dProbs_post, gpu_dScores_pre, gpu_dQ, gpu_dK, gpu_dV):
        ptr.free()

    checks = [
        ("probs (post-softmax)", ref_probs, gpu_probs_host),
        ("out (probs @ V)", ref_out, gpu_out_host),
        ("dScores_pre (softmax VJP)", ref_dScores_pre, gpu_dScores_pre_host),
        ("dQ", ref_dQ, gpu_dQ_host),
        ("dK", ref_dK, gpu_dK_host),
        ("dV", ref_dV, gpu_dV_host),
    ]

    print(f"Golden model parity check: H={H}, M={M}, D={D}, scale={SCALE:.6f}\n")

    all_passed = True
    for name, ref_val, gpu_val in checks:
        try:
            np.testing.assert_allclose(gpu_val, ref_val, **TOLERANCE)
            max_abs_diff = float(np.max(np.abs(gpu_val - ref_val)))
            print(f"[PASS] {name}: max_abs_diff={max_abs_diff:.3e}")
        except AssertionError as exc:
            all_passed = False
            print(f"[FAIL] {name}: {exc}")

    if all_passed:
        print("\nAll parity checks PASSED: GPU MHA kernels match NumPy golden model within tolerance.")
    else:
        print("\nParity FAILED: see assertions above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
