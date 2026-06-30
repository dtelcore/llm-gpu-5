"""
Phase 3 Validation Harness: GPU MHA Kernels vs. NumPy Golden Model (Parity Test).

Three independent layers of guarantee, each catching a different failure class:

1. Kernel-level parity (run_kernel_level_parity)
   Compares the raw PyCUDA kernels in core/mha_kernels.py against an independent
   NumPy reference implementation of causal multi-head attention forward + backward
   (VJP form), using pre-split Q/K/V. Catches: wrong math, wrong indexing, wrong
   reduction axis.

2. Fused-QKV representation equivalence (run_fused_qkv_representation_parity)
   Compares a NumPy "triple projection" (X @ W_q, X @ W_k, X @ W_v as independent
   slices of one shared weight matrix) against the GPU fused-QKV kernel + split
   kernel. Catches: stride corruption, pointer-aliasing bugs, and packing-layout
   mistakes introduced specifically by fusing the projection into one kernel.

3. Controller execution-path parity (run_controller_execution_identity)
   Compares a hand-assembled kernel call sequence (this file, mirroring layer 1/2's
   manual launches) against core/mha_ops.py's MHAController.forward(), which is the
   actual production orchestration path. Catches: "correct kernels, wrong wiring" --
   stale buffers, wrong launch config, wrong call order in the real code path that
   kernel-only tests can never see.

This is the correctness gate that must pass before any tiling/fusion/FlashAttention-style
optimization work begins, and before triple-projection code is deleted in favor of the
fused path.

Run with the project's CUDA-enabled virtualenv, e.g.:
    .\\venv\\Scripts\\python.exe test_mha_golden_model.py
"""

import numpy as np
import pycuda.driver as cuda

import env_config  # noqa: F401  (bootstraps MSVC/CUDA environment)
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

from core.mha_ops import MHAController, compile_mha_module, THREADS_PER_BLOCK, SOFTMAX_BLOCK_THREADS


SEED = 1337
H, M, D = 3, 17, 8  # heads, sequence length, head_dim (odd M to stress causal edge rows)
SCALE = 1.0 / np.sqrt(D)
TOLERANCE = dict(rtol=1e-4, atol=1e-5)


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


def run_kernel_level_parity(module):
    """Layer 1: raw per-kernel parity against an independent NumPy reference."""
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

    print(f"--- Layer 1: kernel-level parity (H={H}, M={M}, D={D}, scale={SCALE:.6f}) ---")

    all_passed = True
    for name, ref_val, gpu_val in checks:
        try:
            np.testing.assert_allclose(gpu_val, ref_val, **TOLERANCE)
            max_abs_diff = float(np.max(np.abs(gpu_val - ref_val)))
            print(f"[PASS] {name}: max_abs_diff={max_abs_diff:.3e}")
        except AssertionError as exc:
            all_passed = False
            print(f"[FAIL] {name}: {exc}")

    return all_passed


def run_fused_qkv_representation_parity(module, X, W_qkv, Din):
    """Layer 2: representation equivalence -- triple projection (NumPy) vs.
    fused-QKV kernel + split kernel (GPU), compared as logical Q/K/V tensors,
    not as raw fused memory regions."""
    print(f"\n--- Layer 2: fused-QKV representation equivalence (Din={Din}) ---")

    fn_qkv_fused = module.get_function("matmul_qkv_fused")
    fn_split_qkv = module.get_function("split_qkv_kernel")

    W_q = W_qkv[:, 0:D]
    W_k = W_qkv[:, D:2 * D]
    W_v = W_qkv[:, 2 * D:3 * D]
    Q_ref = np.einsum("hid,dk->hik", X, W_q).astype(np.float32)
    K_ref = np.einsum("hid,dk->hik", X, W_k).astype(np.float32)
    V_ref = np.einsum("hid,dk->hik", X, W_v).astype(np.float32)

    gpu_X, gpu_W = _to_gpu(X), _to_gpu(W_qkv)
    gpu_fused = cuda.mem_alloc(H * M * 3 * D * 4)
    gpu_Q = cuda.mem_alloc(H * M * D * 4)
    gpu_K = cuda.mem_alloc(H * M * D * 4)
    gpu_V = cuda.mem_alloc(H * M * D * 4)

    threads = THREADS_PER_BLOCK
    fused_blocks = (H * M * 3 * D + threads - 1) // threads
    proj_blocks = (H * M * D + threads - 1) // threads

    fn_qkv_fused(gpu_X, gpu_W, gpu_fused, np.int32(H), np.int32(M), np.int32(Din), np.int32(D),
                 block=(threads, 1, 1), grid=(fused_blocks, 1))
    fn_split_qkv(gpu_fused, gpu_Q, gpu_K, gpu_V, np.int32(H), np.int32(M), np.int32(D),
                 block=(threads, 1, 1), grid=(proj_blocks, 1))

    Q_gpu = _from_gpu(gpu_Q, (H, M, D))
    K_gpu = _from_gpu(gpu_K, (H, M, D))
    V_gpu = _from_gpu(gpu_V, (H, M, D))

    for ptr in (gpu_X, gpu_W, gpu_fused, gpu_Q, gpu_K, gpu_V):
        ptr.free()

    all_passed = True
    for name, ref_val, gpu_val in (("Q", Q_ref, Q_gpu), ("K", K_ref, K_gpu), ("V", V_ref, V_gpu)):
        try:
            np.testing.assert_allclose(gpu_val, ref_val, **TOLERANCE)
            max_abs_diff = float(np.max(np.abs(gpu_val - ref_val)))
            print(f"[PASS] {name} (fused-vs-triple-projection): max_abs_diff={max_abs_diff:.3e}")
        except AssertionError as exc:
            all_passed = False
            print(f"[FAIL] {name} (fused-vs-triple-projection): {exc}")

    return all_passed


def _manual_controller_forward(module, X, W_qkv, Din):
    """Hand-assembled kernel call sequence, independent of core/mha_ops.py, used as
    the execution-path oracle that MHAController.forward() is checked against."""
    fn_qkv_fused = module.get_function("matmul_qkv_fused")
    fn_split_qkv = module.get_function("split_qkv_kernel")
    fn_matmul_score = module.get_function("matmul_score_kernel")
    fn_softmax_fwd = module.get_function("softmax_fused_forward")
    fn_matmul_proj = module.get_function("matmul_proj_kernel")

    threads = THREADS_PER_BLOCK
    gpu_X, gpu_W = _to_gpu(X), _to_gpu(W_qkv)
    gpu_fused = cuda.mem_alloc(H * M * 3 * D * 4)
    gpu_Q = cuda.mem_alloc(H * M * D * 4)
    gpu_K = cuda.mem_alloc(H * M * D * 4)
    gpu_V = cuda.mem_alloc(H * M * D * 4)
    gpu_scores = cuda.mem_alloc(H * M * M * 4)
    gpu_row_max = cuda.mem_alloc(H * M * 4)
    gpu_row_sum = cuda.mem_alloc(H * M * 4)
    gpu_out = cuda.mem_alloc(H * M * D * 4)

    try:
        fn_qkv_fused(gpu_X, gpu_W, gpu_fused, np.int32(H), np.int32(M), np.int32(Din), np.int32(D),
                     block=(threads, 1, 1), grid=((H * M * 3 * D + threads - 1) // threads, 1))
        fn_split_qkv(gpu_fused, gpu_Q, gpu_K, gpu_V, np.int32(H), np.int32(M), np.int32(D),
                     block=(threads, 1, 1), grid=((H * M * D + threads - 1) // threads, 1))
        fn_matmul_score(gpu_Q, gpu_K, gpu_scores, np.int32(H), np.int32(M), np.int32(D), np.float32(SCALE),
                        block=(threads, 1, 1), grid=((H * M * M + threads - 1) // threads, 1))
        fn_softmax_fwd(gpu_scores, gpu_row_max, gpu_row_sum, np.int32(H), np.int32(M),
                       block=(SOFTMAX_BLOCK_THREADS, 1, 1), grid=(H, M),
                       shared=SOFTMAX_BLOCK_THREADS * 4)
        fn_matmul_proj(gpu_scores, gpu_V, gpu_out, np.int32(H), np.int32(M), np.int32(D),
                      block=(threads, 1, 1), grid=((H * M * D + threads - 1) // threads, 1))

        out = _from_gpu(gpu_out, (H, M, D))
        probs = _from_gpu(gpu_scores, (H, M, M))
    finally:
        for ptr in (gpu_X, gpu_W, gpu_fused, gpu_Q, gpu_K, gpu_V,
                    gpu_scores, gpu_row_max, gpu_row_sum, gpu_out):
            ptr.free()

    return out, probs


def run_controller_execution_identity(module, X, W_qkv, Din):
    """Layer 3: production wiring check -- MHAController.forward() (core/mha_ops.py)
    vs. an independently hand-assembled kernel sequence with the same inputs.
    A pass here proves correct kernels are *also* correctly wired in production,
    not just correct in isolation."""
    print(f"\n--- Layer 3: controller execution-path parity ---")

    out_manual, probs_manual = _manual_controller_forward(module, X, W_qkv, Din)

    controller = MHAController(H, M, D, module=module)
    out_controller, probs_controller = controller.forward(X, W_qkv)

    all_passed = True
    for name, manual_val, controller_val in (("out", out_manual, out_controller),
                                              ("probs", probs_manual, probs_controller)):
        try:
            np.testing.assert_allclose(controller_val, manual_val, **TOLERANCE)
            max_abs_diff = float(np.max(np.abs(controller_val - manual_val)))
            print(f"[PASS] {name} (controller-vs-manual): max_abs_diff={max_abs_diff:.3e}")
        except AssertionError as exc:
            all_passed = False
            print(f"[FAIL] {name} (controller-vs-manual): {exc}")

    return all_passed


def main():
    np.random.seed(SEED)
    module = compile_mha_module()

    Din = 6
    X = np.random.normal(0.0, 1.0, size=(H, M, Din)).astype(np.float32)
    W_qkv = np.random.normal(0.0, 1.0 / np.sqrt(Din), size=(Din, 3 * D)).astype(np.float32)

    layer1_passed = run_kernel_level_parity(module)
    layer2_passed = run_fused_qkv_representation_parity(module, X, W_qkv, Din)
    layer3_passed = run_controller_execution_identity(module, X, W_qkv, Din)

    print()
    if layer1_passed and layer2_passed and layer3_passed:
        print("ALL PARITY CHECKS PASSED: kernels, fused-QKV representation, and controller "
              "wiring are all numerically equivalent. Safe to proceed with optimization work.")
    else:
        print("PARITY FAILED: see [FAIL] lines above. Do not delete triple-projection code "
              "or proceed to optimization until every layer passes.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
