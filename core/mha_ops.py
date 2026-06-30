"""
Phase 3: MHA Orchestration Layer (Execution Planner).

MHAController wires the standalone kernels in core/mha_kernels.py into a single
forward pass: fused QKV projection -> split into contiguous Q/K/V -> causal
attention scores -> fused softmax -> output projection.

This is the "production wiring" path. test_mha_golden_model.py separately
hand-assembles the same kernel sequence to prove execution-path parity: that
the controller's internal call order/grid-block configuration matches a
manually verified reference sequence, not just that the underlying kernels are
individually correct in isolation.
"""

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import env_config
from core.mha_kernels import MHA_KERNELS_STRING

THREADS_PER_BLOCK = 256
SOFTMAX_BLOCK_THREADS = 64


def compile_mha_module():
    nvcc_options = [
        "-ccbin", env_config.MSVC_142_BIN,
        "-O3",
        "--use_fast_math",
        "-Xcompiler", "/w",
    ]
    return SourceModule(MHA_KERNELS_STRING, options=nvcc_options)


def _grid1d(total_elements, threads=THREADS_PER_BLOCK):
    return (total_elements + threads - 1) // threads


class MHAController:
    """Owns one compiled MHA kernel module and runs the fused-QKV attention pass.

    H, M, D describe the attention shape this controller is configured for:
        H: number of heads, M: sequence length, D: per-head dimension.
    """

    def __init__(self, H: int, M: int, D: int, module=None):
        self.H = H
        self.M = M
        self.D = D
        self.scale = 1.0 / np.sqrt(D)

        self.module = module if module is not None else compile_mha_module()
        self.fn_qkv_fused = self.module.get_function("matmul_qkv_fused")
        self.fn_split_qkv = self.module.get_function("split_qkv_kernel")
        self.fn_matmul_score = self.module.get_function("matmul_score_kernel")
        self.fn_softmax_fwd = self.module.get_function("softmax_fused_forward")
        self.fn_matmul_proj = self.module.get_function("matmul_proj_kernel")

    def matmul_qkv_fused(self, gpu_X, gpu_W, gpu_fused, Din: int, block, grid):
        """Thin pass-through wrapper kept for harnesses that want to call the
        fused QKV kernel directly with explicit launch configuration."""
        self.fn_qkv_fused(gpu_X, gpu_W, gpu_fused,
                          np.int32(self.H), np.int32(self.M), np.int32(Din), np.int32(self.D),
                          block=block, grid=grid)

    def forward(self, X: np.ndarray, W_qkv: np.ndarray):
        """Run the full production forward pass on host numpy inputs.

        Args:
            X: [H, M, Din] float32 input activations
            W_qkv: [Din, 3*D] float32 shared fused QKV projection weight

        Returns:
            (out, probs): out [H, M, D] float32, probs [H, M, M] float32 (host arrays)
        """
        H, M, D = self.H, self.M, self.D
        Din = X.shape[-1]
        X = np.ascontiguousarray(X, dtype=np.float32)
        W_qkv = np.ascontiguousarray(W_qkv, dtype=np.float32)

        gpu_X = cuda.mem_alloc(X.nbytes)
        gpu_W = cuda.mem_alloc(W_qkv.nbytes)
        cuda.memcpy_htod(gpu_X, X)
        cuda.memcpy_htod(gpu_W, W_qkv)

        gpu_fused = cuda.mem_alloc(H * M * 3 * D * 4)
        gpu_Q = cuda.mem_alloc(H * M * D * 4)
        gpu_K = cuda.mem_alloc(H * M * D * 4)
        gpu_V = cuda.mem_alloc(H * M * D * 4)
        gpu_scores = cuda.mem_alloc(H * M * M * 4)
        gpu_row_max = cuda.mem_alloc(H * M * 4)
        gpu_row_sum = cuda.mem_alloc(H * M * 4)
        gpu_out = cuda.mem_alloc(H * M * D * 4)

        try:
            self.fn_qkv_fused(gpu_X, gpu_W, gpu_fused,
                               np.int32(H), np.int32(M), np.int32(Din), np.int32(D),
                               block=(THREADS_PER_BLOCK, 1, 1), grid=(_grid1d(H * M * 3 * D), 1))

            self.fn_split_qkv(gpu_fused, gpu_Q, gpu_K, gpu_V,
                               np.int32(H), np.int32(M), np.int32(D),
                               block=(THREADS_PER_BLOCK, 1, 1), grid=(_grid1d(H * M * D), 1))

            self.fn_matmul_score(gpu_Q, gpu_K, gpu_scores,
                                  np.int32(H), np.int32(M), np.int32(D), np.float32(self.scale),
                                  block=(THREADS_PER_BLOCK, 1, 1), grid=(_grid1d(H * M * M), 1))

            self.fn_softmax_fwd(gpu_scores, gpu_row_max, gpu_row_sum,
                                 np.int32(H), np.int32(M),
                                 block=(SOFTMAX_BLOCK_THREADS, 1, 1), grid=(H, M),
                                 shared=SOFTMAX_BLOCK_THREADS * 4)

            self.fn_matmul_proj(gpu_scores, gpu_V, gpu_out,
                                 np.int32(H), np.int32(M), np.int32(D),
                                 block=(THREADS_PER_BLOCK, 1, 1), grid=(_grid1d(H * M * D), 1))

            out = np.empty((H, M, D), dtype=np.float32)
            probs = np.empty((H, M, M), dtype=np.float32)
            cuda.memcpy_dtoh(out, gpu_out)
            cuda.memcpy_dtoh(probs, gpu_scores)
        finally:
            for ptr in (gpu_X, gpu_W, gpu_fused, gpu_Q, gpu_K, gpu_V,
                        gpu_scores, gpu_row_max, gpu_row_sum, gpu_out):
                ptr.free()

        return out, probs
