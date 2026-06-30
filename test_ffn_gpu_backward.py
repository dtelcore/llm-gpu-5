"""
Diagnostic Oracle Test: FeedForward GPU-Resident Backward vs. Legacy CPU NumPy Backward.

Verifies numerical parity between FeedForward.backward()'s two execution paths
(use_cpu_backward=True legacy oracle vs. use_cpu_backward=False GPU-resident path)
and reports the wall-clock time delta between them.

Run with the project's CUDA-enabled virtualenv, e.g.:
    .\\venv\\Scripts\\python.exe test_ffn_gpu_backward.py
"""

import time

import numpy as np
import pycuda.driver as cuda

import env_config  # noqa: F401  (bootstraps MSVC/CUDA environment)
import pycuda.autoinit  # noqa: F401  (initializes CUDA context)

from model.gpt import GPTConfig, FeedForward


SEED = 1337
BATCH_TIMES_SEQ = 6  # M = B * T for this isolated layer test
TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def _download(gpu_ptr, shape):
    host_arr = np.empty(shape, dtype=np.float32)
    cuda.memcpy_dtoh(host_arr, gpu_ptr)
    return host_arr


def main():
    np.random.seed(SEED)

    config = GPTConfig(vocab_size=64, embedding_dim=16, num_heads=2, max_len=8)
    ffn = FeedForward(config)

    M = BATCH_TIMES_SEQ
    C = config.embedding_dim

    host_input = np.random.normal(0.0, 1.0, size=(M, C)).astype(np.float32)
    gpu_input = cuda.mem_alloc(host_input.nbytes)
    cuda.memcpy_htod(gpu_input, host_input)

    # Single forward pass feeds both backward paths, isolating the comparison to backward only.
    gpu_output = ffn.forward(gpu_input, B=1, T=M)
    host_output = _download(gpu_output, (M, C))

    # Deterministic synthetic scalar loss: L = 0.5 * sum(output^2) -> dL/dOutput = output
    loss_value = float(0.5 * np.sum(host_output ** 2))
    host_dOut = host_output.copy()
    gpu_dOut = cuda.mem_alloc(host_dOut.nbytes)
    cuda.memcpy_htod(gpu_dOut, host_dOut)

    print(f"Forward scalar loss (identical for both paths, same forward pass): {loss_value:.6f}")

    # --- CPU-ORACLE PATH ---
    ffn.use_cpu_backward = True
    cpu_start = time.perf_counter()
    gpu_dIn_cpu = ffn.backward(gpu_dOut, B=1, T=M, accumulate=False)
    cpu_elapsed_ms = (time.perf_counter() - cpu_start) * 1000.0

    cpu_dIn = _download(gpu_dIn_cpu, (M, C))
    cpu_dFcW = _download(ffn.c_fc_w.gpu_grads, ffn.c_fc_w.shape)
    cpu_dFcB = _download(ffn.c_fc_b.gpu_grads, ffn.c_fc_b.shape)
    cpu_dProjW = _download(ffn.c_proj_w.gpu_grads, ffn.c_proj_w.shape)
    cpu_dProjB = _download(ffn.c_proj_b.gpu_grads, ffn.c_proj_b.shape)
    gpu_dIn_cpu.free()

    # --- GPU-RESIDENT PATH (same forward caches, still alive: free_forward_caches() not called) ---
    ffn.use_cpu_backward = False
    gpu_start = time.perf_counter()
    gpu_dIn_gpu = ffn.backward(gpu_dOut, B=1, T=M, accumulate=False)
    gpu_elapsed_ms = (time.perf_counter() - gpu_start) * 1000.0

    gpu_dIn = _download(gpu_dIn_gpu, (M, C))
    gpu_dFcW = _download(ffn.c_fc_w.gpu_grads, ffn.c_fc_w.shape)
    gpu_dFcB = _download(ffn.c_fc_b.gpu_grads, ffn.c_fc_b.shape)
    gpu_dProjW = _download(ffn.c_proj_w.gpu_grads, ffn.c_proj_w.shape)
    gpu_dProjB = _download(ffn.c_proj_b.gpu_grads, ffn.c_proj_b.shape)
    gpu_dIn_gpu.free()

    print(f"CPU-oracle backward time: {cpu_elapsed_ms:.4f} ms")
    print(f"GPU-resident backward time: {gpu_elapsed_ms:.4f} ms")
    if gpu_elapsed_ms > 0:
        print(f"Speedup: {cpu_elapsed_ms / gpu_elapsed_ms:.2f}x")

    checks = [
        ("dFcW", cpu_dFcW, gpu_dFcW),
        ("dFcB", cpu_dFcB, gpu_dFcB),
        ("dProjW", cpu_dProjW, gpu_dProjW),
        ("dProjB", cpu_dProjB, gpu_dProjB),
        ("dIn", cpu_dIn, gpu_dIn),
    ]

    all_passed = True
    for name, cpu_val, gpu_val in checks:
        try:
            np.testing.assert_allclose(gpu_val, cpu_val, **TOLERANCE)
            max_abs_diff = float(np.max(np.abs(gpu_val - cpu_val)))
            print(f"[PASS] {name}: max_abs_diff={max_abs_diff:.3e}")
        except AssertionError as exc:
            all_passed = False
            print(f"[FAIL] {name}: {exc}")

    ffn.free_forward_caches()
    gpu_input.free()
    gpu_dOut.free()
    gpu_output.free()

    if all_passed:
        print("\nAll parity checks PASSED: GPU-resident FFN backward matches CPU oracle within tolerance.")
    else:
        print("\nParity FAILED: see assertions above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
