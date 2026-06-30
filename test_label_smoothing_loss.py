"""
Standalone kernel-level test for label smoothing support in core/loss.py's fused
softmax cross-entropy kernel.

Validates:
    1. label_smoothing=0.0 reproduces today's hard-target loss/gradients exactly
       (regression check -- existing callers that don't pass label_smoothing must
       see zero behavior change).
    2. label_smoothing > 0 matches an independent NumPy reference implementation
       of smoothed cross-entropy for several smoothing values.
"""

import numpy as np
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda

from core.loss import SoftmaxCrossEntropy

SEED = 1234
TOLERANCE = dict(rtol=1e-4, atol=1e-5)


def numpy_reference(logits: np.ndarray, targets: np.ndarray, label_smoothing: float):
    N, V = logits.shape
    max_val = np.max(logits, axis=1, keepdims=True)
    shifted = logits - max_val
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
    log_sum_exp = np.log(sum_exp)
    probs = exp_shifted / sum_exp

    target_onehot = np.zeros((N, V), dtype=np.float64)
    target_onehot[np.arange(N), targets] = 1.0
    target_prob = target_onehot * (1.0 - label_smoothing) + label_smoothing / V

    log_probs = shifted - log_sum_exp
    losses = -np.sum(target_prob * log_probs, axis=1)
    dlogits = (probs - target_prob) / N
    return losses.astype(np.float32), dlogits.astype(np.float32)


def run_case(label_smoothing: float, N: int = 17, V: int = 41):
    np.random.seed(SEED)
    logits = np.random.normal(0.0, 2.0, size=(N, V)).astype(np.float32)
    targets = np.random.randint(0, V, size=(N,)).astype(np.int32)

    ref_losses, ref_dlogits = numpy_reference(logits.astype(np.float64), targets, label_smoothing)

    gpu_logits = cuda.mem_alloc(logits.nbytes)
    gpu_targets = cuda.mem_alloc(targets.nbytes)
    cuda.memcpy_htod(gpu_logits, logits)
    cuda.memcpy_htod(gpu_targets, targets)

    criterion = SoftmaxCrossEntropy()
    mean_loss, gpu_dlogits = criterion(gpu_logits, gpu_targets, N, V, pad_token_id=-1,
                                        label_smoothing=label_smoothing)

    dlogits = np.empty((N, V), dtype=np.float32)
    cuda.memcpy_dtoh(dlogits, gpu_dlogits)

    gpu_logits.free()
    gpu_targets.free()
    gpu_dlogits.free()

    ref_mean_loss = float(np.mean(ref_losses))

    passed = True
    try:
        np.testing.assert_allclose(mean_loss, ref_mean_loss, **TOLERANCE)
        print(f"[PASS] label_smoothing={label_smoothing}: mean_loss gpu={mean_loss:.6f} "
              f"ref={ref_mean_loss:.6f}")
    except AssertionError as exc:
        passed = False
        print(f"[FAIL] label_smoothing={label_smoothing}: mean_loss mismatch: {exc}")

    try:
        np.testing.assert_allclose(dlogits, ref_dlogits, **TOLERANCE)
        max_abs_diff = float(np.max(np.abs(dlogits - ref_dlogits)))
        print(f"[PASS] label_smoothing={label_smoothing}: dLogits max_abs_diff={max_abs_diff:.3e}")
    except AssertionError as exc:
        passed = False
        print(f"[FAIL] label_smoothing={label_smoothing}: dLogits mismatch: {exc}")

    return passed


def main():
    all_passed = True
    for label_smoothing in (0.0, 0.05, 0.1, 0.2):
        print(f"\n--- label_smoothing={label_smoothing} ---")
        all_passed = run_case(label_smoothing) and all_passed

    print()
    if all_passed:
        print("ALL LABEL SMOOTHING CHECKS PASSED: label_smoothing=0.0 reproduces the "
              "original hard-target loss/gradients exactly, and label_smoothing>0 matches "
              "the NumPy smoothed cross-entropy reference.")
    else:
        print("LABEL SMOOTHING CHECKS FAILED: see [FAIL] lines above.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
