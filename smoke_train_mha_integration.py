"""
Quick smoke test: runs a handful of full GPTModel forward/backward/AdamW steps
with synthetic random token batches (no corpus loading) to confirm the
GPU-resident MultiHeadAttention rewrite works correctly inside the real
multi-layer, multi-head, multi-batch training loop -- not just in isolation.

Run with the project's CUDA-enabled virtualenv:
    .\\venv\\Scripts\\python.exe smoke_train_mha_integration.py
"""

import numpy as np
import pycuda.driver as cuda

import env_config  # noqa: F401
import pycuda.autoinit  # noqa: F401

from model.gpt import GPTConfig, GPTModel
from core.loss import SoftmaxCrossEntropy

SEED = 777
B, T = 2, 16
VOCAB = 200
STEPS = 5


def main():
    np.random.seed(SEED)
    config = GPTConfig(
        vocab_size=VOCAB, max_len=T, embedding_dim=32, num_heads=4, num_layers=2,
        dropout_prob=0.0, attention_impl="strided",
    )
    model = GPTModel(config)
    loss_op = SoftmaxCrossEntropy()

    rng = np.random.RandomState(SEED + 1)
    losses = []
    for step in range(1, STEPS + 1):
        host_tokens = rng.randint(0, VOCAB, size=(B, T)).astype(np.int32)
        host_targets = rng.randint(0, VOCAB, size=(B * T,)).astype(np.int32)

        gpu_tokens = cuda.mem_alloc(host_tokens.nbytes)
        cuda.memcpy_htod(gpu_tokens, host_tokens)
        gpu_targets = cuda.mem_alloc(host_targets.nbytes)
        cuda.memcpy_htod(gpu_targets, host_targets)

        gpu_logits = model.forward(gpu_tokens, B, T)
        loss_val, gpu_dlogits = loss_op(gpu_logits, gpu_targets, B * T, VOCAB)
        model.backward(gpu_dlogits, B, T, scale=1.0, accumulate=False)
        model.update_weights(lr=1e-3, step=step)

        losses.append(float(loss_val))
        print(f"step {step}: loss={float(loss_val):.4f}")

        gpu_tokens.free()
        gpu_targets.free()
        gpu_logits.free()
        gpu_dlogits.free()

    if any(not np.isfinite(l) for l in losses):
        print("[FAIL] non-finite loss encountered")
        raise SystemExit(1)

    print("\n[PASS] smoke run completed: full GPTModel (2 layers, 4 heads) forward/backward "
          "with GPU-resident MultiHeadAttention ran cleanly for "
          f"{STEPS} steps, all losses finite: {[round(l, 4) for l in losses]}")


if __name__ == "__main__":
    main()
