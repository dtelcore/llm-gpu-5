# train.py
"""
End-to-End Training Loop Orchestrator for Custom GPU-Accelerated GPT Model.

Implements complete autoregressive training pipeline:
1. Tokenize corpus to integer sequences
2. Structure mini-batches for GPU ingestion
3. Forward pass: Token IDs → Embeddings → Attention → FFN → Logits
4. Loss computation: Fused softmax cross-entropy with numerical stability
5. Backward pass: Chain rule differentiation through all layers
6. AdamW optimization: Stateful first/second moment tracking
7. Cache cleanup: Explicit forward activation deallocation post-backward

Target: NVIDIA GeForce GT 730 (Kepler) with 1-2GB VRAM constraint
"""

import os
import numpy as np
import pycuda.driver as cuda

# Force initial system driver context initialization mappings
import env_config
import pycuda.autoinit

from corpus_utils import (
    TRAINING_CORPUS,
    FINEWEB_DATASET_NAME,
    DEFAULT_MAX_TRAINING_DOCS,
    build_shared_tokenizer,
    load_dataset_corpus,
    load_or_build_token_matrix,
    sample_token_batch,
    split_corpus_for_validation,
)
from logging_config import logger
from model.gpt import GPTConfig, GPTModel
from core.loss import SoftmaxCrossEntropy
from gpu_memory import install_global_memory_pool, get_memory_pool_stats_mb, free_held_pool_blocks
from training_metrics import TrainingMetrics
from run_config import RunConfig


GOAL_LOSS_THRESHOLD = 2.0
GOAL_PPL_THRESHOLD = 5.0
GOAL_IMPROVEMENT_EPSILON = 1e-6
PROBE_CHECKPOINT_STEPS = (500, 1000, 2000)
PROBE_PROMPT = "the"
PROBE_MEMORIZATION_PREFIX_LEN = 32


def run_training_engine():
    """Main training orchestrator: load → tokenize → forward → loss → backward → optimize."""
    install_global_memory_pool()

    logger.info("="*73)
    logger.info("[INIT] INITIALIZING CUSTOM Kepler GT 730 TRAINING BACKEND ENGINE")
    logger.info("="*73)

    run_config = RunConfig.load("output/last_run_config.json")
    
    # 1. Load training corpus
    dataset_name = run_config.dataset
    corpus, corpus_source = load_dataset_corpus(dataset_name)
    if corpus_source == "minimal":
        logger.info(f"Using fallback minimal corpus: {len(corpus)} sentences")
    else:
        logger.info(f"[OK] Loaded dataset {dataset_name}: {len(corpus):,} documents")
    
    # Optimize for training: limit large corpora to manageable size
    max_training_docs = run_config.corpus_limit
    if len(corpus) > max_training_docs:
        logger.info(f"  Corpus is large ({len(corpus):,} docs) - limiting to {max_training_docs:,} for this training run")
        corpus = corpus[:max_training_docs]
        logger.info(f"  [OK] Using first {len(corpus):,} documents for training")
    
    logger.info(f"Corpus loaded from {corpus_source}: {len(corpus)} documents")
    
    train_corpus, val_corpus = split_corpus_for_validation(corpus)
    logger.info(f"Training docs: {len(train_corpus):,} | Validation docs: {len(val_corpus):,}")

    # 2. Configure and build tokenizer mappings
    tokenizer, tokenizer_docs, tokenizer_source = build_shared_tokenizer(
        dataset_name,
        source_docs=train_corpus,
        fallback_docs=train_corpus,
    )
    logger.info(f"Shared tokenizer vocab built from {len(tokenizer_docs):,} documents ({tokenizer_source})")
    logger.info(f"Vocabulary Size Extracted: {tokenizer.vocab_size} tokens")
    logger.info(f"Special Tokens: BOS={tokenizer.BOS_ID}, PAD={tokenizer.PAD_ID}")

    # 3. Establish strict network boundaries optimized for low VRAM footprints
    B = run_config.batch_size
    T = run_config.max_len
    embedding_dim = run_config.embedding_dim
    num_heads = run_config.num_heads
    num_layers = run_config.num_layers
    
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_len=T,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        attention_impl=run_config.attention_impl,
        dropout_prob=0.0  # Clear structural dropout noise to verify raw mathematical convergence
    )
    logger.info(
        f"Model config: vocab_size={config.vocab_size}, embedding_dim={embedding_dim}, "
        f"num_heads={num_heads}, num_layers={num_layers}, attention_impl={config.attention_impl}"
    )
    
    # 4. Instantiate core model components and loss operator
    logger.info("Instantiating GPTModel and SoftmaxCrossEntropy...")
    model = GPTModel(config)
    criterion = SoftmaxCrossEntropy()
    logger.info("[OK] Model components instantiated successfully")
    
    # 5. Tokenize and structure raw data text matrices
    # Generate structural matrix layouts of shape (Samples, Sequence + 1) to build autoregressive target cuts
    logger.info("Encoding corpus to token matrices...")
    logger.info(f"  Corpus size: {len(train_corpus):,} training documents")
    logger.info(f"  Max sequence length: {T + 1} tokens")
    logger.info("  This may take several minutes for large datasets...")
    raw_aligned_matrix = load_or_build_token_matrix(
        tokenizer,
        train_corpus,
        max_sequence_length=T + 1,
        cache_namespace="train",
        dataset_name=dataset_name,
    )
    logger.info(f"  [OK] Matrix created: shape {raw_aligned_matrix.shape}")

    val_aligned_matrix = load_or_build_token_matrix(
        tokenizer,
        val_corpus,
        max_sequence_length=T + 1,
        cache_namespace="train_val",
        dataset_name=f"{dataset_name}_val",
    )
    logger.info(f"  [OK] Validation matrix created: shape {val_aligned_matrix.shape}")
    
    batch_rng = np.random.default_rng()
    input_tokens_sample, target_tokens_sample, _ = sample_token_batch(raw_aligned_matrix, B, T, rng=batch_rng)
    val_batch_rng = np.random.default_rng(1337)
    val_batch_size = min(B, max(1, len(val_corpus)))
    val_input_tokens_sample, val_target_tokens_sample, _ = sample_token_batch(
        val_aligned_matrix,
        val_batch_size,
        T,
        rng=val_batch_rng,
    )
    
    logger.info(f"Input shape: {input_tokens_sample.shape}, Target shape: {target_tokens_sample.shape}")
    logger.info(f"Validation input shape: {val_input_tokens_sample.shape}, Target shape: {val_target_tokens_sample.shape}")
    
    # Push explicit data matrices over PCIe channel to Device VRAM address locations
    logger.info("Allocating VRAM and transferring token matrices to GPU...")
    gpu_input_tokens = cuda.mem_alloc(input_tokens_sample.nbytes)
    gpu_target_tokens = cuda.mem_alloc(target_tokens_sample.nbytes)
    gpu_val_input_tokens = cuda.mem_alloc(val_input_tokens_sample.nbytes)
    gpu_val_target_tokens = cuda.mem_alloc(val_target_tokens_sample.nbytes)
    
    cuda.memcpy_htod(gpu_input_tokens, input_tokens_sample.astype(np.int32))
    cuda.memcpy_htod(gpu_target_tokens, target_tokens_sample.astype(np.int32))
    cuda.memcpy_htod(gpu_val_input_tokens, val_input_tokens_sample.astype(np.int32))
    cuda.memcpy_htod(gpu_val_target_tokens, val_target_tokens_sample.astype(np.int32))
    logger.info("[OK] Data transferred to VRAM successfully")
    
    N = B * T
    V = config.vocab_size
    
    # Hyperparameters for optimization
    learning_rate = run_config.learning_rate
    total_steps = run_config.total_steps
    grad_accum = run_config.grad_accum
    batch_tokens = B * T * grad_accum
    best_goal_loss = None
    best_goal_ppl = None
    best_checkpoint_path = f"output/checkpoints/{run_config.name}.best.npz"
    
    logger.info(f"Training config: batch_size={B}, grad_accum={grad_accum}, seq_len={T}, learning_rate={learning_rate}, total_steps={total_steps}")
    logger.info("="*73)
    logger.info(f"Starting Training Loop: {total_steps} Iterations")
    logger.info("="*73)

    metrics = TrainingMetrics(total_steps=total_steps, log_interval=1, backend="cuda")
    metrics.start()

    try:
        for step in range(1, total_steps + 1):
            logger.debug(f"Step {step}: Starting forward/backward/optimize cycle")
            metrics.step_start()

            model.zero_grad()
            step_loss_value = 0.0

            for micro_step in range(grad_accum):
                input_tokens_sample, target_tokens_sample, _ = sample_token_batch(raw_aligned_matrix, B, T, rng=batch_rng)
                cuda.memcpy_htod(gpu_input_tokens, input_tokens_sample)
                cuda.memcpy_htod(gpu_target_tokens, target_tokens_sample)
                
                # --- PHASE A: FORWARD PROPAGATION PASS ---
                gpu_logits = model.forward(gpu_input_tokens, B, T)
                
                # --- PHASE B: LOSS COMPUTATION & INITIAL GRADIENTS DERIVATION ---
                micro_loss, gpu_dLogits = criterion(gpu_logits, gpu_target_tokens, N, V)
                gpu_logits.free()
                
                step_loss_value += micro_loss / grad_accum

                # --- PHASE C: MANUAL BACKPROPAGATION TRAVERSAL LOOP ---
                model.backward(gpu_dLogits, B, T, scale=1.0/grad_accum, accumulate=True)
                gpu_dLogits.free()
                
                # --- PHASE E: MANDATORY PROACTIVE VRAM FOOTPRINT SCRUBBING ---
                model.free_forward_caches()
                
            loss_value = step_loss_value

            current_ppl = float(np.exp(loss_value))
            if loss_value < GOAL_LOSS_THRESHOLD and current_ppl < GOAL_PPL_THRESHOLD:
                is_first_goal_save = best_goal_loss is None
                loss_improvement = None if best_goal_loss is None else float(best_goal_loss - loss_value)
                if is_first_goal_save or loss_improvement > GOAL_IMPROVEMENT_EPSILON:
                    if is_first_goal_save:
                        logger.info(
                            f"[GOAL] Targets reached at step {step}: loss={loss_value:.4f}, ppl={current_ppl:.2f}"
                        )
                    else:
                        logger.info(
                            f"[GOAL] Improved best checkpoint at step {step}: loss={loss_value:.4f}, "
                            f"ppl={current_ppl:.2f}, delta_loss={loss_improvement:.6f}"
                        )
                    logger.info(
                        f"[GOAL] Saving best checkpoint to {best_checkpoint_path}..."
                    )
                    model.save_checkpoint(best_checkpoint_path)
                    logger.info("[GOAL] Best checkpoint saved")
                    best_goal_loss = float(loss_value)
                    best_goal_ppl = current_ppl

            grad_norm = None
            if metrics.should_log_step(step):
                grad_norm = model.compute_grad_norm()

            val_loss = None
            if metrics.should_log_step(step):
                val_logits = model.forward(gpu_val_input_tokens, val_batch_size, T)
                val_loss, gpu_val_dLogits = criterion(val_logits, gpu_val_target_tokens, val_batch_size * T, V)
                val_logits.free()
                gpu_val_dLogits.free()
                model.free_forward_caches()
            
            # --- PHASE D: STATEFUL ADAMW WEIGHT OPTIMIZATION STEP ---
            model.update_weights(lr=learning_rate, step=step)
            logger.debug(f"Step {step}: AdamW weight update applied")
            pool_used_mb, pool_total_mb = get_memory_pool_stats_mb()

            metrics.step_end(
                loss_value,
                lr=learning_rate,
                grad_norm=grad_norm,
                batch_tokens=batch_tokens,
                pool_used_mb=pool_used_mb,
                pool_total_mb=pool_total_mb,
                val_loss=val_loss,
            )

        metrics.finalize()
        logger.info(f"\nGoal Metrics:")
        logger.info(f"  Target loss:      < {GOAL_LOSS_THRESHOLD:.2f}")
        logger.info(f"  Target PPL:       < {GOAL_PPL_THRESHOLD:.2f}")
        if best_goal_loss is not None:
            logger.info(f"  Reached:          YES")
            logger.info(f"  Best loss:        {best_goal_loss:.6f}")
            logger.info(f"  Best PPL:         {best_goal_ppl:.2f}")
            logger.info(f"  Best checkpoint:  {best_checkpoint_path}")
        else:
            logger.info(f"  Reached:          NO")

    finally:
        # 6. Save trained model weights to disk checkpoint before cleanup
        logger.info("="*73)
        logger.info("[SAVE] Saving trained model checkpoint to disk...")
        checkpoint_path = "output/checkpoints/gpt_model_latest.npz"
        model.save_checkpoint(checkpoint_path)

        try:
            from generate import format_generation_probes, run_generation_probes

            def save_and_probe(probe_checkpoint_path, label):
                model.save_checkpoint(probe_checkpoint_path)
                probe_results = run_generation_probes(
                    probe_checkpoint_path,
                    prompt=PROBE_PROMPT,
                    memorization_prefix=corpus[0][:PROBE_MEMORIZATION_PREFIX_LEN] if corpus else PROBE_PROMPT,
                    top_k=10,
                    max_new_tokens=40,
                    temperature=0.0,
                    num_heads=config.num_heads,
                    source_docs=corpus,
                )
                logger.info(f"\n[{label}] Generation probes:\n" + format_generation_probes(probe_results))

            checkpoint_root, checkpoint_ext = os.path.splitext(checkpoint_path)
            for probe_step in PROBE_CHECKPOINT_STEPS:
                if probe_step > total_steps:
                    continue
                probe_checkpoint_path = f"{checkpoint_root}.step{probe_step}{checkpoint_ext}"
                save_and_probe(probe_checkpoint_path, f"PROBE@{probe_step}")

            probe_results = run_generation_probes(
                checkpoint_path,
                prompt=PROBE_PROMPT,
                memorization_prefix=corpus[0][:PROBE_MEMORIZATION_PREFIX_LEN] if corpus else PROBE_PROMPT,
                top_k=10,
                max_new_tokens=40,
                temperature=0.0,
                num_heads=config.num_heads,
                source_docs=corpus,
            )
            logger.info("\nGeneration probes:\n" + format_generation_probes(probe_results))
        except Exception as exc:
            logger.warning(f"Generation probes skipped: {exc}")
        
        # 7. Complete physical hardware teardown cleanup tracking to protect OS environments
        logger.info("="*73)
        logger.info("Launching hardware memory cleanup and VRAM deallocation...")
        logger.debug("Deallocating token tensors...")
        gpu_input_tokens.free()
        gpu_target_tokens.free()
        gpu_val_input_tokens.free()
        gpu_val_target_tokens.free()
        
        # Complete full structural parameters unmappings inside inner layers
        logger.debug("Deallocating embedding parameters...")
        model.embedding.wte.free()
        model.embedding.wpe.free()
        
        logger.debug("Deallocating transformer block parameters...")
        for i, block in enumerate(model.blocks):
            block.ln_1_gamma.free()
            block.ln_1_beta.free()
            block.ln_2_gamma.free()
            block.ln_2_beta.free()
            block.attn.c_attn_w.free()
            block.attn.c_attn_b.free()
            block.attn.c_proj_w.free()
            block.attn.c_proj_b.free()
            block.mlp.c_fc_w.free()
            block.mlp.c_fc_b.free()
            block.mlp.c_proj_w.free()
            block.mlp.c_proj_b.free()
        
        logger.debug("Deallocating final layer parameters...")
        model.ln_f_gamma.free()
        model.ln_f_beta.free()
        model.lm_head_w.free()
        free_held_pool_blocks()
        
        logger.info("[OK] Clean Teardown Successful. VRAM fully restored to OS environment allocations.")
        logger.info("="*73)


if __name__ == "__main__":
    run_training_engine()
