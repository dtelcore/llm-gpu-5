"""Training metrics tracking and logging."""

import time
import json
import csv
import os
import numpy as np
from datetime import datetime, timedelta
from logging_config import logger
import pycuda.driver as cuda

from gpu_memory import get_memory_pool_stats_mb


class TrainingMetrics:
    """Track and log training metrics in real-time."""
    
    def __init__(self, total_steps, log_interval=1, backend="cuda", log_prefix="output/training_metrics_latest"):
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.backend = backend
        self.csv_path = f"{log_prefix}.csv"
        self.jsonl_path = f"{log_prefix}.jsonl"
        
        self.step = 0
        self.start_time = None
        self.step_times = []
        self.losses = []
        self.perplexities = []
        self.last_lr = None
        self.last_grad_norm = None
        self.last_batch_tokens = None
        self.last_pool_used_mb = None
        self.last_pool_total_mb = None
        self.last_val_loss = None
        self.last_val_ppl = None
        
        self.initial_free_mem = None
        self.initial_total_mem = None

    def should_log_step(self, step_number):
        """Return True when a step should emit the compact progress line."""
        return step_number % self.log_interval == 0 or step_number == 1 or step_number == self.total_steps
    
    def start(self):
        """Initialize training metrics."""
        self.start_time = time.time()
        
        # Get initial GPU memory
        self.initial_free_mem, self.initial_total_mem = cuda.mem_get_info()
        logger.info(f"GPU Memory: {self.initial_total_mem / 1024**2:.0f}MB total, {self.initial_free_mem / 1024**2:.0f}MB free")
        
        # Initialize telemetry files
        os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss", "perplexity", "learning_rate", "tokens_per_sec", "grad_norm", "vram_mb"])
            
        with open(self.jsonl_path, "w") as f:
            pass  # Create empty file
    
    def step_start(self):
        """Mark start of a training step."""
        self.step_time_start = time.time()
    
    def step_end(self, loss_value, lr=None, grad_norm=None, batch_tokens=None,
                 pool_used_mb=None, pool_total_mb=None, val_loss=None):
        """Record metrics for completed step."""
        step_time = time.time() - self.step_time_start
        self.step_times.append(step_time)
        self.losses.append(loss_value)
        self.last_lr = lr
        self.last_grad_norm = grad_norm
        self.last_batch_tokens = batch_tokens
        self.last_pool_used_mb = pool_used_mb
        self.last_pool_total_mb = pool_total_mb
        self.last_val_loss = val_loss
        self.last_val_ppl = np.exp(val_loss) if val_loss is not None else None
        
        # Calculate perplexity
        perplexity = np.exp(loss_value)
        self.perplexities.append(perplexity)
        
        self.step += 1
        
        # Calculate tokens_per_sec for telemetry
        tokens_per_sec = 0.0
        if batch_tokens and step_time > 0:
            tokens_per_sec = batch_tokens / step_time
            
        # Write telemetry
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([self.step, loss_value, perplexity, lr, tokens_per_sec, grad_norm, pool_used_mb])
            
        with open(self.jsonl_path, "a") as f:
            json.dump({
                "step": self.step,
                "loss": loss_value,
                "perplexity": perplexity,
                "learning_rate": lr,
                "tokens_per_sec": tokens_per_sec,
                "grad_norm": grad_norm,
                "vram_mb": pool_used_mb
            }, f)
            f.write("\n")
        
        # Log if interval reached
        if self.should_log_step(self.step):
            self.log_step()

    def _format_duration(self, seconds):
        """Format seconds as XmYYs or XhYYmZZs for compact progress logs."""
        total_seconds = max(0, int(seconds))
        minutes, secs = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours}h{minutes:02d}m{secs:02d}s"
        return f"{minutes}m{secs:02d}s"

    def _format_optional(self, value, precision):
        """Format optional numeric values without inventing fake measurements."""
        if value is None:
            return "n/a"
        return f"{value:.{precision}f}"

    def _format_lr(self, value):
        """Keep very small learning rates visible instead of rounding them to zero."""
        if value is None:
            return "n/a"
        if value == 0 or abs(value) >= 1e-4:
            return f"{value:.6f}"
        return f"{value:.2e}"
    
    def log_step(self):
        """Log current step statistics."""
        elapsed = time.time() - self.start_time
        pool_used_mb = self.last_pool_used_mb
        pool_total_mb = self.last_pool_total_mb
        if pool_used_mb is None and pool_total_mb is None:
            pool_used_mb, pool_total_mb = get_memory_pool_stats_mb()
        
        # Memory stats
        free_mem, total_mem = cuda.mem_get_info()
        device_used_mb = (total_mem - free_mem) / 1024**2
        
        # Timing stats
        current_step_time = self.step_times[-1] if self.step_times else 0
        avg_step_time = np.mean(self.step_times) if self.step_times else 0
        eta_step_time = np.mean(self.step_times[-10:]) if self.step_times else 0
        steps_remaining = self.total_steps - self.step
        eta_seconds = steps_remaining * eta_step_time
        eta = str(timedelta(seconds=int(eta_seconds)))
        
        # Loss stats
        current_loss = self.losses[-1]
        avg_loss = np.mean(self.losses)
        current_ppl = self.perplexities[-1]
        tokens_per_sec = 0.0
        if self.last_batch_tokens and current_step_time > 0:
            tokens_per_sec = self.last_batch_tokens / current_step_time
        
        # Log step info
        logger.info(
            f"[train][{self.backend}] "
            f"step={self.step}/{self.total_steps} "
            f"loss={current_loss:.4f} "
            f"avg_loss={avg_loss:.4f} "
            f"val_loss={self._format_optional(self.last_val_loss, 4)} "
            f"val_ppl={self._format_optional(self.last_val_ppl, 2)} "
            f"lr={self._format_lr(self.last_lr)} "
            f"ppl={current_ppl:.2f} "
            f"grad_norm={self._format_optional(self.last_grad_norm, 4)} "
            f"elapsed={self._format_duration(elapsed)} "
            f"eta={self._format_duration(eta_seconds)} "
            f"step_ms={current_step_time * 1000:.2f} "
            f"avg_step_ms={avg_step_time * 1000:.2f} "
            f"tok/s={tokens_per_sec:.1f} "
            f"pool_used_mb={self._format_optional(pool_used_mb, 1)} "
            f"pool_total_mb={self._format_optional(pool_total_mb, 1)} "
            f"device_used_mb={device_used_mb:.1f}"
        )
    
    def finalize(self):
        """Log final training statistics."""
        total_time = time.time() - self.start_time
        
        logger.info("="*80)
        logger.info("TRAINING COMPLETE - FINAL STATISTICS")
        logger.info("="*80)
        
        if self.losses:
            logger.info(f"\nLoss Statistics:")
            logger.info(f"  Initial loss:     {self.losses[0]:.6f}")
            logger.info(f"  Final loss:       {self.losses[-1]:.6f}")
            logger.info(f"  Min loss:         {min(self.losses):.6f}")
            logger.info(f"  Max loss:         {max(self.losses):.6f}")
            logger.info(f"  Mean loss:        {np.mean(self.losses):.6f}")
            logger.info(f"  Loss improvement: {(self.losses[0] - self.losses[-1]):.6f} ({100*(self.losses[0] - self.losses[-1])/self.losses[0]:.1f}%)")
            
            logger.info(f"\nPerplexity Statistics:")
            logger.info(f"  Initial PPL:      {self.perplexities[0]:.2f}")
            logger.info(f"  Final PPL:        {self.perplexities[-1]:.2f}")
            logger.info(f"  Best PPL:         {min(self.perplexities):.2f}")
            logger.info(f"  Mean PPL:         {np.mean(self.perplexities):.2f}")

        if self.last_val_loss is not None:
            logger.info(f"\nValidation Stats:")
            logger.info(f"  Last val loss:    {self.last_val_loss:.6f}")
            logger.info(f"  Last val PPL:     {self.last_val_ppl:.2f}")
        
        if self.step_times:
            logger.info(f"\nTiming Statistics:")
            logger.info(f"  Total time:       {total_time:.1f}s ({str(timedelta(seconds=int(total_time)))})")
            logger.info(f"  Avg step time:    {np.mean(self.step_times)*1000:.1f}ms")
            logger.info(f"  Min step time:    {min(self.step_times)*1000:.1f}ms")
            logger.info(f"  Max step time:    {max(self.step_times)*1000:.1f}ms")
            logger.info(f"  Steps/sec:        {self.total_steps/total_time:.2f}")
        
        # Final GPU memory
        free_mem, total_mem = cuda.mem_get_info()
        used_mem = self.initial_total_mem - free_mem
        used_mem_mb = used_mem / 1024**2
        logger.info(f"\nGPU Memory:")
        logger.info(f"  Peak usage:       {used_mem_mb:.0f}MB")
        logger.info(f"  Current usage:    {used_mem_mb:.0f}MB")
        
        logger.info("="*80)


def estimate_vram_usage(vocab_size, embedding_dim, num_heads, num_layers, batch_size, seq_len):
    """Estimate VRAM required for model and training.
    
    Returns: (model_vram_mb, training_vram_mb, total_vram_mb)
    """
    # Model parameters (float32 = 4 bytes per param)
    
    # Embeddings: (vocab_size + seq_len) * embedding_dim
    embed_params = (vocab_size + seq_len) * embedding_dim
    
    # Transformer blocks: per block ~3*embedding_dim*4*embedding_dim parameters
    # (attention projections + FFN)
    head_dim = embedding_dim // num_heads
    attn_params_per_layer = (
        4 * embedding_dim * embedding_dim +  # QKV + output projections
        2 * embedding_dim  # layer norms
    )
    ffn_params_per_layer = (
        embedding_dim * (embedding_dim * 4) +  # up proj
        (embedding_dim * 4) * embedding_dim +  # down proj
        2 * embedding_dim  # layer norm
    )
    block_params = num_layers * (attn_params_per_layer + ffn_params_per_layer)
    
    # Output head
    output_params = embedding_dim * vocab_size + vocab_size
    
    total_params = embed_params + block_params + output_params
    model_vram_mb = (total_params * 4) / (1024**2)
    
    # Training overhead (activations, gradients, optimizer state)
    # Roughly: activations ~2x params, gradients ~1x params, optimizer state ~2x params
    # Total ~5x for safety
    training_overhead_mb = model_vram_mb * 5
    
    # Data: batch_size * seq_len * 2 (input + target) * 4 bytes
    data_mb = (batch_size * seq_len * 2 * 4) / (1024**2)
    
    total_vram_mb = model_vram_mb + training_overhead_mb + data_mb
    
    return model_vram_mb, training_overhead_mb, total_vram_mb


def validate_model_config(vocab_size, embedding_dim, num_heads, num_layers, batch_size, seq_len, available_vram_mb=800):
    """Validate model config fits in available VRAM.
    
    Returns: (is_valid, estimated_vram_mb, warning_msg)
    """
    model_vram, training_vram, total_vram = estimate_vram_usage(
        vocab_size, embedding_dim, num_heads, num_layers, batch_size, seq_len
    )
    
    is_valid = total_vram <= available_vram_mb
    warning_msg = None
    
    if total_vram > available_vram_mb:
        warning_msg = (
            f"Model too large for available VRAM!\n"
            f"  Estimated VRAM needed: {total_vram:.0f}MB\n"
            f"  Available VRAM: {available_vram_mb:.0f}MB\n"
            f"  Reduction: {(total_vram - available_vram_mb):.0f}MB needed\n"
            f"\n  Suggestions:\n"
            f"    - Use Tiny model (32D) instead of Medium (128D)\n"
            f"    - Reduce sequence length from {seq_len} to 32\n"
            f"    - Reduce corpus docs to 100-200\n"
        )
    elif total_vram > available_vram_mb * 0.8:
        warning_msg = (
            f"WARNING: Model will use ~{total_vram:.0f}MB ({100*total_vram/available_vram_mb:.0f}% of VRAM)\n"
            f"  This may be tight on GT730. Consider using Tiny or Small model.\n"
        )
    
    return is_valid, total_vram, warning_msg
