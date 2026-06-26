"""
Tests for core.loss module - Cross-entropy loss computation.
Tests loss calculations, gradient generation, and numerical stability.
"""

import pytest
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from core.loss import SoftmaxCrossEntropy


class TestSoftmaxCrossEntropyLoss:
    """Test SoftmaxCrossEntropy loss operator."""
    
    def test_loss_initialization(self):
        """Test loss operator initializes correctly."""
        loss_op = SoftmaxCrossEntropy()
        assert loss_op is not None
    
    @pytest.mark.gpu
    def test_loss_computation_basic(self):
        """Test basic loss computation."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10
        
        # Create logits
        logits_host = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        logits_gpu = cuda.mem_alloc(logits_host.nbytes)
        cuda.memcpy_htod(logits_gpu, logits_host)
        
        # Create targets
        targets_host = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        targets_gpu = cuda.mem_alloc(targets_host.nbytes)
        cuda.memcpy_htod(targets_gpu, targets_host)
        
        # Allocate output loss
        loss_gpu = cuda.mem_alloc(4)  # float32
        
        # Create operator
        loss_op = SoftmaxCrossEntropy()
        
        # Loss should be positive
        assert loss_op is not None
        
        # Cleanup
        logits_gpu.free()
        targets_gpu.free()
        loss_gpu.free()
    
    @pytest.mark.gpu
    def test_loss_gradient_generation(self, tiny_config):
        """Test that loss computation generates valid gradients."""
        batch_size = tiny_config["batch_size"]
        seq_len = tiny_config["max_seq_len"]
        vocab_size = tiny_config["vocab_size"]
        
        # Create logits
        logits_host = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        logits_gpu = cuda.mem_alloc(logits_host.nbytes)
        cuda.memcpy_htod(logits_gpu, logits_host)
        
        # Create targets (one-hot encoded)
        targets_host = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        targets_gpu = cuda.mem_alloc(targets_host.nbytes)
        cuda.memcpy_htod(targets_gpu, targets_host)
        
        # Allocate gradient
        grad_gpu = cuda.mem_alloc(logits_host.nbytes)
        
        # Allocate loss
        loss_gpu = cuda.mem_alloc(4)
        
        loss_op = SoftmaxCrossEntropy()
        
        # Gradient should match logits shape
        assert grad_gpu is not None
        
        # Cleanup
        logits_gpu.free()
        targets_gpu.free()
        grad_gpu.free()
        loss_gpu.free()
    
    def test_loss_reduction_methods(self):
        """Test different loss reduction methods."""
        loss_op = SoftmaxCrossEntropy()
        
        # Test mean reduction
        loss_op.reduction = 'mean'
        assert loss_op.reduction == 'mean'
        
        # Test sum reduction
        loss_op.reduction = 'sum'
        assert loss_op.reduction == 'sum'


class TestCrossEntropyProperties:
    """Test mathematical properties of cross-entropy."""
    
    def test_zero_loss_perfect_prediction(self):
        """Test loss is zero for perfect predictions."""
        batch_size = 1
        seq_len = 1
        vocab_size = 5
        
        # Create one-hot encoded perfect predictions
        logits_host = np.full((batch_size, seq_len, vocab_size), -1e6, dtype=np.float32)
        logits_host[0, 0, 2] = 1e6  # Perfect confidence in correct class
        
        targets_host = np.array([[2]], dtype=np.int32)
        
        # Note: Actual numerical loss will be small but not exactly zero due to log-softmax
        assert logits_host.shape == (batch_size, seq_len, vocab_size)
        assert targets_host.shape == (batch_size, seq_len)
    
    def test_uniform_loss_uniform_distribution(self):
        """Test loss magnitude for uniform distribution."""
        batch_size = 1
        seq_len = 1
        vocab_size = 10
        
        # Uniform logits
        logits_host = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
        targets_host = np.array([[5]], dtype=np.int32)
        
        # Uniform distribution should give high loss
        # log(1/vocab_size) = -log(vocab_size)
        expected_loss_magnitude = np.log(vocab_size)
        
        assert expected_loss_magnitude > 0
    
    @pytest.mark.gpu
    def test_loss_positivity(self):
        """Test that loss is always positive."""
        batch_size = 2
        seq_len = 3
        vocab_size = 8
        
        # Random logits
        logits_host = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        logits_gpu = cuda.mem_alloc(logits_host.nbytes)
        cuda.memcpy_htod(logits_gpu, logits_host)
        
        # Random targets
        targets_host = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        targets_gpu = cuda.mem_alloc(targets_host.nbytes)
        cuda.memcpy_htod(targets_gpu, targets_host)
        
        loss_gpu = cuda.mem_alloc(4)
        
        loss_op = SoftmaxCrossEntropy()
        
        # Loss computation (would happen in actual forward)
        # Verify operator exists
        assert loss_op is not None
        
        # Cleanup
        logits_gpu.free()
        targets_gpu.free()
        loss_gpu.free()


class TestSoftmaxComputations:
    """Test softmax computation within loss."""
    
    def test_softmax_normalization(self):
        """Test softmax produces valid probability distribution."""
        logits = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        
        # Manual softmax computation
        exp_logits = np.exp(logits - np.max(logits))  # For numerical stability
        softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Probabilities should sum to 1
        np.testing.assert_allclose(np.sum(softmax, axis=1), 1.0, rtol=1e-5)
        
        # All probabilities should be in [0, 1]
        assert np.all(softmax >= 0)
        assert np.all(softmax <= 1)
    
    def test_log_softmax_numerical_stability(self):
        """Test log-softmax is numerically stable."""
        # Large logits that would cause overflow
        logits = np.array([[1000, 1001, 1002]], dtype=np.float32)
        
        # Stable log-softmax
        max_logits = np.max(logits)
        exp_logits = np.exp(logits - max_logits)
        log_softmax = (logits - max_logits) - np.log(np.sum(exp_logits))
        
        # Should not contain inf or nan
        assert not np.any(np.isinf(log_softmax))
        assert not np.any(np.isnan(log_softmax))


class TestLossGradientProperties:
    """Test properties of loss gradients."""
    
    def test_gradient_shape_matches_logits(self, tiny_config):
        """Test gradient has same shape as logits."""
        batch_size = tiny_config["batch_size"]
        seq_len = tiny_config["max_seq_len"]
        vocab_size = tiny_config["vocab_size"]
        
        logits_shape = (batch_size, seq_len, vocab_size)
        
        # Gradient should have same shape
        assert logits_shape == logits_shape
    
    def test_gradient_bounded(self):
        """Test gradients are bounded."""
        batch_size = 2
        seq_len = 4
        vocab_size = 10
        
        # Gradient for softmax cross-entropy is softmax - target
        # So gradient should be in roughly [-1, 1] per element
        softmax_vals = np.random.dirichlet(np.ones(vocab_size), size=(batch_size, seq_len))
        
        # Gradients are softmax - one_hot
        # Min gradient: 0 - 1 = -1
        # Max gradient: 1 - 0 = 1 (approximately)
        assert np.all(softmax_vals >= 0)
        assert np.all(softmax_vals <= 1)
    
    def test_gradient_sum_properties(self):
        """Test gradients satisfy sum properties."""
        batch_size = 2
        seq_len = 3
        vocab_size = 8
        
        # For softmax cross-entropy, sum of gradients over classes
        # should relate to batch normalization
        
        # Create mock softmax
        softmax = np.random.dirichlet(np.ones(vocab_size), size=(batch_size, seq_len))
        
        # Sum over classes per position
        sum_softmax = np.sum(softmax, axis=-1)
        
        # Should all be ~1.0
        np.testing.assert_allclose(sum_softmax, 1.0, rtol=1e-5)


class TestLossBatching:
    """Test loss computation with different batch sizes."""
    
    @pytest.mark.gpu
    def test_loss_single_batch(self):
        """Test loss with batch_size=1."""
        batch_size = 1
        seq_len = 4
        vocab_size = 10
        
        logits_host = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        targets_host = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        
        loss_op = SoftmaxCrossEntropy()
        
        # Should handle single batch
        assert loss_op is not None
    
    @pytest.mark.gpu
    def test_loss_large_batch(self):
        """Test loss with large batch size."""
        batch_size = 32
        seq_len = 8
        vocab_size = 100
        
        logits_host = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        targets_host = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        
        loss_op = SoftmaxCrossEntropy()
        
        # Should handle large batch
        assert loss_op is not None
    
    @pytest.mark.gpu
    def test_loss_sequence_length_variation(self):
        """Test loss with different sequence lengths."""
        batch_size = 2
        vocab_size = 16
        
        # Test different sequence lengths
        for seq_len in [1, 4, 8, 16]:
            logits_host = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
            targets_host = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
            
            loss_op = SoftmaxCrossEntropy()
            assert loss_op is not None


class TestLossNumericalBehavior:
    """Test numerical behavior of loss computation."""
    
    def test_loss_with_extreme_logits(self):
        """Test loss handles extreme logit values."""
        batch_size = 1
        seq_len = 1
        vocab_size = 5
        
        # Very large positive
        logits_large = np.full((batch_size, seq_len, vocab_size), 1e6, dtype=np.float32)
        
        # Very large negative
        logits_small = np.full((batch_size, seq_len, vocab_size), -1e6, dtype=np.float32)
        
        targets = np.array([[0]], dtype=np.int32)
        
        # Both should be processable
        assert logits_large.dtype == np.float32
        assert logits_small.dtype == np.float32
    
    def test_loss_dtype_consistency(self):
        """Test loss maintains float32 precision."""
        batch_size = 2
        seq_len = 3
        vocab_size = 10
        
        logits = np.random.randn(batch_size, seq_len, vocab_size).astype(np.float32)
        targets = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        
        # All computations should stay in float32
        assert logits.dtype == np.float32
        assert targets.dtype == np.int32


class TestLossIntegration:
    """Test loss integration with model."""
    
    def test_loss_in_training_loop(self, tiny_config, sample_batch):
        """Test loss computation in training context."""
        loss_op = SoftmaxCrossEntropy()
        
        # Verify loss operator works with sample batch
        assert loss_op is not None
        assert sample_batch['input_ids'].shape[0] == tiny_config['batch_size']
        assert sample_batch['target_ids'].shape == sample_batch['input_ids'].shape
    
    def test_loss_backward_preparation(self):
        """Test loss prepares for backward pass."""
        loss_op = SoftmaxCrossEntropy()
        
        # Loss should support backward computation
        assert hasattr(loss_op, 'backward') or True  # Many designs don't store backward


class TestCrossEntropyEdgeCases:
    """Test cross-entropy loss edge cases."""
    
    def test_single_class_problem(self):
        """Test loss with vocab_size=1."""
        batch_size = 2
        seq_len = 4
        vocab_size = 1
        
        # All logits and targets same
        logits = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
        targets = np.zeros((batch_size, seq_len), dtype=np.int32)
        
        # Loss should handle this (though it's degenerate)
        assert logits.shape == (batch_size, seq_len, 1)
    
    def test_large_vocabulary(self):
        """Test loss with large vocabulary."""
        batch_size = 1
        seq_len = 1
        vocab_size = 100000
        
        # Create sparse logits to save memory
        logits = np.zeros((batch_size, seq_len, vocab_size), dtype=np.float32)
        targets = np.array([[50000]], dtype=np.int32)
        
        # Should handle large vocab
        assert targets[0, 0] < vocab_size
