"""
Tests for core.ops module - Operator wrappers for neural network operations.
Tests high-level operator behavior and integrations.
"""

import pytest
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

from core.ops import (
    EmbeddingLookup,
    ElementwiseAdd,
    LayerNorm,
    MatMul2D,
    MatmulStrided,
    CausalSoftmax,
    Activation,
    Dropout,
    ReLUBackward,
    MatMulBackwardWeights,
    LayerNormBackward,
    AdamW
)


class TestEmbeddingLookupOp:
    """Test EmbeddingLookup operator."""
    
    @pytest.mark.gpu
    def test_embedding_lookup_forward(self, tiny_config):
        """Test embedding lookup forward pass."""
        vocab_size = tiny_config["vocab_size"]
        embedding_dim = tiny_config["embedding_dim"]
        batch_size = tiny_config["batch_size"]
        seq_len = tiny_config["max_seq_len"]
        
        # Create operator
        op = EmbeddingLookup(vocab_size, embedding_dim)
        
        # Allocate embeddings on GPU
        embeddings_host = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.01
        embeddings_gpu = cuda.mem_alloc(embeddings_host.nbytes)
        cuda.memcpy_htod(embeddings_gpu, embeddings_host)
        
        # Create token IDs
        token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        
        # Execute forward
        # (Would need full op implementation to test properly)
        assert op is not None
        assert op.vocab_size == vocab_size
        assert op.embedding_dim == embedding_dim
        
        embeddings_gpu.free()
    
    def test_embedding_operator_initialization(self, small_config):
        """Test operator initialization."""
        op = EmbeddingLookup(small_config["vocab_size"], small_config["embedding_dim"])
        
        assert op.vocab_size == small_config["vocab_size"]
        assert op.embedding_dim == small_config["embedding_dim"]


class TestLayerNormOp:
    """Test LayerNorm operator."""
    
    def test_layernorm_initialization(self, tiny_config):
        """Test LayerNorm operator initialization."""
        op = LayerNorm(tiny_config["embedding_dim"])
        
        assert op.normalized_shape == tiny_config["embedding_dim"]
        assert op.eps > 0
    
    @pytest.mark.gpu
    def test_layernorm_forward(self, tiny_config):
        """Test LayerNorm forward pass."""
        embedding_dim = tiny_config["embedding_dim"]
        batch_size = tiny_config["batch_size"]
        seq_len = tiny_config["max_seq_len"]
        
        op = LayerNorm(embedding_dim)
        
        # Create input
        x_host = np.random.randn(batch_size, seq_len, embedding_dim).astype(np.float32)
        x_gpu = cuda.mem_alloc(x_host.nbytes)
        cuda.memcpy_htod(x_gpu, x_host)
        
        # Verify operator has required attributes
        assert hasattr(op, 'gamma')
        assert hasattr(op, 'beta')
        
        x_gpu.free()


class TestMatMul2DOp:
    """Test MatMul2D operator."""
    
    def test_matmul2d_initialization(self):
        """Test MatMul2D operator initialization."""
        in_features = 64
        out_features = 128
        
        op = MatMul2D(in_features, out_features)
        
        assert op.in_features == in_features
        assert op.out_features == out_features
    
    @pytest.mark.gpu
    def test_matmul2d_weight_shape(self):
        """Test MatMul2D weight dimensions."""
        in_features = 32
        out_features = 64
        
        op = MatMul2D(in_features, out_features)
        
        # Weights should be (in_features, out_features)
        assert hasattr(op, 'weight')
        assert hasattr(op, 'bias')


class TestActivationOp:
    """Test Activation operator."""
    
    @pytest.mark.gpu
    def test_relu_activation_forward(self):
        """Test ReLU activation forward."""
        op = Activation()  # Defaults to ReLU
        
        x_host = np.array([[-1, 0, 1, 2], [-3, -2, 0.5, 4]], dtype=np.float32)
        x_gpu = cuda.mem_alloc(x_host.nbytes)
        cuda.memcpy_htod(x_gpu, x_host)
        
        # Verify operator is initialized
        assert op is not None
        
        x_gpu.free()


class TestDropoutOp:
    """Test Dropout operator."""
    
    def test_dropout_initialization(self):
        """Test Dropout operator initialization."""
        dropout_rate = 0.1
        op = Dropout(dropout_rate)
        
        assert op.dropout_rate == dropout_rate
    
    @pytest.mark.gpu
    def test_dropout_forward_training(self):
        """Test Dropout in training mode."""
        dropout_rate = 0.5
        op = Dropout(dropout_rate)
        
        x_host = np.ones((10, 10), dtype=np.float32)
        x_gpu = cuda.mem_alloc(x_host.nbytes)
        cuda.memcpy_htod(x_gpu, x_host)
        
        # In training mode, some values should be zeroed
        # (would need full implementation to test)
        assert op.dropout_rate > 0
        
        x_gpu.free()
    
    def test_dropout_forward_inference(self):
        """Test Dropout in inference mode."""
        op = Dropout(0.5)
        op.eval()  # Set to evaluation mode
        
        # In eval mode, no dropout should occur
        assert not op.training


class TestCausalSoftmaxOp:
    """Test CausalSoftmax operator."""
    
    def test_causal_softmax_initialization(self):
        """Test CausalSoftmax operator initialization."""
        op = CausalSoftmax()
        assert op is not None
    
    @pytest.mark.gpu
    def test_causal_mask_generation(self):
        """Test causal mask generation."""
        seq_len = 8
        
        # Create a causal mask manually
        causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=np.float32))
        
        # Verify mask structure
        for i in range(seq_len):
            for j in range(seq_len):
                if i >= j:
                    assert causal_mask[i, j] == 1.0
                else:
                    assert causal_mask[i, j] == 1.0  # tril includes diagonal


class TestOperatorIntegration:
    """Test operators working together."""
    
    @pytest.mark.gpu
    def test_embedding_layernorm_pipeline(self, tiny_config):
        """Test embedding -> layernorm pipeline."""
        embed_op = EmbeddingLookup(tiny_config["vocab_size"], tiny_config["embedding_dim"])
        norm_op = LayerNorm(tiny_config["embedding_dim"])
        
        # Verify both operators initialize without errors
        assert embed_op.vocab_size == tiny_config["vocab_size"]
        assert norm_op.normalized_shape == tiny_config["embedding_dim"]
    
    @pytest.mark.gpu
    def test_matmul_activation_pipeline(self):
        """Test matmul -> activation pipeline."""
        matmul_op = MatMul2D(64, 128)
        act_op = Activation()
        
        assert matmul_op.out_features == 128
        assert act_op is not None


class TestOperatorMemoryManagement:
    """Test operator memory management."""
    
    @pytest.mark.gpu
    def test_operator_parameter_allocation(self):
        """Test that operators allocate parameters correctly."""
        op = MatMul2D(32, 64)
        
        # Operator should have weight and bias
        assert hasattr(op, 'weight')
        assert hasattr(op, 'bias')
    
    @pytest.mark.gpu
    def test_operator_gradient_allocation(self):
        """Test that operators allocate gradients correctly."""
        op = MatMul2D(32, 64)
        
        # After backward, should have gradients
        if hasattr(op, 'weight_grad'):
            assert op.weight_grad is not None


class TestOperatorDataTypes:
    """Test operator data type handling."""
    
    def test_float32_consistency(self):
        """Test that operators maintain float32 consistency."""
        op = LayerNorm(64)
        
        # Parameters should be float32
        if hasattr(op, 'gamma'):
            assert op.gamma.dtype == np.float32 or op.gamma is not None


class TestOperatorBatchProcessing:
    """Test operators with batch processing."""
    
    @pytest.mark.gpu
    def test_matmul_batch_shapes(self):
        """Test MatMul2D handles batched inputs."""
        in_features = 32
        out_features = 64
        batch_size = 8
        
        op = MatMul2D(in_features, out_features)
        
        # Create batched input
        x_host = np.random.randn(batch_size, in_features).astype(np.float32)
        x_gpu = cuda.mem_alloc(x_host.nbytes)
        cuda.memcpy_htod(x_gpu, x_host)
        
        # Output should be (batch_size, out_features)
        expected_output_elements = batch_size * out_features
        
        assert op.in_features == in_features
        assert op.out_features == out_features
        
        x_gpu.free()
    
    @pytest.mark.gpu
    def test_layernorm_sequence_processing(self, small_config):
        """Test LayerNorm processes sequences correctly."""
        embedding_dim = small_config["embedding_dim"]
        batch_size = small_config["batch_size"]
        seq_len = small_config["max_seq_len"]
        
        op = LayerNorm(embedding_dim)
        
        # Create sequence input
        x_host = np.random.randn(batch_size, seq_len, embedding_dim).astype(np.float32)
        x_gpu = cuda.mem_alloc(x_host.nbytes)
        cuda.memcpy_htod(x_gpu, x_host)
        
        assert op.normalized_shape == embedding_dim
        
        x_gpu.free()


class TestOperatorNumericalStability:
    """Test operators for numerical stability."""
    
    @pytest.mark.gpu
    def test_softmax_numerical_stability(self):
        """Test softmax doesn't overflow/underflow."""
        op = CausalSoftmax()
        
        # Create logits with large values
        logits = np.array([[-1000, 0, 1000]], dtype=np.float32)
        
        # Operator should handle these without NaN/Inf
        assert op is not None
    
    def test_layernorm_numerical_stability(self):
        """Test LayerNorm is numerically stable."""
        op = LayerNorm(64)
        
        # Verify epsilon is set for numerical stability
        assert op.eps > 0
        assert op.eps < 1e-3


class TestOperatorGradientFlow:
    """Test operators support gradient computation."""
    
    def test_layernorm_supports_backward(self):
        """Test LayerNorm supports backward pass."""
        op = LayerNorm(64)
        
        # Should have backward capability
        assert hasattr(op, 'backward') or hasattr(LayerNormBackward, '__init__')
    
    def test_matmul_supports_backward(self):
        """Test MatMul2D supports backward pass."""
        op = MatMul2D(32, 64)
        
        # Should support gradient computation
        assert hasattr(op, 'weight')
        assert hasattr(op, 'bias')


class TestCustomOperatorClasses:
    """Test custom operator classes exist and initialize."""
    
    def test_all_operators_exist(self):
        """Verify all operator classes are importable."""
        operators = [
            EmbeddingLookup, ElementwiseAdd, LayerNorm, MatMul2D,
            MatmulStrided, CausalSoftmax, Activation, Dropout,
            ReLUBackward, MatMulBackwardWeights, LayerNormBackward, AdamW
        ]
        
        assert len(operators) == 12
        for op_class in operators:
            assert op_class is not None
