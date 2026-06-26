"""
Tests for core.kernels module - CUDA kernel definitions.
Verifies that all 13 required CUDA kernels are properly defined.
"""

import pytest
from core.kernels import KERNELS


class TestKernelDefinitions:
    """Test that all required CUDA kernels are properly defined."""
    
    def test_kernels_dict_exists(self):
        """Test KERNELS dictionary is properly defined."""
        assert KERNELS is not None
        assert isinstance(KERNELS, dict)
        assert len(KERNELS) == 13, f"Expected 13 kernels, got {len(KERNELS)}"
    
    def test_all_required_kernels_present(self):
        """Test all required kernels are in KERNELS dict."""
        required_kernels = [
            "embedding_lookup_kernel",
            "elementwise_add_kernel",
            "layernorm_kernel",
            "matmul_2d_kernel",
            "matrix_multiply_strided_kernel",
            "causal_softmax_kernel",
            "activation_kernel",
            "dropout_forward_kernel",
            "dropout_backward_kernel",
            "relu_backward_kernel",
            "matmul_backward_weights_kernel",
            "layernorm_backward_kernel",
            "adamw_update_kernel",
        ]
        for kernel_name in required_kernels:
            assert kernel_name in KERNELS, f"Missing kernel: {kernel_name}"
            assert isinstance(KERNELS[kernel_name], str), f"Kernel {kernel_name} should be CUDA code string"
            assert len(KERNELS[kernel_name]) > 0, f"Kernel {kernel_name} is empty"
            assert "__global__" in KERNELS[kernel_name], f"Kernel {kernel_name} missing __global__"


@pytest.mark.gpu
class TestKernelCompilation:
    """Test that kernels compile successfully through ops module."""
    
    def test_kernels_compile_via_ops(self):
        """Test that kernels compile successfully when imported through ops."""
        from core.ops import _SHARED_GPU_MODULE
        assert _SHARED_GPU_MODULE is not None
