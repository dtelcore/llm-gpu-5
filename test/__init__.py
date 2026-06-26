"""
KepleGPT Test Suite

Comprehensive testing framework for custom CUDA-based GPT transformer
on NVIDIA GeForce GT 730 (Kepler architecture).

Test Organization:
- test_core/: Low-level CUDA kernels and operators
- test_model/: Model architecture and components
- test_tokenizer/: Text tokenization and encoding
- test_setup/: Configuration and setup modules
- test_integration/: Cross-module integration tests
- test_end_to_end/: Complete system workflows

Usage:
    pytest test/                          # Run all tests
    pytest test/ -m gpu                   # Only GPU tests
    pytest test/ -m "not slow"            # Exclude slow tests
    pytest test/test_core/               # Only core tests
    pytest test/ -v                       # Verbose output
    pytest test/ --tb=short              # Short traceback format
"""

__version__ = "1.0.0"
__author__ = "KepleGPT Test Suite"
