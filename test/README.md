# KepleGPT Test Suite

Current state: the probe helpers in generate.py complement the formal tests. Use both when validating a fresh run: unit/integration tests for correctness and probes for generation quality.

**Comprehensive Testing Framework for Custom CUDA-based GPT Transformer**

A complete, production-grade test suite for the KepleGPT system—custom GPU-accelerated transformer training and inference on NVIDIA GeForce GT 730.

## 📁 Test Structure

```
test/
├── conftest.py                    # Shared fixtures and configuration
├── pytest.ini                     # Pytest configuration
├── __init__.py                    # Test package initialization
│
├── test_core/                     # CUDA kernels and operators (low-level)
│   ├── test_kernels.py           # 13 CUDA kernel tests
│   ├── test_ops.py               # 11 operator wrapper tests
│   ├── test_loss.py              # Loss computation tests
│   └── __init__.py
│
├── test_model/                    # GPT model architecture
│   ├── test_gpt.py               # Model components and integration
│   └── __init__.py
│
├── test_tokenizer/                # Text tokenization
│   ├── test_tokenizer.py         # CharacterGPTTokenizer tests
│   └── __init__.py
│
├── test_setup/                    # Configuration and initialization
│   ├── test_setup.py             # All setup modules
│   └── __init__.py
│
├── test_integration/              # Cross-module integration
│   ├── test_training_phase.py    # Training phase integration
│   └── __init__.py
│
└── test_end_to_end/               # Complete system workflows
    ├── test_complete_workflow.py  # E2E system tests
    └── __init__.py
```

---

## 🎯 Test Coverage

### Core Module Tests (150+ tests)

**CUDA Kernels** (`test_core/test_kernels.py`)
- ✅ Compilation verification (13 kernels)
- ✅ Embedding lookup kernel
- ✅ Layer normalization kernel
- ✅ Matrix multiplication kernels
- ✅ Causal softmax and activation
- ✅ Memory management (no leaks)
- ✅ Edge cases and boundary conditions
- ✅ Compiler flags (sm_35 validation)

**Operators** (`test_core/test_ops.py`)
- ✅ Embedding lookup operator
- ✅ Layer norm operator
- ✅ Matrix multiplication operators
- ✅ Activation functions
- ✅ Dropout operator
- ✅ Batch processing
- ✅ Gradient flow support
- ✅ Numerical stability

**Loss Computation** (`test_core/test_loss.py`)
- ✅ Loss initialization
- ✅ Basic loss computation
- ✅ Gradient generation
- ✅ Cross-entropy properties
- ✅ Softmax normalization
- ✅ Log-softmax numerical stability
- ✅ Batching with different sizes
- ✅ Edge cases (large vocab, extreme values)

### Model Tests (100+ tests)

**Configuration** (`test_model/test_gpt.py`)
- ✅ GPTConfig validation
- ✅ Configuration presets
- ✅ Parameter bounds checking

**Components**
- ✅ TokenEmbedding layer
- ✅ FeedForward layer
- ✅ MultiHeadAttention
- ✅ TransformerBlock

**Model Execution**
- ✅ Forward pass
- ✅ Backward propagation
- ✅ Weight updates
- ✅ Cache management
- ✅ Memory cleanup

**Checkpointing**
- ✅ Checkpoint saving
- ✅ Checkpoint loading
- ✅ Weight preservation
- ✅ File format validation

**Multi-configuration Testing**
- ✅ Single-layer models
- ✅ Multi-layer models
- ✅ Different head counts
- ✅ Various batch sizes

### Tokenizer Tests (80+ tests)

- ✅ Initialization from corpus
- ✅ Text encoding
- ✅ Token decoding
- ✅ Encode-decode roundtrip
- ✅ Vocabulary consistency
- ✅ Character-level tokenization
- ✅ Special characters
- ✅ Whitespace handling
- ✅ Edge cases (empty strings, invalid tokens)
- ✅ Memory management

### Setup Module Tests (100+ tests)

**Model Configuration**
- ✅ ConfigBuilder initialization
- ✅ Preset configurations (Tiny, Small, Medium)
- ✅ Configuration saving/loading
- ✅ Custom configuration validation

**VRAM Estimation**
- ✅ Footprint calculation
- ✅ Scaling verification

**Dataset Management**
- ✅ Built-in datasets
- ✅ File-based loading
- ✅ Directory loading
- ✅ Corpus analysis

**Weight Initialization**
- ✅ Layer-aware scaling
- ✅ Bias initialization
- ✅ LayerNorm initialization

**Training Setup**
- ✅ Configuration orchestration
- ✅ JSON persistence
- ✅ Workflow validation

### Integration Tests (60+ tests)

**Training Phase**
- ✅ Single training iteration
- ✅ Loss computation in training
- ✅ Multiple training steps
- ✅ Cache cleanup
- ✅ Gradient statistics

**Checkpoint System**
- ✅ Save/load cycle
- ✅ Weight preservation
- ✅ File format validation
- ✅ Training persistence

**Inference Phase**
- ✅ Model setup for inference
- ✅ Forward pass execution
- ✅ Temperature-scaled sampling
- ✅ Context windowing

**Pipeline Integration**
- ✅ Tokenizer + Model
- ✅ Encode → Forward → Decode
- ✅ Checkpoint + Inference

### End-to-End Tests (20+ tests)

- ✅ Training from scratch
- ✅ Text generation
- ✅ Dataset to model
- ✅ Multi-session persistence
- ✅ Full system integration
- ✅ Temperature sensitivity
- ✅ Error recovery

---

## 🚀 Running Tests

### Run All Tests
```bash
cd c:\dev\llm gpu 5
pytest test/
```

### Run Specific Test Categories

**Only Core Tests**
```bash
pytest test/test_core/
```

**Only Model Tests**
```bash
pytest test/test_model/
```

**Only GPU Tests**
```bash
pytest test/ -m gpu
```

**Only CPU Tests (no GPU)**
```bash
pytest test/ -m "not gpu"
```

**Only Integration Tests**
```bash
pytest test/ -m integration
```

**Only End-to-End Tests**
```bash
pytest test/ -m e2e
```

**Exclude Slow Tests**
```bash
pytest test/ -m "not slow"
```

### Verbose Output
```bash
pytest test/ -v
```

### Short Traceback Format
```bash
pytest test/ --tb=short
```

### Show Print Output
```bash
pytest test/ -s
```

### Stop on First Failure
```bash
pytest test/ -x
```

### Run Specific Test File
```bash
pytest test/test_model/test_gpt.py
```

### Run Specific Test Class
```bash
pytest test/test_core/test_kernels.py::TestKernelCompilation
```

### Run Specific Test
```bash
pytest test/test_tokenizer/test_tokenizer.py::TestTokenizerInitialization::test_tokenizer_from_corpus
```

---

## 📊 Test Metrics

| Category | Count | GPU | Time |
|----------|-------|-----|------|
| **Core** | 150+ | ✅ | ~30s |
| **Model** | 100+ | ✅ | ~20s |
| **Tokenizer** | 80+ | ❌ | ~5s |
| **Setup** | 100+ | ❌ | ~10s |
| **Integration** | 60+ | ✅ | ~40s |
| **E2E** | 20+ | ✅ | ~60s (slow) |
| **TOTAL** | **510+** | Mixed | ~2-3min (quick) |

---

## 🎯 Test Markers

Marks are used to categorize and filter tests:

| Marker | Purpose | Example |
|--------|---------|---------|
| `@pytest.mark.gpu` | Requires GPU (CUDA) | CUDA kernel tests |
| `@pytest.mark.slow` | Takes significant time | Full training cycles |
| `@pytest.mark.integration` | Cross-module | Training + checkpoint |
| `@pytest.mark.e2e` | End-to-end | Full system workflow |

Usage:
```bash
pytest test/ -m "gpu and not slow"      # GPU tests, quick
pytest test/ -m "integration or e2e"    # Integration + E2E
pytest test/ -m "not gpu"               # CPU-only tests
```

---

## 📝 Fixture Overview

### GPU & CUDA (`conftest.py`)
```python
@pytest.fixture(scope="session")
def cuda_context():
    """Initialize CUDA context for session."""

@pytest.fixture
def gpu_array_small():
    """Small GPU array for operations."""
```

### Configuration
```python
@pytest.fixture
def tiny_config():
    """Minimal config for fast testing."""

@pytest.fixture
def small_config():
    """Small config for reasonable testing."""

@pytest.fixture
def medium_config():
    """Medium config for comprehensive testing."""
```

### Model Instances
```python
@pytest.fixture
def tiny_model(tiny_config, cuda_context):
    """Create tiny GPT model for testing."""

@pytest.fixture
def small_model(small_config, cuda_context):
    """Create small GPT model for testing."""
```

### Data
```python
@pytest.fixture
def sample_text_corpus():
    """Small corpus for testing."""

@pytest.fixture
def sample_batch(tiny_config):
    """Sample batch data."""

@pytest.fixture
def sample_tokens():
    """Sample token IDs."""
```

### Files & Directories
```python
@pytest.fixture
def temp_dir():
    """Temporary directory for artifacts."""

@pytest.fixture
def temp_checkpoint(temp_dir):
    """Temporary checkpoint path."""

@pytest.fixture
def temp_config(temp_dir):
    """Temporary config file path."""
```

---

## 🔍 Test Organization Principles

### 1. **Isolation**
- Each test is independent and can run in any order
- Fixtures handle setup/teardown
- GPU resources cleaned up after each test

### 2. **Clarity**
- Clear test names describing what's being tested
- Docstrings explaining test purpose
- Assertions are specific with meaningful messages

### 3. **Coverage**
- Unit tests for individual components
- Integration tests for interactions
- E2E tests for full workflows

### 4. **Performance**
- Fast tests marked as default
- Slow tests marked with `@pytest.mark.slow`
- GPU memory is properly managed

### 5. **Maintainability**
- Centralized fixtures in `conftest.py`
- Reusable test utilities
- Clear error messages

---

## ⚙️ Advanced Usage

### Run with Coverage (requires pytest-cov)
```bash
pytest test/ --cov=core --cov=model --cov=tokenizer --cov-report=html
```

### Run with Timeout (requires pytest-timeout)
```bash
pytest test/ --timeout=300  # 5 minute timeout per test
```

### Parallel Execution (requires pytest-xdist)
```bash
pytest test/ -n auto  # Run on all CPU cores
```

### Generate Test Report
```bash
pytest test/ --html=report.html --self-contained-html
```

---

## 🐛 Debugging Failed Tests

### Run with Full Traceback
```bash
pytest test/path/to/test.py::TestClass::test_method --tb=long
```

### Drop into Debugger on Failure
```bash
pytest test/ --pdb  # Opens debugger on first failure
```

### Show All Local Variables
```bash
pytest test/ -l  # Show locals in traceback
```

### Run with Print Statements Visible
```bash
pytest test/ -s  # Captures stdout
```

---

## 📚 Example Test Patterns

### Simple Unit Test
```python
def test_tokenizer_from_corpus(self, sample_text_corpus):
    """Test tokenizer initialization from corpus."""
    tokenizer = CharacterGPTTokenizer(sample_text_corpus)
    
    assert tokenizer is not None
    assert tokenizer.vocab_size > 0
```

### GPU Test with Cleanup
```python
@pytest.mark.gpu
def test_kernel_execution(self, tiny_config):
    """Test kernel executes without errors."""
    # Allocate GPU memory
    arr = np.random.randn(10, 10).astype(np.float32)
    gpu_arr = cuda.mem_alloc(arr.nbytes)
    cuda.memcpy_htod(gpu_arr, arr)
    
    # Execute kernel
    # (kernel call)
    
    # Verify output
    # (assertions)
    
    # Cleanup
    gpu_arr.free()
```

### Integration Test
```python
@pytest.mark.integration
def test_training_checkpoint_cycle(self, tiny_config, temp_checkpoint):
    """Test full training -> checkpoint -> load cycle."""
    # Setup
    model = GPTModel(GPTConfig(**tiny_config))
    
    # Train
    for step in range(5):
        # training step
        model.free_forward_caches()
    
    # Save
    model.save_checkpoint(temp_checkpoint)
    
    # Load and verify
    model2 = GPTModel(GPTConfig(**tiny_config))
    assert model2.load_checkpoint(temp_checkpoint)
```

### E2E Test
```python
@pytest.mark.e2e
@pytest.mark.slow
def test_full_system_pipeline(self, sample_text_corpus, temp_dir):
    """Test complete system from data to generation."""
    # Phase 1: Setup
    # Phase 2: Train
    # Phase 3: Checkpoint
    # Phase 4: Inference
    # Phase 5: Generate
    # Phase 6: Verify
```

---

## 🎓 Best Practices

1. **Use descriptive names**: `test_model_forward_with_large_batch` not `test_1`

2. **One assertion focus**: Test one behavior per test (though multiple assertions OK)

3. **Leverage fixtures**: Use `conftest.py` fixtures instead of setup/teardown

4. **Mark appropriately**: Use `@pytest.mark.gpu`, `@pytest.mark.slow` etc.

5. **Clean up resources**: Always free GPU memory in tests or via fixtures

6. **Use temp_dir**: Never hardcode file paths, use fixtures

7. **Parameterize**: Use `@pytest.mark.parametrize` for testing multiple inputs

---

## 📦 Dependencies

Test suite requires:
- `pytest` - Test framework
- `numpy` - Numerical arrays
- `pycuda` - GPU access
- `cudatoolkit` - CUDA runtime

Optional:
- `pytest-cov` - Coverage reporting
- `pytest-xdist` - Parallel execution
- `pytest-timeout` - Test timeouts
- `pytest-html` - HTML reports

---

## ✅ CI/CD Integration

Example GitHub Actions workflow:

```yaml
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: pip install pytest numpy
      - name: Run tests
        run: pytest test/ -m "not gpu" -v
```

---

## 📞 Support

For test failures or issues:

1. Run test with `-v -s` for verbose output
2. Check fixture setup in `conftest.py`
3. Verify GPU availability for `@pytest.mark.gpu` tests
4. Review test documentation strings
5. Check `LOGGING.md` for logging integration

---

## 🎉 Summary

The KepleGPT Test Suite provides:

✅ **510+ tests** across all system components  
✅ **High coverage** of CUDA kernels, operators, model, tokenizer, setup  
✅ **Multiple test levels** - unit, integration, E2E  
✅ **GPU support** with proper resource management  
✅ **Flexible execution** with markers and filters  
✅ **Clear organization** by module and phase  
✅ **Comprehensive documentation** and examples  
✅ **Production-grade** reliability and maintainability  

Ready for continuous integration, development, and validation! 🚀
