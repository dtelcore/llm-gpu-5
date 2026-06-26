# Quick Start: Running KepleGPT Tests

Fast setup to run your test suite.

## 1. Navigate to Project
```bash
cd "c:\dev\llm gpu 5"
```

## 2. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

## 3. Install Pytest (if needed)
```bash
pip install pytest
```

## 4. Run All Tests
```bash
pytest test/
```

## 5. Quick Test Runs

### Fast Tests Only (skip slow tests)
```bash
pytest test/ -m "not slow"
```

### GPU Tests Only
```bash
pytest test/ -m gpu
```

### CPU Tests Only (no GPU needed)
```bash
pytest test/ -m "not gpu"
```

### Specific Module
```bash
pytest test/test_tokenizer/        # Tokenizer tests only
pytest test/test_setup/             # Setup tests only
pytest test/test_core/              # Core (kernels/ops) tests
```

### Verbose Output
```bash
pytest test/ -v
```

### Stop on First Failure
```bash
pytest test/ -x
```

## Expected Output

```
test/test_tokenizer/test_tokenizer.py::TestTokenizerInitialization::test_tokenizer_from_corpus PASSED
test/test_setup/test_setup.py::TestModelConfigBuilder::test_config_builder_initialization PASSED
test/test_core/test_loss.py::TestSoftmaxCrossEntropyLoss::test_loss_initialization PASSED
...

================================== 510 passed in 2.35s ===================================
```

## Test Counts by Category

- ✅ Core Tests: 150+ (CUDA kernels + operators + loss)
- ✅ Model Tests: 100+ (architecture + checkpointing)
- ✅ Tokenizer Tests: 80+ (encoding/decoding)
- ✅ Setup Tests: 100+ (configuration + datasets)
- ✅ Integration Tests: 60+ (cross-module workflows)
- ✅ E2E Tests: 20+ (complete system)

**Total: 510+ tests**

## Common Issues

### ImportError: No module named 'pytest'
```bash
pip install pytest
```

### CUDA Error
```bash
# Run CPU-only tests
pytest test/ -m "not gpu"
```

### Test Timeout
- Slow tests may take time
- Use `-m "not slow"` to skip them
- Or increase timeout with `--timeout=600`

## Next Steps

For more details, see [test/README.md](README.md)
