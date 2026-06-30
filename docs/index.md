# Documentation Index

Last updated: 2026-06-30

Current state: GPU-resident FeedForward backward and Phase 3 Multi-Head Attention CUDA kernels (`core/mha_kernels.py`, `core/mha_ops.py`) have landed behind a three-layer NumPy golden-model correctness gate (`test_mha_golden_model.py`). Training/generation flow still uses sampled batch windows, newline-preserving corpus loading, and shared generation probes for tokenizer roundtrip, greedy decode, memorization-prefix, and first-step logits checks. Tokenizer/corpus building now reports live `tqdm` progress.

## Start here

1. GETTING_STARTED.md
2. QUICK_REFERENCE.md
3. TRAINING_WORKFLOW.md

## By topic

- Project overview: readme.md
- Architecture: ARCHITECTURE_GUIDE.md
- Training entry points: TRAINING_MODES.md
- Dataset setup: FINEWEB_SETUP.md
- Logging system: LOGGING.md
- Log inspection commands: LOGS_GUIDE.md
- CUDA and MSVC background: MSVC_FIX_REPORT.md
- Context scaling notes: SCALING_T64_CHANGES.md
- Progress and history: changelog.md, stepchangelog.md
- Current backlog: todo.md, todos-all.md
- Completion snapshot: PROJECT_COMPLETION.md
- GPU kernel internals: ARCHITECTURE_GUIDE.md (FFN backward, MHA kernels, golden-model test harnesses)
- Publishing changes: `gitpush.py` (repo root) runs status -> add -> commit -> push to `origin/main`
