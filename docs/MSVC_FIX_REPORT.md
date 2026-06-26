# MSVC and CUDA Compatibility Report

Last updated: 2026-05-24

Current state: the compiler/runtime fixes remain relevant, but the main model-quality issue was the data path, now repaired in the shared corpus utilities and training scripts.

## Current approach in code

- env_config.py loads vcvarsall for MSVC 14.29 toolset into the current process environment.
- env_config.py prepends MSVC and CUDA paths so compatible toolchain resolution happens first.
- core/ops.py and related CUDA compilation paths rely on this environment setup before kernel compilation.

## Why this matters

CUDA 10.1 and GT730-era workflows are sensitive to modern MSVC header/toolset drift. Keeping a deterministic toolchain path order avoids many nvcc+cl compatibility failures.

## Operational guidance

- Import env_config before PyCUDA-heavy modules.
- Use the project venv consistently.
- If kernel compilation starts failing after tool updates, re-check the active VS toolset path and CUDA path ordering.
