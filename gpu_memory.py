"""GPU memory allocation helpers for pooled PyCUDA runs."""

import pycuda.driver as cuda
from pycuda import tools

from logging_config import logger


_ORIGINAL_MEM_ALLOC = cuda.mem_alloc
_MEMORY_POOL = None


def install_global_memory_pool():
    """Route future `cuda.mem_alloc` calls through a shared PyCUDA memory pool."""
    global _MEMORY_POOL

    if _MEMORY_POOL is None:
        _MEMORY_POOL = tools.DeviceMemoryPool()
        cuda.mem_alloc = _MEMORY_POOL.allocate
        logger.info("[OK] PyCUDA pooled allocator enabled")

    return _MEMORY_POOL


def get_memory_pool_stats_mb():
    """Return active and managed pool bytes in MB when pooling is enabled."""
    if _MEMORY_POOL is None:
        return None, None

    return _MEMORY_POOL.active_bytes / 1024**2, _MEMORY_POOL.managed_bytes / 1024**2


def free_held_pool_blocks():
    """Release held blocks back to the CUDA driver after a run."""
    if _MEMORY_POOL is not None:
        _MEMORY_POOL.free_held()