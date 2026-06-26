#!/usr/bin/env python3
"""Minimal test to verify CUDA kernel compilation and loading works."""

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import env_config

test_code = '''
extern "C" {
    __global__ void test_kernel(float *out) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        out[idx] = 42.0f;
    }
}
'''

try:
    print("Compiling minimal test kernel...")
    mod = SourceModule(test_code, options=[
        "-ccbin", env_config.MSVC_142_BIN,
        "-O3"
    ])
    print("✓ Test kernel compiled successfully")
    
    print("Loading kernel function...")
    kernel = mod.get_function('test_kernel')
    print("✓ Test kernel loaded successfully")
    
    print("\nTesting kernel execution...")
    output = np.zeros(10, dtype=np.float32)
    gpu_out = cuda.mem_alloc(output.nbytes)
    kernel(gpu_out, block=(10, 1, 1), grid=(1, 1))
    cuda.memcpy_dtoh(output, gpu_out)
    gpu_out.free()
    
    if output[0] == 42.0:
        print("✓ Kernel executed correctly, output[0] = 42.0")
    else:
        print(f"✗ Unexpected output: {output}")
        
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
