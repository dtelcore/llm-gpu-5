import os
import numpy as np

# 1. Enforce Path Dominance for your legacy GT 730 Environment
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin"
MSVC_142_BIN = r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64"

if hasattr(os, "add_dll_directory") and os.path.exists(CUDA_BIN):
    os.add_dll_directory(CUDA_BIN)
os.environ["PATH"] = f"{CUDA_BIN};{MSVC_142_BIN};" + os.environ["PATH"]

# Safely import PyCUDA hooks directly to satisfy Pylance type tracking
import pycuda.autoinit
from pycuda.driver import mem_alloc, memcpy_htod  # type: ignore

from tokenizer import CharacterGPTTokenizer  # type: ignore

# 2. Mock a list of documents to build the character vocab dictionary
dataset_docs = ["cuda", "kepler", "gt730", "gpu", "matrix"]
tokenizer = CharacterGPTTokenizer(dataset_docs)

print(f"=== Character Dictionary Initialized ===")
print(f"Vocabulary Size: {tokenizer.vocab_size} | BOS/PAD ID: {tokenizer.BOS_ID}\n")

# 3. Create a batch of text sequence inputs
text_batch = ["cuda", "gpu"]

# 4. CPU STAGING: Generate the fixed-shape 2D Matrix (Batch Size=2, Max Length=8)
# This uses Section 1 of your code to generate the clean NumPy layout
max_len = 8
cpu_matrix = tokenizer.encode_batch_gpu_aligned(text_batch, max_sequence_length=max_len)

print("=== CPU Staging Matrix (Ready for GPU) ===")
print(f"Shape: {cpu_matrix.shape} | DataType: {cpu_matrix.dtype}")
print(cpu_matrix)
print(f"Total Bytes to Transfer: {cpu_matrix.nbytes} bytes\n")

# 5. GPU TRANSLATION: Allocate VRAM and copy across PCIe bus to the GT 730
try:
    print("=== Pushing Matrix to NVIDIA GeForce GT 730 VRAM ===")
    
    # Allocate explicit memory blocks on the GPU
    gpu_vram_pointer = mem_alloc(cpu_matrix.nbytes)
    
    # Synchronously copy the structured array from Host RAM to Device VRAM
    memcpy_htod(gpu_vram_pointer, cpu_matrix)
    print(f"[SUCCESS] Array safely copied over PCIe bus to VRAM address: {int(gpu_vram_pointer)}")
    
    # --- Next Project Steps live here ---
    # In your upcoming core/model modules, you will pass 'gpu_vram_pointer' 
    # directly into custom CUDA SourceModule kernels (like an embedding lookup layer)
    # ------------------------------------

    # Free the allocation handle manually to clean up VRAM
    gpu_vram_pointer.free()
    print("[SUCCESS] VRAM allocation de-allocated cleanly.")

except Exception as e:
    print(f"[FAILURE] Hardware memory handshake failed: {e}")