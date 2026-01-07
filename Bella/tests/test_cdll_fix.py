import os
import sys
import ctypes
import site

# Fix for CUDA/cuDNN library paths
packages = site.getsitepackages()
site_packages = packages[0]
added_paths = []
for lib in ['cudnn', 'cublas', 'cuda_runtime']:
    lib_path = os.path.join(site_packages, "nvidia", lib, "lib")
    if os.path.isdir(lib_path):
        added_paths.append(lib_path)

print(f"DEBUG: Added paths: {added_paths}")

# Try to load specifically the good ones
good_libs = []
for path in added_paths:
    for f in os.listdir(path):
        if f.startswith("libcudnn_cnn.so.9") or f.startswith("libcudnn.so.9"):
            lib_full_path = os.path.join(path, f)
            print(f"DEBUG: Attempting to preload {lib_full_path}")
            try:
                ctypes.CDLL(lib_full_path, mode=ctypes.RTLD_GLOBAL)
                print(f"DEBUG: Preloaded {f} successfully")
            except Exception as e:
                print(f"DEBUG: Failed to preload {f}: {e}")

try:
    from faster_whisper import WhisperModel
    import torch
    
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    model_size = "tiny"
    print(f"Loading Whisper model: {model_size}...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("Whisper model loaded successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
