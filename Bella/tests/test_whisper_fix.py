import os
import sys
import subprocess

# Set up environment for the test
env_path = "/home/theww/miniconda3/envs/bella"
site_packages = os.path.join(env_path, "lib/python3.11/site-packages")
nvidia_cudnn_lib = os.path.join(site_packages, "nvidia/cudnn/lib")
nvidia_cublas_lib = os.path.join(site_packages, "nvidia/cublas/lib")

# Prepend pip's CUDA libs to LD_LIBRARY_PATH
current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{nvidia_cudnn_lib}:{nvidia_cublas_lib}:{current_ld_path}"

print(f"Testing environment with LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")

try:
    from faster_whisper import WhisperModel
    import torch
    
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    model_size = "tiny" # Use tiny for fast test
    print(f"Loading Whisper model: {model_size}...")
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    print("Whisper model loaded successfully!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
