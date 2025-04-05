# Sesame Project Environment Setup Guide

This guide provides step-by-step instructions for setting up the development environment for the Sesame project.

## Prerequisites
- Anaconda or Miniconda installed
- NVIDIA GPU with compatible drivers
- Git installed

## Step-by-Step Installation

### 1. Create and Activate Conda Environment
```bash
# Create a new environment with Python 3.11
conda create -n bella python=3.11

# Activate the environment
conda activate bella
```
 Status: Environment created with Python 3.11

### 2. Install CUDA Toolkit

```bash
# Install CUDA toolkit 12.1 from NVIDIA channel

conda install -c nvidia cuda-toolkit=12.1
```
 Status: CUDA toolkit installed

### 3. Install PyTorch with CUDA Support
```bash
# Install PyTorch 2.5.1 with CUDA 12.4 support
pip3 install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
 Status: PyTorch installed with CUDA support

### 4. Install cuDNN
If you encounter the error `libcudnn.so.9: cannot open shared object file: No such file or directory`, install cuDNN:
```bash
# Install cuDNN from conda-forge
conda install -c conda-forge cudnn
```
 Status: cuDNN installed successfully

### 5. Verify CUDA Setup
Run this Python code to verify CUDA is working:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Name:", torch.cuda.get_device_name(0))
```
Expected output should show:
- CUDA is available
- CUDA Version: 12.4
- Your GPU name

### 6. Install Project Dependencies
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt
```

### 7. Dependency Resolution Notes
If you encounter these specific issues:

#### NumPy Compatibility
If you get numpy compatibility errors:
```bash
# Install numpy with version compatible with PyTorch
pip install numpy>=1.21.6
```

#### Missing Dependencies
If you encounter missing dependency errors, install them individually:
```bash
pip install protobuf
pip install urllib3>=1.25
```

## Current Environment Status
- Python Version: 3.11
- CUDA Version: 12.4
- cuDNN Version: 9.8.0.87
- PyTorch: 2.5.1+cu124
- GPU Support: Enabled and verified

## Verification Steps
After installation, verify your setup:

1. Check CUDA Availability:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())  # Should return True
print("CUDA Version:", torch.version.cuda)          # Should show 12.4
print("GPU Name:", torch.cuda.get_device_name(0))   # Should show your GPU name
```

2. Try importing key libraries:
```python
import torch
import torchaudio
import transformers
import numpy as np
```

## Maintenance Tips
- Keep this document updated when making environment changes
- Document any version updates or new dependencies
- Test CUDA functionality after major updates
- Update requirements.txt when adding new dependencies

## Troubleshooting
1. If CUDA is not available:
   - Check if your GPU drivers are properly installed
   - Verify CUDA toolkit installation
   - Ensure cuDNN is properly installed

2. If you encounter dependency conflicts:
   - Try installing conflicting packages individually
   - Check version compatibility with PyTorch
   - Use pip freeze to check installed versions

## Environment Information
- Last Updated: March 20, 2025
- Tested GPU: NVIDIA GeForce RTX 4070 Ti SUPER
- OS: Linux