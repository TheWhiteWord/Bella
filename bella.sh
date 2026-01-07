#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bella

# Change to the project directory
cd /media/theww/AI/Code/AI/Bella/Bella

# Set up CUDA library paths for pip-installed packages (fixes Whisper/cuDNN errors)
# This includes cudnn, cublas, and the runtime libs
SITE_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
for lib in cudnn cublas cuda_runtime; do
    LIB_PATH="$SITE_PACKAGES/nvidia/$lib/lib"
    if [ -d "$LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$LIB_PATH:$LD_LIBRARY_PATH"
    fi
done

# Run the voice assistant with main.py
python main.py "$@"
