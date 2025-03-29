#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bella

# Change to the project directory
cd /media/theww/AI/Code/AI/Bella/Bella

# Run the voice assistant
python main.py "$@"
