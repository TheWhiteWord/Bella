#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate bella

# Change to the project directory
cd /media/theww/AI/Code/AI/The_White_Words/TheWW_tui

# Run the voice assistant
python main_bella.py "$@"
