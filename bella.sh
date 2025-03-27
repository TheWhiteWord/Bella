#!/bin/bash

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate chatbot

# Change to the project directory
cd /media/theww/AI/Code/AI/sesame/Low-latency-AI-Voice-Assistant

# Run the voice assistant
python main.py "$@"
