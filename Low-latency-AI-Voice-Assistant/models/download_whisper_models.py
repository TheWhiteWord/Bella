import os
import sys
from faster_whisper import download_model
from pathlib import Path
import shutil

def download_whisper_models(model_sizes=["tiny", "base", "small", "medium"], output_dir=None):
    """
    Download Whisper models for offline use.
    
    Args:
        model_sizes (list): List of model sizes to download
        output_dir (str): Directory to save models. Defaults to ./models/whisper/
    """
    if output_dir is None:
        # Get the directory where this script is located
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "whisper")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for size in model_sizes:
        try:
            print(f"\nDownloading {size} model...")
            size_dir = os.path.join(output_dir, size)
            os.makedirs(size_dir, exist_ok=True)
            
            # Download to a temporary location
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            download_model(size, output_dir=temp_dir)
            
            # Move files to the size-specific directory
            for file in os.listdir(temp_dir):
                src = os.path.join(temp_dir, file)
                dst = os.path.join(size_dir, file)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            
            print(f"Successfully downloaded {size} model to {size_dir}")
            
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            print(f"Error downloading {size} model: {e}")

if __name__ == "__main__":
    # Allow specifying model sizes via command line
    model_sizes = sys.argv[1:] if len(sys.argv) > 1 else ["tiny", "base", "small", "medium"]
    download_whisper_models(model_sizes)