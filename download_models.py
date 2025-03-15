from huggingface_hub import hf_hub_download
import os
from dotenv import load_dotenv

def download_models():
    # Load environment variables from .env file
    load_dotenv()
    
    # Get the token from environment variable
    token = os.getenv('HUGGING_FACE_TOKEN')
    if not token:
        raise ValueError("Please set HUGGING_FACE_TOKEN in your .env file")
    
    print("Downloading CSM-1B model...")
    model_path = hf_hub_download(
        repo_id="sesame/csm-1b",
        filename="ckpt.pt",
        token=token
    )
    
    # Create symbolic link in models directory
    target_path = os.path.join("models", "ckpt.pt")
    if os.path.exists(target_path):
        os.remove(target_path)
    os.symlink(model_path, target_path)
    
    print("Model downloaded successfully!")
    print(f"Model path: {model_path}")

if __name__ == "__main__":
    download_models()