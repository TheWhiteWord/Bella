from huggingface_hub import snapshot_download
import os

def download_phi4_model():
    """Download the Phi-4 multimodal model from HuggingFace"""
    print("Downloading Phi-4 multimodal model...")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Download the model
    model_path = snapshot_download(
        repo_id="microsoft/Phi-4-multimodal-instruct",
        local_dir=os.path.join(models_dir, "microsoft--Phi-4-multimodal-instruct"),
        local_dir_use_symlinks=False,  # Get actual files instead of symlinks
    )
    
    print(f"Model downloaded successfully to: {model_path}")
    return model_path

if __name__ == "__main__":
    download_phi4_model()