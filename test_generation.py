from huggingface_hub import hf_hub_download
from generator import load_csm_1b, CACHE_DIR
import torchaudio
import torch
from dotenv import load_dotenv
import os

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Set the token for model access
    os.environ["HUGGING_FACE_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")
    
    print("Testing basic generation...")
    
    # Use CUDA since it's available
    device = "cuda"
    
    print("Loading model...")
    # Use the model from local models directory
    model_path = os.path.join(CACHE_DIR, "ckpt.pt")
    generator = load_csm_1b(model_path, device)
    
    # Generate a basic test sentence
    text = "Hello! This is a test of CSM Multi-speaker generation."
    print(f"\nGenerating audio for: '{text}'")
    
    audio = generator.generate(
        text=text,
        speaker=0,
        context=[],
        max_audio_length_ms=10_000,
    )

    output_file = "test_output.wav"
    print(f"\nSaving audio to {output_file}...")
    torchaudio.save(output_file, audio.unsqueeze(0).cpu(), generator.sample_rate)
    print("Audio generated successfully!")
    
    # Test multi-speaker functionality
    multi_text = "This is speaker one.||And this is speaker two!"
    print(f"\nTesting multi-speaker generation with: '{multi_text}'")
    
    # Split the text and generate for first speaker
    texts = multi_text.split("||")
    
    audio1 = generator.generate(
        text=texts[0],
        speaker=0,
        context=[],
        max_audio_length_ms=10_000,
    )
    
    torchaudio.save("speaker1.wav", audio1.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Generate for second speaker
    audio2 = generator.generate(
        text=texts[1],
        speaker=1,
        context=[],
        max_audio_length_ms=10_000,
    )
    
    torchaudio.save("speaker2.wav", audio2.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Use ffmpeg to concatenate the audio files
    os.system("ffmpeg -i speaker1.wav -i speaker2.wav -filter_complex '[0:0][1:0]concat=n=2:v=0:a=1[out]' -map '[out]' multi_speaker_test.wav")
    print("\nMulti-speaker test complete! Check multi_speaker_test.wav for the combined output.")

if __name__ == "__main__":
    main()