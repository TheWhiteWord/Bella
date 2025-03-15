from pathlib import Path
import torch
import torchaudio
from dotenv import load_dotenv
import os

def get_model_path():
    """Locate the model in the HuggingFace cache structure"""
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
    hf_path = os.path.join(CACHE_DIR, "models--sesame--csm-1b", "snapshots")
    
    if os.path.exists(hf_path):
        for snapshot in os.listdir(hf_path):
            model_path = os.path.join(hf_path, snapshot, "ckpt.pt")
            if os.path.exists(model_path):
                return model_path
    return None

def load_csm_1b_with_seq_len(ckpt_path: str, device: str = "cuda", max_seq_len: int = 4096) -> any:
    """Load CSM-1B model with configurable sequence length"""
    from models import FLAVORS, llama3_2, Model, ModelArgs
    
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
        max_seq_len=max_seq_len  # Pass the sequence length to ModelArgs
    )
    
    model = Model(model_args).to(device=device, dtype=torch.bfloat16)
    print(f"Loading model from: {ckpt_path}")
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)
    
    # Import generator after model modification
    from generator import Generator
    generator = Generator(model, max_seq_len=max_seq_len)
    return generator

def chunk_audio(audio: torch.Tensor, max_chunk_seconds: float = 20.0, sample_rate: int = 24000):
    """Split audio into manageable chunks"""
    chunk_size = int(max_chunk_seconds * sample_rate)
    chunks = []
    
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:min(i + chunk_size, len(audio))]
        chunks.append(chunk)
    
    return chunks

def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Read transcript
    with open("clone_files/Bella_transcript.txt", "r") as f:
        transcript = f.read().strip()
    
    # Import after environment is loaded so token is available
    from generator import CACHE_DIR, Segment
    
    print("Loading model...")
    model_path = get_model_path()
    if not model_path:
        raise ValueError("Could not find model file in the cache directory")
        
    generator = load_csm_1b_with_seq_len(model_path, "cuda", max_seq_len=4096)
    
    # Load and process reference audio
    print("Loading reference audio...")
    audio_path = "clone_files/Bella_edit.mp3"
    audio, sr = torchaudio.load(audio_path)
    audio = audio.mean(dim=0)  # Convert to mono
    
    if sr != generator.sample_rate:
        audio = torchaudio.functional.resample(
            audio, orig_freq=sr, new_freq=generator.sample_rate
        )
    
    # Normalize audio
    audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    
    # Split audio into chunks
    print("\nSplitting reference audio into chunks...")
    audio_chunks = chunk_audio(audio, max_chunk_seconds=15.0, sample_rate=generator.sample_rate)
    print(f"Created {len(audio_chunks)} chunks")
    
    # Split transcript into roughly equal parts for each chunk
    words = transcript.split()
    words_per_chunk = len(words) // len(audio_chunks)
    transcript_chunks = []
    
    for i in range(len(audio_chunks)):
        start_idx = i * words_per_chunk
        end_idx = start_idx + words_per_chunk if i < len(audio_chunks) - 1 else len(words)
        chunk_transcript = " ".join(words[start_idx:end_idx])
        transcript_chunks.append(chunk_transcript)
    
    # Create context segments from chunks
    context_segments = [
        Segment(
            text=text,
            speaker=999,  # Use a high number for the cloned voice
            audio=chunk
        )
        for text, chunk in zip(transcript_chunks, audio_chunks)
    ]
    
    # Test with a simple phrase
    test_text = "Hello, this is a test of my cloned voice. I hope it sounds natural!"
    print(f"\nGenerating test phrase: '{test_text}'")
    
    # Generate audio
    generated_audio = generator.generate(
        text=test_text,
        speaker=999,
        context=context_segments[-1:],  # Use only the last chunk for context
        max_audio_length_ms=25_000,
        temperature=0.6,  # Lower temperature for more stable output
        topk=20
    )
    
    # Save the generated audio
    output_path = "outputs/bella_test.wav"
    os.makedirs("outputs", exist_ok=True)
    torchaudio.save(output_path, generated_audio.unsqueeze(0).cpu(), generator.sample_rate)
    print(f"\nGenerated audio saved to: {output_path}")
    
    # Test multi-speaker interaction
    multi_text = "Hi, this is my voice after cloning!||And I'm a different speaker responding.||Well, nice to meet you!0"
    print(f"\nTesting multi-speaker generation with: '{multi_text}'")
    
    texts = multi_text.split("||")
    
    # Generate for first speaker (cloned voice)
    audio1 = generator.generate(
        text=texts[0],
        speaker=999,
        context=context_segments[-1:],  # Use only the last chunk for context
        max_audio_length_ms=25_000,
        temperature=0.6,
        topk=20
    )
    
    torchaudio.save("outputs/bella_multi1.wav", audio1.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Generate for second speaker
    audio2 = generator.generate(
        text=texts[1],
        speaker=1,  # Different speaker ID
        context=context_segments[-1:],  # Use only the last chunk for context
        max_audio_length_ms=25_000,
        temperature=0.6,
        topk=20
    )
    
    torchaudio.save("outputs/bella_multi2.wav", audio2.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Generate for first speaker again
    audio3 = generator.generate(
        text=texts[2],
        speaker=999,  # Back to cloned voice
        context=context_segments[-1:],  # Use only the last chunk for context
        max_audio_length_ms=25_000,
        temperature=0.6,
        topk=20
    )
    
    torchaudio.save("outputs/bella_multi3.wav", audio3.unsqueeze(0).cpu(), generator.sample_rate)
    
    # Combine all segments
    print("\nCombining multi-speaker segments...")
    os.system('ffmpeg -i outputs/bella_multi1.wav -i outputs/bella_multi2.wav -i outputs/bella_multi3.wav -filter_complex "[0:0][1:0][2:0]concat=n=3:v=0:a=1[out]" -map "[out]" outputs/bella_multi_combined.wav')
    print("\nTest complete! Check the outputs directory for the generated audio files.")

if __name__ == "__main__":
    main()