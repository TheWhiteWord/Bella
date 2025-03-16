import os
from pathlib import Path
import torch
import torchaudio
import traceback
from subprocess import run
import shlex
from generator import load_csm_1b, Segment, CACHE_DIR
from dotenv import load_dotenv
from models import ModelArgs


def load_csm_1b_with_seq_len(ckpt_path: str, device: str = "cuda", max_seq_len: int = 4096) -> any:
    """Load CSM-1B model with configurable sequence length"""
    from models import Model, ModelArgs
    import torch
    
    # Enable CUDA optimization
    torch.backends.cudnn.benchmark = True
    
    # Print CUDA memory status
    if device == "cuda":
        print(f"\nGPU Memory before model load:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
    
    model_args = ModelArgs(
        backbone_flavor="llama-1B",
        decoder_flavor="llama-100M",
        text_vocab_size=128256,
        audio_vocab_size=2051,
        audio_num_codebooks=32,
        max_seq_len=max_seq_len  # Pass the sequence length to ModelArgs
    )
    
    # Use mixed precision for better memory efficiency
    with torch.amp.autocast('cuda'):  # Updated autocast syntax
        model = Model(model_args).to(device=device, dtype=torch.bfloat16)
        print(f"Loading model from: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    
    if device == "cuda":
        print(f"\nGPU Memory after model load:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
        print(f"Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f}MB")
    
    from generator import Generator
    generator = Generator(model, max_seq_len=max_seq_len)
    return generator


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


def process_text(text: str, generator, context_segments: list, output_dir: str, speaker_id: int = 0):
    """Process input text and generate audio"""
    try:
        segments = [
            Segment(
                text=text,
                audio=None,
                speaker=speaker_id
            )
        ]
        audio = generator.generate(
            text=text,
            speaker=speaker_id,
            context=context_segments,
            max_audio_length_ms=25_000,
        )
        
        # Save the generated audio - ensure tensor is on CPU
        output_path = os.path.join(output_dir, "output.wav")
        audio_cpu = audio.cpu() if audio.device.type != "cpu" else audio
        torchaudio.save(output_path, audio_cpu.unsqueeze(0), generator.sample_rate)
        print(f"Saved audio to {output_path}")
        
        # Update context with the new segment - keep the audio on original device
        context_segments.append(Segment(text=text, audio=audio, speaker=speaker_id))
        return True
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return False


def clone_voice(
    context_audio_path: str,
    context_text: str,
    initial_text: str = None,
    max_seq_len: int = 4096,
    output_dir: str = "outputs",
    speaker_id: int = 0
):
    """Main voice cloning function with interactive mode"""
    load_dotenv()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading model...")
    model_path = get_model_path()
    if not model_path:
        raise ValueError("Could not find model file in the cache directory")
    
    generator = load_csm_1b_with_seq_len(model_path, "cuda", max_seq_len)
    print("Model loaded successfully!")
    
    # Load and process reference audio
    print(f"Loading reference audio: {context_audio_path}")
    context_audio, sr = torchaudio.load(context_audio_path)
    context_audio = context_audio.mean(dim=0)  # Convert to mono
    
    if sr != generator.sample_rate:
        context_audio = torchaudio.functional.resample(
            context_audio, orig_freq=sr, new_freq=generator.sample_rate
        )
    
    # Normalize audio
    context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)
    
    # Create initial context
    context_segments = [
        Segment(
            text=context_text,
            audio=context_audio,
            speaker=speaker_id
        )
    ]
    
    # Generate initial text if provided
    if initial_text:
        process_text(initial_text, generator, context_segments, output_dir, speaker_id)
    
    # Interactive mode
    print("\nEntering interactive mode. Type 'exit' to quit.")
    print("- $CLEAR$ : Clear context")
    
    while True:
        try:
            text = input("\nEnter text (or 'exit' to quit): ")
            if text.lower() == 'exit':
                break
            elif text == "$CLEAR$":
                context_segments = [context_segments[0]]  # Keep only reference segment
                print("Context cleared.")
                continue
                
            process_text(text, generator, context_segments, output_dir, speaker_id)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Voice Cloning with CSM-1B")
    parser.add_argument("--audio", required=True, help="Path to reference audio file")
    parser.add_argument("--text", required=True, help="Transcription of the reference audio")
    parser.add_argument("--initial", help="Initial text to generate (optional)")
    parser.add_argument("--seq-len", type=int, default=4096, help="Model sequence length (default: 4096)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    parser.add_argument("--speaker-id", type=int, default=0, help="Speaker ID (default: 0)")
    
    args = parser.parse_args()
    
    clone_voice(
        context_audio_path=args.audio,
        context_text=args.text,
        initial_text=args.initial,
        max_seq_len=args.seq_len,
        output_dir=args.output_dir,
        speaker_id=args.speaker_id
    )