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


def clone_voice(
    context_audio_path: str,
    context_text: str,
    initial_text: str = None,
    speaker_id: int = 999,
    max_seq_len: int = 4096,
    output_dir: str = "outputs"
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
            speaker=speaker_id,
            audio=context_audio
        )
    ]
    spkr = speaker_id
    
    # Generate initial text if provided
    if initial_text:
        process_text(initial_text, generator, context_segments, spkr, output_dir)
    
    # Interactive mode
    print("\nEntering interactive mode. Available commands:")
    print("- $CLEAR$ : Clear context")
    print("- $SWAP$ : Increment speaker ID")
    print("- $BACK$ : Decrement speaker ID")
    print("- Use || to separate multiple speakers (e.g., 'Hello||Hi there')")
    print("- Ctrl+C to exit")
    
    while True:
        try:
            text = input("\nEnter text to generate (or command): ")
            
            if text == "$CLEAR$":
                print("Clearing context...")
                context_segments = [
                    Segment(text=context_text, speaker=speaker_id, audio=context_audio)
                ]
                print("Context cleared.")
            elif text == "$SWAP$":
                spkr += 1
                print(f"Speaker ID increased to {spkr}")
            elif text == "$BACK$":
                spkr -= 1
                print(f"Speaker ID decreased to {spkr}")
            else:
                process_text(text, generator, context_segments, spkr, output_dir)
                
        except KeyboardInterrupt:
            print("\nExiting voice cloning session...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            traceback.print_exc()


def process_text(text: str, generator: any, context_segments: list, spkr: int, output_dir: str):
    """Process text input and generate audio, handling multi-speaker functionality"""
    import torch
    original_context = context_segments[0]  # Keep reference to original audio context
    generated_audio = None  # Store reference for context update
    
    # Track memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear unused memory
        print(f"\nGPU Memory before generation:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    
    if "||" in text:
        segments = text.split("||")
        audio_files = []
        
        for i, segment in enumerate(segments):
            if i >= 4:  # Maximum 4 speakers
                print("Too many segments (maximum 4), skipping remaining...")
                break
                
            try:
                # Check for speaker offset at the end of text
                if segment[-2:].startswith('-') and segment[-1].isdigit():
                    spkr_offset = int(segment[-2:])
                    segment = segment[:-2]
                elif segment[-1].isdigit():
                    spkr_offset = int(segment[-1])
                    segment = segment[:-1]
                else:
                    spkr_offset = i
            except:
                spkr_offset = i
                
            current_spkr = spkr + spkr_offset
            print(f"\nGenerating audio for speaker {current_spkr}: '{segment}'")
            
            # Generate with float32 precision for better audio quality
            with torch.cuda.amp.autocast(enabled=False):  # Disable mixed precision for better quality
                audio = generator.generate(
                    text=segment,
                    speaker=current_spkr,
                    context=[original_context],  # Use only original reference audio
                    max_audio_length_ms=25_000,
                    temperature=0.7,
                    topk=25
                )
            
            # Minimal processing - just normalize
            max_val = torch.max(torch.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95  # Light headroom
            
            # Move to CPU and ensure float32
            audio = audio.cpu().float()
            
            # Save audio with high quality settings
            filename = f"segment_{i+1}.wav"
            filepath = os.path.join(output_dir, filename)
            torchaudio.save(filepath, audio.unsqueeze(0), generator.sample_rate, bits_per_sample=32)
            audio_files.append(filepath)
            
            # Store last generated audio for context
            if i == len(segments) - 1:
                generated_audio = audio.clone()
            
            # Clear memory
            del audio
            torch.cuda.empty_cache()
        
        # Combine audio files with minimal processing
        if len(audio_files) > 1:
            print("\nCombining audio segments...")
            output_path = os.path.join(output_dir, "combined_output.wav")
            ffmpeg_cmd = (
                f"ffmpeg -y " + 
                " ".join(f"-i {f}" for f in audio_files) +
                f" -filter_complex '" +
                "".join(f"[{i}:0]" for i in range(len(audio_files))) +
                f"concat=n={len(audio_files)}:v=0:a=1[out]' " +
                "-map '[out]' " +
                "-acodec pcm_f32le " +  # Use 32-bit float output
                f"{output_path}"
            )
            run(shlex.split(ffmpeg_cmd))
            print(f"Combined audio saved to: {output_path}")
            
            # Clean up intermediate files
            for f in audio_files:
                os.remove(f)
    else:
        # Single speaker generation
        print(f"\nGenerating audio for speaker {spkr}: '{text}'")
        
        # Generate with float32 precision
        with torch.cuda.amp.autocast(enabled=False):
            audio = generator.generate(
                text=text,
                speaker=spkr,
                context=[original_context],
                max_audio_length_ms=25_000,
                temperature=0.7,
                topk=25
            )
        
        # Minimal processing - just normalize
        max_val = torch.max(torch.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95  # Light headroom
        
        # Move to CPU and ensure float32
        audio = audio.cpu().float()
        
        # Store reference for context update
        generated_audio = audio.clone()
        
        # Save audio with high quality settings
        output_path = os.path.join(output_dir, "output.wav")
        torchaudio.save(output_path, audio.unsqueeze(0), generator.sample_rate, bits_per_sample=32)
        print(f"Audio saved to: {output_path}")
        
        # Clear memory
        del audio
        torch.cuda.empty_cache()
    
    if torch.cuda.is_available():
        print(f"\nGPU Memory after generation:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f}MB")
    
    # Update context with the last generated audio
    if generated_audio is not None:
        context_segments.append(
            Segment(text=text, speaker=spkr, audio=generated_audio)
        )
        if len(context_segments) > 5:
            context_segments = context_segments[-5:]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Voice Cloning with CSM-1B")
    parser.add_argument("--audio", required=True, help="Path to reference audio file")
    parser.add_argument("--text", required=True, help="Transcription of the reference audio")
    parser.add_argument("--initial", help="Initial text to generate (optional)")
    parser.add_argument("--speaker-id", type=int, default=999, help="Speaker ID (default: 999)")
    parser.add_argument("--seq-len", type=int, default=4096, help="Model sequence length (default: 4096)")
    parser.add_argument("--output-dir", default="outputs", help="Output directory (default: outputs)")
    
    args = parser.parse_args()
    
    clone_voice(
        context_audio_path=args.audio,
        context_text=args.text,
        initial_text=args.initial,
        speaker_id=args.speaker_id,
        max_seq_len=args.seq_len,
        output_dir=args.output_dir
    )