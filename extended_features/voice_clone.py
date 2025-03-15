import os
from huggingface_hub import hf_hub_download
import torchaudio
from generator import load_csm_1b, Segment, CACHE_DIR
import argparse

def load_audio(audio_path, target_sample_rate):
    """Load and resample audio file."""
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    if sample_rate != target_sample_rate:
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), 
            orig_freq=sample_rate, 
            new_freq=target_sample_rate
        )
    return audio_tensor

class VoiceCloner:
    def __init__(self, device="cuda"):
        print("Initializing voice cloner...")
        self.model_path = os.path.join(CACHE_DIR, "ckpt.pt")
        self.generator = load_csm_1b(self.model_path, device)
        self.context_segments = []
        print("Voice cloner initialized successfully!")

    def add_reference_audio(self, audio_path, transcript):
        """Add a reference audio sample for voice cloning."""
        print(f"Loading reference audio from {audio_path}")
        reference_audio = load_audio(audio_path, self.generator.sample_rate)
        
        self.context_segments = [
            Segment(text=transcript, speaker=0, audio=reference_audio)
        ]
        print("Reference audio loaded and added to context")

    def clone_voice(self, text, output_path="output.wav", max_duration_ms=25000):
        """Generate speech with the cloned voice."""
        if not self.context_segments:
            raise ValueError("No reference audio added. Please add reference audio first.")

        print(f"Generating speech for: '{text}'")
        audio = self.generator.generate(
            text=text,
            speaker=0,
            context=self.context_segments,
            max_audio_length_ms=max_duration_ms,
        )
        
        torchaudio.save(output_path, audio.unsqueeze(0).cpu(), self.generator.sample_rate)
        print(f"Generated audio saved to {output_path}")
        return audio

def main():
    parser = argparse.ArgumentParser(description="Voice Cloning Tool")
    parser.add_argument("--reference", type=str, help="Path to reference audio file")
    parser.add_argument("--transcript", type=str, help="Transcript of the reference audio")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output file path")
    
    args = parser.parse_args()

    if not (args.reference and args.transcript and args.text):
        parser.print_help()
        return

    cloner = VoiceCloner()
    cloner.add_reference_audio(args.reference, args.transcript)
    cloner.clone_voice(args.text, args.output)

if __name__ == "__main__":
    main()
