import os
import torch
import yaml
import torchaudio
from pathlib import Path
import sys

# Add CSM path to system path to import its modules
sys.path.append(str(Path(__file__).parent.parent.parent / 'csm-multi'))
from generator import Segment, load_csm_1b
from models import ModelArgs

class CSMHandler:
    def __init__(self, config_path="../config/settings.yaml", reference_audio=None, reference_text=None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['models']['csm']
            
        self.setup_environment()
        self.load_model()
        
        if reference_audio and reference_text:
            self.setup_voice_context(reference_audio, reference_text)
            
    def setup_environment(self):
        """Configure environment for optimal performance"""
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True
        
    def load_model(self):
        """Initialize the CSM model with configured settings"""
        model_path = os.path.join(self.config['path'], 'ckpt.pt')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"CSM model not found at {model_path}")
            
        self.generator = load_csm_1b(
            ckpt_path=model_path,
            device=self.config['device'],
            max_seq_len=self.config['max_seq_len']
        )
        
    def setup_voice_context(self, audio_path, text):
        """Set up voice context from reference audio"""
        context_audio, sr = torchaudio.load(audio_path)
        context_audio = context_audio.mean(dim=0)  # Convert to mono
        
        if sr != self.generator.sample_rate:
            context_audio = torchaudio.functional.resample(
                context_audio, 
                orig_freq=sr,
                new_freq=self.generator.sample_rate
            )
        
        # Normalize audio
        context_audio = context_audio / (torch.max(torch.abs(context_audio)) + 1e-8)
        
        # Create initial context
        self.context_segments = [
            Segment(
                text=text,
                audio=context_audio,
                speaker=0
            )
        ]
        
    def generate_speech(self, text, output_dir="outputs", speaker_id=0):
        """Generate speech from text"""
        try:
            if not hasattr(self, 'context_segments'):
                raise ValueError("Voice context not set up. Call setup_voice_context first.")
                
            os.makedirs(output_dir, exist_ok=True)
                
            segments = [
                Segment(
                    text=text,
                    audio=None,
                    speaker=speaker_id
                )
            ]
            
            with torch.amp.autocast(device_type=self.config['device']):
                audio = self.generator.generate(
                    text=text,
                    speaker=speaker_id,
                    context=self.context_segments,
                    max_audio_length_ms=25_000,
                )
            
            # Save the generated audio
            output_path = os.path.join(output_dir, "output.wav")
            audio_cpu = audio.cpu() if audio.device.type != "cpu" else audio
            torchaudio.save(output_path, audio_cpu.unsqueeze(0), self.generator.sample_rate)
            
            # Update context
            self.context_segments.append(
                Segment(text=text, audio=audio, speaker=speaker_id)
            )
            
            return output_path
            
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None
        finally:
            torch.cuda.empty_cache()
            
    def clear_context(self):
        """Clear all context except the reference segment"""
        if hasattr(self, 'context_segments') and len(self.context_segments) > 0:
            self.context_segments = [self.context_segments[0]]
            
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            del self.generator
            torch.cuda.empty_cache()
        except:
            pass