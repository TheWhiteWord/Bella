import os
import sys
import yaml
from pathlib import Path
import torch
import gc

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from audio.recorder import AudioRecorder
from models.llm import Phi4Handler

class PipelineManager:
    def __init__(self, config_path=None):
        # Force CUDA cleanup before initialization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
        
        if config_path is None:
            config_path = os.path.join(parent_dir, "config", "settings.yaml")
        
        self.recorder = AudioRecorder(config_path)
        self.llm = Phi4Handler(config_path)
        
    def process_voice_input(self, duration=None, save_audio=False):
        """Record and process voice input"""
        print("Starting voice recording...")
        try:
            # Record audio
            audio_data, sample_rate = self.recorder.record(duration)
            if audio_data is None:
                print("Failed to record audio")
                return None
                
            # Optionally save recorded audio
            if save_audio:
                self.recorder.save_to_file(audio_data, "last_recording.wav")
            
            # Process with LLM
            print("Processing audio with Phi-4 Multimodal...")
            response = self.llm.process_audio(audio_data, sample_rate)
            
            return response
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return None
            
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'llm'):
                del self.llm
            if hasattr(self, 'recorder'):
                del self.recorder
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        except:
            pass