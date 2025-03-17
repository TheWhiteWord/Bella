import os
import sys
import yaml
from pathlib import Path
import gc

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from audio.recorder import AudioRecorder

class PipelineManager:
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(parent_dir, "config", "settings.yaml")
            
        # Load config file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize audio recorder
        self.recorder = AudioRecorder(config_path)
            
    async def process_voice_input_async(self, duration=None, save_audio=False):
        """Record voice input asynchronously."""
        print("Starting voice recording...")
        try:
            # Record audio
            audio_data = await self.recorder.record_async(duration)
            if audio_data is None:
                print("Failed to record audio")
                return None
            
            return audio_data
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            return None
            
    def process_voice_input(self, duration=None, save_audio=False):
        """Synchronous wrapper for process_voice_input_async."""
        return asyncio.run(self.process_voice_input_async(duration, save_audio))
            
    def __del__(self):
        """Cleanup"""
        try:
            if hasattr(self, 'recorder'):
                del self.recorder
            gc.collect()
        except:
            pass