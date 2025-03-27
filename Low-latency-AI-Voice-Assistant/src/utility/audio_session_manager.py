"""
Audio Session Manager - Handles the processing of audio clips from the recorder.

This simplified version processes each clip immediately when received from the recorder.
The recorder handles voice detection and silence detection, while this manager
handles the coordination between recording, transcription, and the main conversation loop.
"""
import time
from typing import List, Optional, Tuple
import asyncio
import os
from src.whisper.faster_whisper_stt_tiny import transcribe_audio

class AudioSessionManager:
    def __init__(self, gap_timeout: float = 2.0, debug: bool = False):
        """Initialize the session manager.
        
        Args:
            gap_timeout (float): Kept for compatibility but no longer used
            debug (bool): Whether to print debug information
        """
        self.debug = debug
        self.clips: List[str] = []
        self.should_stop = False
        self.should_pause = False
        self.recorder = None

    def set_recorder(self, recorder) -> None:
        """Set the recorder instance."""
        self.recorder = recorder

    async def process_audio_clip(self, audio_file: str) -> None:
        """Process a recorded audio clip through the STT pipeline."""
        if not os.path.exists(audio_file):
            return
            
        # Transcribe the audio
        transcribed_text = await transcribe_audio(audio_file)
        
        if transcribed_text:
            if self.debug:
                print(f"\nAdding clip: {transcribed_text}")
            self.clips.append(transcribed_text)
            print(f"\nProcessed clip: {transcribed_text}")

    def pause_session(self) -> None:
        """Pause the current session."""
        self.should_pause = True
        if self.debug:
            print("\nPausing session...")

    def resume_session(self) -> None:
        """Resume or start a new session."""
        self.should_pause = False
        if self.debug:
            print("\nResuming session...")

    async def start_session(self) -> Tuple[Optional[str], List[str]]:
        """Start a new session and process audio clips.
        
        In this simplified version, we process each clip immediately when it's received
        from the recorder. The recorder handles voice detection and silence detection.
        
        Returns:
            tuple: (transcribed_text, list_of_clips)
            - transcribed_text: The transcribed text from the current clip
            - list_of_clips: List containing all processed clips in this session
        """
        self.clips = []
        self.should_stop = False
        self.should_pause = False

        if self.debug:
            print("\nStarting new session...")

        while not self.should_stop and not self.should_pause:
            # Check for new recording
            if self.recorder and self.recorder.last_recording:
                audio_file = self.recorder.last_recording
                self.recorder.last_recording = None  # Clear it immediately to prevent reprocessing
                
                await self.process_audio_clip(audio_file)
                
                # Return immediately after processing one clip
                if self.clips:
                    return self.clips[-1], self.clips

            await asyncio.sleep(0.1)

        # If we're stopping/pausing without processing a clip
        if not self.clips:
            return None, []
            
        return self.clips[-1], self.clips

    def stop_session(self) -> None:
        """Stop the current session."""
        self.should_stop = True