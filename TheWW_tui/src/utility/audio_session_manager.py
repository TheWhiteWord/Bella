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
from src.audio.whisper.faster_whisper_stt_tiny import transcribe_audio

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
        self._current_temp_file = None  # Track current temporary file for cleanup

    def set_recorder(self, recorder) -> None:
        """Set the recorder instance."""
        self.recorder = recorder

    async def process_audio_clip(self, audio_file: str) -> None:
        """Process a recorded audio clip through the STT pipeline."""
        if not os.path.exists(audio_file):
            return
            
        try:
            # Store current temp file for cleanup
            self._current_temp_file = audio_file
            
            # Transcribe the audio
            transcribed_text = await transcribe_audio(audio_file)
            if transcribed_text:
                if self.debug:
                    print(f"\nAdding clip: {transcribed_text}")
                self.clips.append(transcribed_text)
                print(f"\nProcessed clip: {transcribed_text}")
            
            # Clean up the temporary file after transcription if it's not a debug file
            if not (self.recorder and self.recorder.debug_mode and 
                   audio_file.startswith(self.recorder.debug_dir)):
                try:
                    os.unlink(audio_file)
                    if self.debug:
                        print(f"\nCleaned up temporary file: {audio_file}")
                except Exception as e:
                    if self.debug:
                        print(f"\nError cleaning up temporary file: {e}")
                        
        except Exception as e:
            print(f"\nError processing audio clip: {e}")
        finally:
            self._current_temp_file = None

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
        """Start a new recording session.
        
        Returns:
            Tuple[Optional[str], List[str]]: The transcribed text and any segment info
        """
        self.clips = []
        self.should_stop = False
        self.should_pause = False
        
        while not self.should_stop and not self.should_pause:
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
        """Stop the current session and clean up."""
        self.should_stop = True
        # Clean up any remaining temporary file, but not debug files
        if self._current_temp_file and os.path.exists(self._current_temp_file):
            if not (self.recorder and self.recorder.debug_mode and 
                   self._current_temp_file.startswith(self.recorder.debug_dir)):
                try:
                    os.unlink(self._current_temp_file)
                    if self.debug:
                        print(f"\nCleaned up temporary file during stop: {self._current_temp_file}")
                except Exception as e:
                    if self.debug:
                        print(f"\nError cleaning up temporary file during stop: {e}")
            self._current_temp_file = None
