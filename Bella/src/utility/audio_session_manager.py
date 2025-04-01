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
        self._state_transition_lock = asyncio.Lock()
        self._last_state_change = 0

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

    async def pause_session(self) -> None:
        """Pause the current session with state transition protection."""
        async with self._state_transition_lock:
            current_time = time.time()
            if current_time - self._last_state_change < 1.0:
                # Prevent rapid state changes
                return
                
            self.should_pause = True
            self._last_state_change = current_time
            if self.debug:
                print("\nPausing session...")
            
            # Ensure recorder is properly stopped
            if self.recorder and self.recorder.is_recording:
                self.recorder.stop_recording()
                await asyncio.sleep(0.2)  # Small delay to ensure cleanup

    async def start_session(self) -> Tuple[Optional[str], List[str]]:
        """Start a new recording session with improved timeouts and state management.
        
        Returns:
            Tuple[Optional[str], List[str]]: The transcribed text and any segment info
        """
        self.clips = []
        self.should_stop = False
        self.should_pause = False
        
        # Track time to ensure we don't get stuck waiting
        start_time = time.time()
        timeout = 60.0  # Maximum time to wait for audio input
        last_debug_time = time.time()
        debug_interval = 10.0  # Status update every 10 seconds
        
        while not self.should_stop and not self.should_pause:
            # Periodic debug status updates
            current_time = time.time()
            if self.debug and current_time - last_debug_time > debug_interval:
                recorder_status = "recording" if self.recorder and self.recorder.is_recording else "not recording"
                print(f"\nAudio session state: wait_time={current_time - start_time:.1f}s, recorder={recorder_status}")
                last_debug_time = current_time
                
                # If recorder isn't recording but should be, try to restart it
                if self.recorder and not self.recorder.is_recording and not self.should_pause:
                    print("\nDetected recorder not active, attempting to restart...")
                    self.recorder.start_recording()
            
            if self.recorder and self.recorder.last_recording:
                audio_file = self.recorder.last_recording
                self.recorder.last_recording = None  # Clear it immediately to prevent reprocessing
                
                await self.process_audio_clip(audio_file)
                
                # Return immediately after processing one clip
                if self.clips:
                    return self.clips[-1], self.clips
            
            # Check for timeout to avoid infinite waiting
            if time.time() - start_time > timeout:
                if self.debug:
                    print("\nTimeout waiting for audio input in start_session")
                    
                # Try to restart the recorder as a recovery attempt
                if self.recorder:
                    print("\nTimeout reached - restarting recorder")
                    self.recorder.reset_state()
                    self.recorder.start_recording()
                
                break
                
            await asyncio.sleep(0.1)

        # If we're stopping/pausing without processing a clip
        if not self.clips:
            return None, []
            
        return self.clips[-1], self.clips
        
    async def resume_session(self) -> None:
        """Resume or start a new session with state transition protection."""
        async with self._state_transition_lock:
            current_time = time.time()
            if current_time - self._last_state_change < 1.0:
                # Prevent rapid state changes
                return
                
            self.should_pause = False
            self._last_state_change = current_time
            self.clips = []  # Clear any previous clips to start fresh
            
            # Ensure recorder is properly started
            if self.recorder:
                if self.recorder.is_recording:
                    self.recorder.stop_recording()
                    await asyncio.sleep(0.2)  # Small delay for cleanup
                self.recorder.reset_state()
                self.recorder.start_recording()
            
            if self.debug:
                print("\nResuming session - ready for new transcriptions")
