"""
Audio Session Manager - Handles gap detection between received clips. THIS HAS ABSOLUTELY NOTHING TO DO WITH SILENCE DETECTION.
WE ARE NOT USING SILENCE DETECTION TO MEASURE THIS GAP. THIS IS A SIMPLE TIMER THAT WAITS FOR A SPECIFIED TIME BETWEEN RECEIVED AUDIO CLIPS THAT ARE SENT TO WHISPER FOR TRANSCRIPTION.
If a recording is received by Whisper, whisper transcribes. If more clips are received, and they are received within a GAP of 2 sec, they are added to the session.
If a clip is not received for a gap of 2 or more seconds: 
- the recorder gets paused
- the full transcription of the session is sent to the LLM
- the LLM generates a response
- the response is converted to speech
Once the response is played, the recorder is resumed and waits for the next clip.
"""
import time
from typing import List, Optional
import asyncio
import os
from .faster_whisper_stt_tiny import transcribe_audio

class AudioSessionManager:
    def __init__(self, gap_timeout: float = 2.0, max_session_time: float = 10.0, debug: bool = False):
        """Initialize the session manager.
        
        Args:
            gap_timeout (float): Time in seconds to wait for new clips before ending session
            max_session_time (float): Maximum total time for a session before forcing end
            debug (bool): Whether to print debug information
        """
        self.gap_timeout = gap_timeout
        self.max_session_time = max_session_time
        self.debug = debug
        self.last_clip_time = None
        self.session_start_time = None
        self.clips = []
        self.should_stop = False
        self.should_pause = False
        self.recorder = None

    def set_recorder(self, recorder):
        """Set the recorder instance."""
        self.recorder = recorder

    async def process_audio_clip(self, audio_file: str):
        """Process a recorded audio clip through the STT pipeline."""
        from .faster_whisper_stt_tiny import transcribe_audio
        
        # Transcribe the audio
        transcribed_text = await transcribe_audio(audio_file)
        
        if transcribed_text:
            if self.debug:
                print(f"\nAdding clip: {transcribed_text}")
            self.clips.append(transcribed_text)
            self.last_clip_time = time.time()
            
            # Initialize session start time with first clip
            if not self.session_start_time:
                self.session_start_time = time.time()
            
            print(f"\nProcessed clip: {transcribed_text}")

    def pause_session(self):
        """Pause the current session."""
        self.should_pause = True
        if self.debug:
            print("\nPausing session...")

    def resume_session(self):
        """Resume or start a new session."""
        self.should_pause = False
        self.session_start_time = None  # Reset session start time
        if self.debug:
            print("\nResuming session...")

    async def start_session(self) -> tuple[Optional[str], List[str]]:
        """Start a new session and wait for clips.
        
        A session ends when either:
        1. No new clips received for gap_timeout seconds
        2. Max session time reached since first clip
        
        Returns:
            tuple: (combined_text, list_of_clips)
            - combined_text: All transcribed text combined
            - list_of_clips: List of individual transcribed clips
        """
        self.clips = []
        self.last_clip_time = None
        self.session_start_time = None
        self.should_stop = False
        self.should_pause = False

        if self.debug:
            print("\nStarting new session...")

        while not self.should_stop and not self.should_pause:
            current_time = time.time()
            
            if self.recorder and self.recorder.last_recording:
                audio_file = self.recorder.last_recording
                self.recorder.last_recording = None
                await self.process_audio_clip(audio_file)

            # Check for session end conditions
            current_time = time.time()
            
            # Condition 1: Gap timeout - No new clips for gap_timeout seconds
            gap_timeout_reached = (
                self.last_clip_time and 
                current_time - self.last_clip_time >= self.gap_timeout
            )
            
            # Condition 2: Max session time reached
            session_timeout_reached = (
                self.session_start_time and 
                current_time - self.session_start_time >= self.max_session_time
            )

            if self.clips and (gap_timeout_reached or session_timeout_reached):
                if self.debug:
                    if gap_timeout_reached:
                        print(f"\nGap timeout reached ({self.gap_timeout}s), ending session...")
                    else:
                        print(f"\nMax session time reached ({self.max_session_time}s), ending session...")
                break

            await asyncio.sleep(0.1)

        # Combine all clips into one text
        if not self.clips:
            return None, []

        combined_text = " ".join(self.clips)
        return combined_text, self.clips

    def stop_session(self):
        """Stop the current session."""
        self.should_stop = True

# Example usage
async def main():
    manager = AudioSessionManager(debug=True)
    text, clips = await manager.start_session()
    if text:
        print(f"\nFinal transcription: {text}")

if __name__ == "__main__":
    asyncio.run(main())