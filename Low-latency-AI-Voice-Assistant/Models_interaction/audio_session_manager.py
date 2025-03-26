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

class AudioSessionManager:
    def __init__(self, gap_timeout: float = 2.0, debug: bool = False):
        """Initialize the session manager.
        
        Args:
            gap_timeout (float): Time in seconds to wait between receiving clips before considering a session complete
            debug (bool): Whether to print debug information
        """
        self.gap_timeout = gap_timeout
        self.debug = debug
        self.last_clip_time = None
        self.clips = []
        self.should_stop = False
        
    async def start_session(self) -> List[str]:
        """Start a new session and wait for clips until gap timeout is reached.
        
        Returns:
            List[str]: List of transcribed clips from the session
        """
        self.clips = []
        self.last_clip_time = None
        self.should_stop = False
        
        if self.debug:
            print("\nStarting new session...")
        
        while not self.should_stop:
            current_time = time.time()
            
            # If we have clips and enough time has passed since the last one
            if self.last_clip_time and (current_time - self.last_clip_time) > self.gap_timeout:
                if self.debug:
                    print(f"\nGap timeout reached ({self.gap_timeout}s), ending session...")
                break
                
            await asyncio.sleep(0.1)  # Prevent CPU overload
            
        return self.clips
    
    def add_clip(self, transcribed_text: str):
        """Add a new transcribed clip to the current session and reset the gap timer.
        
        Args:
            transcribed_text (str): The transcribed text from the audio clip
        """
        if transcribed_text:
            if self.debug:
                print(f"\nAdding clip: {transcribed_text}")
            self.clips.append(transcribed_text)
            self.last_clip_time = time.time()
    
    def stop_session(self):
        """Stop the current session."""
        self.should_stop = True

# Example usage
async def main():
    manager = AudioSessionManager(debug=True)
    text = await manager.start_session()
    if text:
        print(f"\nFinal transcription: {text}")

if __name__ == "__main__":
    asyncio.run(main())