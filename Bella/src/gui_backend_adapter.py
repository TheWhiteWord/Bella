"""GUI Backend Adapter for Voice Assistant.

This module provides an adapter interface between the existing backend implementation
and the GUI. It translates between the async/await pattern used in the core backend
and the callback-based approach expected by the GUI.
"""
import os
import asyncio
import tempfile
import shutil
import threading
import time
from typing import Callable, Optional, Dict, Any
import wave
import numpy as np
import traceback

from src.utility.audio_session_manager import AudioSessionManager
from src.utility.buffered_recorder import BufferedRecorder
from src.audio.kokoro_tts.kokoro_tts import KokoroTTSWrapper
from src.llm.chat_manager import generate_chat_response

class BackendAdapter:
    """Adapter class that bridges between the GUI interface and the actual backend implementation."""
    
    def __init__(self, sink_name: Optional[str] = None):
        """Initialize the backend adapter.
        
        Args:
            sink_name (str, optional): Name of PulseAudio sink to use for output
        """
        self.recorder = BufferedRecorder()
        self.audio_manager = AudioSessionManager(debug=True)  # Enable debug for better logging
        self.audio_manager.set_recorder(self.recorder)
        self.tts_engine = None
        self.sink_name = sink_name
        self.speech_callback = None
        self.listening_thread = None
        self.running = False
        self.context = None
        self.asyncio_loop = None
        self.is_paused = False
        self.temp_file_path = None
        self.thread_lock = threading.Lock()
        
        # Add state monitoring
        self.last_state_log = 0
        
    async def _init_tts_engine(self):
        """Initialize the TTS engine asynchronously."""
        if self.tts_engine is None:
            print("Initializing TTS engine...")
            self.tts_engine = KokoroTTSWrapper(
                default_voice="af_bella",
                speed=0.9,
                sink_name=self.sink_name
            )
            print("TTS engine initialized")
        return self.tts_engine
        
    def _run_async(self, coro):
        """Run a coroutine in the asyncio event loop.
        
        Args:
            coro: The coroutine to run
            
        Returns:
            Any: The result of the coroutine
        """
        if self.asyncio_loop is None or self.asyncio_loop.is_closed():
            self.asyncio_loop = asyncio.new_event_loop()
            
        future = asyncio.run_coroutine_threadsafe(coro, self.asyncio_loop)
        return future.result()
        
    def _listening_worker(self):
        """Background thread that continuously listens for speech."""
        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.asyncio_loop = loop
        
        async def listen_continuously():
            # Initialize the TTS engine in this thread
            await self._init_tts_engine()
            
            while self.running:
                # Periodically log state for debugging
                current_time = time.time()
                if current_time - self.last_state_log > 10:  # Log every 10 seconds
                    print(f"\nListening thread state: paused={self.is_paused}, recorder_recording={self.recorder.is_recording}")
                    self.last_state_log = current_time
                
                if self.is_paused:
                    # When paused, just wait
                    await asyncio.sleep(0.1)
                    continue
                
                try:
                    # Reset the recorder state if it's not already recording
                    if not self.recorder.is_recording:
                        print("\nStarting new recording session...")
                        self.recorder.reset_state()
                        self.recorder.start_recording()
                    
                    # Wait for speech
                    transcript, segments = await self.audio_manager.start_session()
                    
                    # If we got a transcript, call the callback
                    if transcript and self.speech_callback and not self.is_paused:
                        print(f"\nSpeech detected in listening thread: {transcript[:30]}...")
                        # Call the speech callback in the main thread
                        if self.speech_callback:
                            self.speech_callback(transcript)
                    
                except Exception as e:
                    print(f"Error in listening worker: {e}")
                    traceback.print_exc()
                    # Wait a bit before trying again
                    await asyncio.sleep(0.5)
                    
        # Run the listen_continuously coroutine in this thread's event loop
        try:
            loop.run_until_complete(listen_continuously())
        except Exception as e:
            print(f"Error in listening thread: {e}")
            traceback.print_exc()
        finally:
            print("Listening thread exiting")
            loop.close()
            
    def start_continuous_listening(self, callback_on_speech: Callable[[str], None]):
        """Start the background listening loop.
        
        Args:
            callback_on_speech: Function to call when speech is detected and transcribed
        """
        with self.thread_lock:
            if self.listening_thread is not None and self.listening_thread.is_alive():
                # Already running
                print("Listening thread already running")
                return
            
            print("\nStarting continuous listening...")
            self.speech_callback = callback_on_speech
            self.running = True
            self.is_paused = False
            
            # Initialize recorder
            self.recorder.initialize_audio_settings()
            self.recorder.reset_state()
            
            # Start the listening thread
            self.listening_thread = threading.Thread(target=self._listening_worker, daemon=True)
            self.listening_thread.start()
            print("Listening thread started")
    
    def pause_listening(self):
        """Temporarily pause the audio input/VAD."""
        with self.thread_lock:
            print("\nPausing listening...")
            self.is_paused = True
            if self.recorder.is_recording:
                self.recorder.should_stop = True
                self.audio_manager.pause_session()
            print("Listening paused")
    
    def resume_listening(self):
        """Resume the audio input/VAD after a pause."""
        with self.thread_lock:
            print("\nResuming listening...")
            self.is_paused = False
            # Ensure recorder is reset before resuming
            self.recorder.reset_state()
            # Signal the listening thread to resume recording
            self.audio_manager.resume_session()
            print("Listening resumed")
    
    async def _get_llm_response_async(self, transcript: str, context: Optional[str] = None) -> str:
        """Get a response from the LLM (async version).
        
        Args:
            transcript: The user's speech transcript
            context: Optional context to include with the request
            
        Returns:
            str: The LLM's response
        """
        # Prepare history context with optional context prepended
        history_context = ""
        if context:
            history_context = f"Context: {context}\n\n"
        
        # Generate the response
        response = await generate_chat_response(transcript, history_context)
        return response
    
    def get_llm_response(self, transcript: str, context: Optional[str] = None) -> str:
        """Get a response from the LLM.
        
        Args:
            transcript: The user's speech transcript
            context: Optional context to include with the request
            
        Returns:
            str: The LLM's response
        """
        return self._run_async(self._get_llm_response_async(transcript, context))
    
    async def _synthesize_and_play_async(self, text: str, output_filepath: Optional[str] = None) -> Optional[str]:
        """Generate speech from text and play it (async version).
        
        Args:
            text: The text to synthesize
            output_filepath: Optional path to save the audio file
            
        Returns:
            str or None: The path to the saved audio file if successful, otherwise None
        """
        # Initialize TTS engine if needed
        tts_engine = await self._init_tts_engine()
        
        # If we need to save the audio, generate it to a temporary file first
        saved_path = None
        
        if output_filepath:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                self.temp_file_path = temp_path
            
            # Generate audio data chunks from text
            generator = tts_engine.pipeline(
                text,
                voice="af_bella",
                speed=0.9,
                split_pattern=r'[.!?]+\s+'
            )
            
            # Process all segments and combine them
            audio_segments = []
            for _, _, audio in generator:
                # Convert to numpy array
                audio_array = audio.detach().cpu().numpy()
                if audio_array.size > 0:
                    # Remove DC offset and normalize
                    audio_array = audio_array - np.mean(audio_array)
                    max_val = np.max(np.abs(audio_array))
                    if max_val > 0:
                        audio_array = audio_array / max_val * 0.9
                    audio_segments.append(audio_array)
            
            # Combine segments with small pauses
            if audio_segments:
                sample_rate = 24000  # Kokoro's default sample rate
                silence = np.zeros(int(0.1 * sample_rate))  # 100ms silence between segments
                combined_audio = audio_segments[0]
                for segment in audio_segments[1:]:
                    combined_audio = np.concatenate([combined_audio, silence, segment])
                
                # Save to temporary file
                with wave.open(temp_path, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(sample_rate)
                    # Convert to 16-bit PCM
                    wav_int16 = (combined_audio * 32767).astype(np.int16)
                    wf.writeframes(wav_int16.tobytes())
                
                # Now play the file
                await tts_engine.generate_speech(text)
                
                # Copy temporary file to the requested output path
                try:
                    shutil.copy2(temp_path, output_filepath)
                    saved_path = output_filepath
                except Exception as e:
                    print(f"Error saving audio file: {e}")
            
            return saved_path
        else:
            # Just play without saving
            await tts_engine.generate_speech(text)
            return None
    
    def synthesize_and_play(self, text: str, output_filepath: Optional[str] = None) -> Optional[str]:
        """Generate speech from text and play it.
        
        Args:
            text: The text to synthesize
            output_filepath: Optional path to save the audio file
            
        Returns:
            str or None: The path to the saved audio file if successful, otherwise None
        """
        print(f"\nSynthesizing speech: {text[:50]}...")
        result = self._run_async(self._synthesize_and_play_async(text, output_filepath))
        print("Speech synthesis complete")
        return result
    
    def cleanup(self):
        """Clean up resources when done."""
        print("\nCleaning up backend adapter resources...")
        # Stop the listening thread
        self.running = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1)
        
        # Stop recording
        if self.recorder:
            self.recorder.should_stop = True
            self.recorder.reset_state()
        
        # Stop TTS
        if self.tts_engine:
            self.tts_engine.stop()
        
        # Clean up temporary files
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except Exception:
                pass
        
        # Close asyncio loop
        if self.asyncio_loop and not self.asyncio_loop.is_closed():
            self.asyncio_loop.stop()
            self.asyncio_loop.close()
        print("Cleanup complete")