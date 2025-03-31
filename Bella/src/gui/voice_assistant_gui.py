"""Voice Assistant GUI implementation based on Tkinter.

This module provides a graphical user interface for the voice assistant,
implementing the design and features described in the Dev.md document.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import tempfile
import shutil
import time
from typing import Optional, List, Tuple

# Try to import tkinterdnd2 for drag and drop functionality
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:
    HAS_DND = False
    print("Warning: tkinterdnd2 not found. Drag and drop will be disabled.")

# Add src to path if needed
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.gui_backend_adapter import BackendAdapter

class VoiceAssistantGUI:
    """Main GUI class for the Voice Assistant application."""
    
    def __init__(self, sink_name: Optional[str] = None):
        """Initialize the GUI.
        
        Args:
            sink_name (str, optional): Name of PulseAudio sink to use for output
        """
        # Initialize state variables
        self.is_context_loaded = False
        self.context_content = None
        self.is_reading_context = False
        self.is_processing = False
        self.last_tts_filepath = None
        self.sink_name = sink_name
        
        # Set up the backend
        self.backend = BackendAdapter(sink_name=sink_name)
        
        # Create and configure the main window
        if HAS_DND:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
            
        self.root.title("Voice Assistant")
        self.root.geometry("450x280")
        self.root.configure(bg="#C8E6C9")
        
        # Create the GUI elements
        self._create_widgets()
        
        # Set up cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Initialize canvas with listening text
        self.root.update()  # Ensure the canvas has been drawn
        canvas_width = self.display_canvas.winfo_width()
        canvas_height = self.display_canvas.winfo_height()
        self.display_canvas.create_text(
            canvas_width / 2, 
            canvas_height / 2,
            text="Listening...",
            fill="#FFFFFF",
            font=("Helvetica", 14, "bold")
        )
        
        # Start continuous listening
        self.backend.start_continuous_listening(self._on_speech_detected)
        
        # Update status
        self._update_status("Listening...")
        
    def _create_widgets(self):
        """Create and arrange all GUI widgets."""
        # Display area (Canvas for waveform)
        self.canvas_frame = ttk.Frame(self.root, padding=5)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.display_canvas = tk.Canvas(
            self.canvas_frame, 
            bg="#F08080",  # Light Coral background
            highlightbackground="#8B4513",  # Sienna border
            highlightthickness=2,
            height=120
        )
        self.display_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = ttk.Label(
            self.root, 
            text="Drop a .txt file to load context...",
            background="#C8E6C9",
            font=("Helvetica", 10)
        )
        self.status_label.pack(pady=5)
        
        # Button row
        self.button_frame = ttk.Frame(self.root, padding=5)
        self.button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create custom styles for buttons
        self._create_button_styles()
        
        # Context button (left)
        self.context_button = ttk.Button(
            self.button_frame, 
            text="Context: OFF",
            style="ContextOff.TButton",
            command=self._toggle_context
        )
        self.context_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Download button (middle)
        self.download_button = ttk.Button(
            self.button_frame, 
            text="📥 Download",
            style="Download.TButton",
            command=self._download_audio,
            state=tk.DISABLED
        )
        self.download_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Record button (right)
        self.record_button = ttk.Button(
            self.button_frame, 
            text="Read Context",
            style="Record.TButton",
            command=self._read_context_aloud,
            state=tk.DISABLED
        )
        self.record_button.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Set up drag and drop if available
        if HAS_DND:
            self.display_canvas.drop_target_register(DND_FILES)
            self.display_canvas.dnd_bind("<<Drop>>", self._on_drop)
        
    def _create_button_styles(self):
        """Create custom styles for buttons."""
        style = ttk.Style()
        
        # Context button styles
        style.configure("ContextOff.TButton", 
                        background="#696969",
                        foreground="white")
        style.map("ContextOff.TButton",
                 background=[('active', '#595959')])
        
        style.configure("ContextOn.TButton", 
                        background="#ADFF2F",
                        foreground="black")
        style.map("ContextOn.TButton",
                 background=[('active', '#9EEF20')])
        
        # Download button style
        style.configure("Download.TButton", 
                        background="#333333",
                        foreground="white")
        style.map("Download.TButton",
                 background=[('active', '#444444')],
                 foreground=[('disabled', '#888888')])
        
        # Record button style
        style.configure("Record.TButton", 
                        background="#FF4136",
                        foreground="white")
        style.map("Record.TButton",
                 background=[('active', '#E03126')],
                 foreground=[('disabled', '#888888')])
    
    def _on_drop(self, event):
        """Handle file drop events."""
        if self.is_processing or self.is_reading_context:
            return
        
        # Get file path from the event
        try:
            file_path = event.data
            # Remove curly braces if present (Windows)
            file_path = file_path.strip("{}").strip()
            
            # Check if it's a text file
            if not file_path.lower().endswith(".txt"):
                self._update_status("Error: Only .txt files are supported")
                return
                
            # Read the file content
            with open(file_path, 'r') as file:
                self.context_content = file.read()
                
            # Update state
            self.is_context_loaded = True
            file_name = os.path.basename(file_path)
            self._update_status(f"Context loaded: {file_name}")
            
            # Update UI
            self.context_button.config(text="Context: ON", style="ContextOn.TButton")
            self.record_button.config(state=tk.NORMAL)
            
        except Exception as e:
            self._update_status(f"Error loading file: {str(e)}")
    
    def _toggle_context(self):
        """Toggle context on/off state."""
        if self.is_processing or self.is_reading_context:
            return
            
        if self.is_context_loaded:
            # Clear context
            self.context_content = None
            self.is_context_loaded = False
            self.context_button.config(text="Context: OFF", style="ContextOff.TButton")
            self.record_button.config(state=tk.DISABLED)
            self.download_button.config(state=tk.DISABLED)
            self._update_status("Context cleared")
            
            # Clean up any stored audio file
            self.last_tts_filepath = None
    
    def _read_context_aloud(self):
        """Handle the Read Context button click."""
        if not self.is_context_loaded or not self.context_content or self.is_processing:
            return
        
        self.is_reading_context = True
        self.is_processing = True
        self.record_button.config(state=tk.DISABLED)
        self.download_button.config(state=tk.DISABLED)
        self._update_status("Reading context...")
        
        # Pause listening during context reading
        self.backend.pause_listening()
        
        # Create a temporary file path for the audio
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        output_path = temp_file.name
        temp_file.close()
        
        # Update canvas to show reading state
        self.display_canvas.delete("all")
        self.display_canvas.configure(bg="#6A5ACD")  # Slate Blue
        
        canvas_width = self.display_canvas.winfo_width()
        canvas_height = self.display_canvas.winfo_height()
        self.display_canvas.create_text(
            canvas_width / 2, 
            canvas_height / 2,
            text="Reading Context...",
            fill="#FFFFFF",
            font=("Helvetica", 14, "bold"),
            tags="reading_indicator"
        )
        
        # Start a thread for TTS to avoid blocking the GUI
        threading.Thread(
            target=self._tts_context_thread,
            args=(self.context_content, output_path),
            daemon=True
        ).start()
    
    def _tts_context_thread(self, text, output_path):
        """Thread function for TTS processing.
        
        Args:
            text: The text to convert to speech
            output_path: The path to save the audio file
        """
        try:
            # Generate speech and save to file
            saved_path = self.backend.synthesize_and_play(text, output_path)
            
            # Update the GUI from the main thread
            self.root.after(0, lambda: self._on_tts_finished(saved_path))
            
        except Exception as e:
            # Handle errors on the main thread
            self.root.after(0, lambda: self._update_status(f"Error: {str(e)}"))
            self.root.after(0, self._cleanup_context_reading)
    
    def _on_tts_finished(self, saved_path):
        """Handle TTS completion."""
        # Store the file path for download
        self.last_tts_filepath = saved_path
        
        # Enable the download button if we have a valid file
        if saved_path and os.path.exists(saved_path):
            self.download_button.config(state=tk.NORMAL)
        
        # Clean up the context reading state
        self._cleanup_context_reading()
        
        # Update status
        self._update_status("Context read complete.")
    
    def _cleanup_context_reading(self):
        """Clean up after context reading."""
        self.is_reading_context = False
        self.is_processing = False
        
        # Reset canvas to listening state
        self.display_canvas.delete("all")
        self.display_canvas.configure(bg="#F08080")  # Light Coral background
        
        canvas_width = self.display_canvas.winfo_width()
        canvas_height = self.display_canvas.winfo_height()
        self.display_canvas.create_text(
            canvas_width / 2, 
            canvas_height / 2,
            text="Listening...",
            fill="#FFFFFF",
            font=("Helvetica", 14, "bold")
        )
        
        # Resume listening
        self.backend.resume_listening()
        
        # Update UI
        if self.is_context_loaded:
            self.record_button.config(state=tk.NORMAL)
        
        self._update_status("Listening...")
    
    def _download_audio(self):
        """Handle Download button click."""
        if not self.last_tts_filepath or not os.path.exists(self.last_tts_filepath):
            self._update_status("Error: No audio file available")
            return
        
        # Open a save dialog
        save_path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav"), ("All files", "*.*")],
            title="Save Audio As"
        )
        
        if not save_path:
            return  # User cancelled
            
        try:
            # Copy the file
            shutil.copy2(self.last_tts_filepath, save_path)
            self._update_status(f"Audio saved to: {os.path.basename(save_path)}")
            
        except Exception as e:
            self._update_status(f"Error saving file: {str(e)}")
    
    def _on_speech_detected(self, transcript):
        """Callback for when speech is detected and transcribed."""
        # Schedule the processing in the main thread
        self.root.after(0, lambda: self._process_chat_interaction(transcript))
    
    def _process_chat_interaction(self, transcript):
        """Process a chat interaction with the backend."""
        if self.is_processing or self.is_reading_context:
            # Ignore if we're busy
            return
            
        self.is_processing = True
        self._update_status("Thinking...")
        
        # Start a thread for getting the response to avoid freezing the GUI
        threading.Thread(
            target=self._chat_response_thread,
            args=(transcript,),
            daemon=True
        ).start()
    
    def _chat_response_thread(self, transcript):
        """Thread function for chat response processing.
        
        Args:
            transcript: The user's speech transcript
        """
        try:
            # Get response from LLM
            response = self.backend.get_llm_response(
                transcript, 
                self.context_content if self.is_context_loaded else None
            )
            
            # Update the GUI from the main thread
            self.root.after(0, lambda: self._speak_response(response))
            
        except Exception as e:
            # Handle errors on the main thread
            self.root.after(0, lambda: self._update_status(f"Error: {str(e)}"))
            self.root.after(0, self._cleanup_processing)
    
    def _speak_response(self, response):
        """Speak the response using TTS.
        
        Args:
            response: The text response to speak
        """
        self._update_status("Speaking...")
        
        # Ensure we're paused while speaking (to avoid self-listening)
        self.backend.pause_listening()
        
        # Clear canvas and show speaking indicator
        self.display_canvas.delete("all")
        self.display_canvas.configure(bg="#6A5ACD")  # Slate Blue
            
        # Draw TTS indicator text
        canvas_width = self.display_canvas.winfo_width()
        canvas_height = self.display_canvas.winfo_height()
        self.display_canvas.create_text(
            canvas_width / 2, 
            canvas_height / 2,
            text="Assistant Speaking...",
            fill="#FFFFFF",
            font=("Helvetica", 14, "bold"),
            tags="tts_indicator"
        )
        
        # Start a thread for TTS to avoid blocking the GUI
        threading.Thread(
            target=self._tts_response_thread,
            args=(response,),
            daemon=True
        ).start()
    
    def _tts_response_thread(self, text):
        """Thread function for TTS processing of chat responses.
        
        Args:
            text: The text to speak
        """
        try:
            # Generate speech without saving
            self.backend.synthesize_and_play(text)
            
            # Update the GUI from the main thread when done
            self.root.after(0, self._cleanup_processing)
            
        except Exception as e:
            # Handle errors on the main thread
            self.root.after(0, lambda: self._update_status(f"Error: {str(e)}"))
            self.root.after(0, self._cleanup_processing)
    
    def _cleanup_processing(self):
        """Clean up after processing and ensure listening resumes."""
        print("\nCleaning up after interaction and resuming listening...")
        self.is_processing = False
        
        # Reset canvas to listening state
        self.display_canvas.delete("all")
        self.display_canvas.configure(bg="#F08080")  # Light Coral background
        
        canvas_width = self.display_canvas.winfo_width()
        canvas_height = self.display_canvas.winfo_height()
        self.display_canvas.create_text(
            canvas_width / 2, 
            canvas_height / 2,
            text="Listening...",
            fill="#FFFFFF",
            font=("Helvetica", 14, "bold")
        )
        
        # Resume listening
        self.backend.resume_listening()
        
        # Start a separate thread to verify that listening resumed correctly
        threading.Thread(
            target=self._verify_listening_resumed,
            daemon=True
        ).start()
        
        # Update status
        self._update_status("Listening...")
    
    def _verify_listening_resumed(self):
        """Verify that listening resumed correctly after a delay, with retry logic."""
        # Give the backend a moment to resume
        time.sleep(1.0)
        
        # Check if listening was not properly resumed
        if self.backend.is_paused or not self.backend.running:
            print("\nDetected listening did not resume correctly, forcing restart...")
            
            # Try to restart listening more aggressively
            try:
                # First try to pause again to reset state
                self.backend.pause_listening()
                time.sleep(0.2)
                
                # Now fully restart
                self.backend.resume_listening()
                time.sleep(0.5)
                
                # Verify one more time
                if self.backend.is_paused or not self.backend.running:
                    print("\nSecond verification failed, trying full restart...")
                    
                    # Try more aggressive approach - recreate the listening thread
                    if hasattr(self.backend, 'speech_callback') and self.backend.speech_callback:
                        callback = self.backend.speech_callback
                        self.backend.cleanup()
                        time.sleep(0.3)
                        self.backend.start_continuous_listening(callback)
            except Exception as e:
                print(f"\nError during verification restart: {e}")
        else:
            print("\nVerified listening resumed correctly")
    
    def _update_status(self, text):
        """Update the status label text."""
        self.status_label.config(text=text)
    
    def _on_close(self):
        """Handle window close event."""
        # Clean up backend resources
        self.backend.cleanup()
        
        # Destroy the window
        self.root.destroy()
    
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()

def main(sink_name=None):
    """Main entry point for the GUI application."""
    app = VoiceAssistantGUI(sink_name=sink_name)
    app.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice Assistant GUI")
    parser.add_argument(
        "--sink",
        type=str,
        help="Name of PulseAudio sink to use"
    )
    args = parser.parse_args()
    main(sink_name=args.sink)