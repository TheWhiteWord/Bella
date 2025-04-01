## Documentation: Voice Assistant GUI Implementation

**Project:** Voice assistant with Python Tkinter GUI, Kokoro TTS, and local LLM integration.

**Core Technologies:**
*   GUI: Python `tkinter`, `tkinter.ttk`, `tkinterdnd2` (for drag-and-drop)
*   Audio Visualization: `subprocess` module calling `parec` (PulseAudio) or `pw-record` (PipeWire), `numpy` for data processing
*   Threading: Python `threading` module, `queue` module, and `asyncio` for backend operations
*   Backend Integration: `BackendAdapter` bridging GUI with async speech, LLM, and TTS components
*   TTS: Kokoro TTS with PulseAudio/PipeWire output
*   STT: Whisper model for speech recognition
*   LLM: Local models via Ollama

**1. Overview**

The voice assistant provides a simple, visually appealing GUI for interacting with a local LLM using speech. The application:
*   Continuously listens for user voice input by default
*   Processes user speech via whisper-based STT, interacts with a local Ollama LLM, and responds via Kokoro TTS
*   Allows users to optionally load context from a `.txt` file via drag-and-drop
*   Uses loaded context in LLM interactions for more informed responses
*   Provides a function to read the loaded text context aloud using TTS
*   Allows downloading the generated audio (`.wav`) of the read-aloud context
*   Displays a real-time audio waveform visualization during audio playback

**2. Core Features & Interaction Flow**

*   **Initialization:**
    *   GUI window appears with waveform display area and control buttons
    *   Backend components (recorder, audio manager, TTS) are initialized
    *   Continuous listening starts automatically
*   **Default Chat Interaction:**
    *   User speaks → BufferedRecorder detects speech through VAD
    *   AudioSessionManager processes audio and obtains transcript
    *   GUI receives transcript and updates status to "Thinking..."
    *   LLM generates response (with context if available)
    *   Kokoro TTS synthesizes and plays speech response
    *   Continuous listening automatically resumes after response
*   **Context Loading (Drag & Drop):**
    *   User drags a `.txt` file onto the application window
    *   GUI reads file content and stores it as context
    *   Context Button changes to "ON" state (green)
    *   "Read Context" button becomes enabled
*   **Read Context Aloud:**
    *   User clicks the "Read Context" button
    *   Continuous listening pauses temporarily
    *   Kokoro TTS synthesizes and plays the context content
    *   Audio is saved to a temporary file for potential download
    *   Download Button becomes enabled
    *   Continuous listening automatically resumes after playback
*   **Download Context Audio:**
    *   User clicks the Download Button
    *   A "Save As" dialog appears
    *   User selects destination for the audio file
    *   GUI copies the audio file from temporary location to user-selected path
*   **Context Clearing:**
    *   User clicks the Context Button when in "ON" state
    *   Context is cleared from memory
    *   Context Button returns to "OFF" state (grey)
    *   Read Context and Download buttons become disabled

**3. GUI Components & Styling**

*   **Main Window:**
    *   Background: Light green (`#C8E6C9`)
    *   Size: 450x280 pixels (defaults, resizable)
    *   Title: "Voice Assistant"
*   **Display Area:**
    *   Background: Light Coral (`#F08080`)
    *   Border: Sienna (`#8B4513`)
    *   Content: Displays current status: "Listening...", "Thinking...", "Speaking...", etc.
*   **Button Row:**
    *   Context Button:
        *   OFF state: Dark grey (`#696969`), white text "Context: OFF"
        *   ON state: Light green (`#ADFF2F`), black text "Context: ON"
    *   Download Button:
        *   Dark grey (`#333333`), white text "📥 Download"
        *   Initially disabled, enables after context reading
    *   Read Context Button:
        *   Red (`#FF4136`), white text "Read Context"
        *   Initially disabled, enables when context is loaded

**4. State Management**

The GUI maintains these important state variables:
*   `is_context_loaded`: Tracks if context is currently loaded
*   `context_content`: Stores the text content from loaded file
*   `is_reading_context`: True during context TTS playback
*   `is_processing`: True during any backend operation
*   `last_tts_filepath`: Path to the generated audio file
*   `audio_subprocess`: Reference to the audio capture process
*   `audio_queue`: Thread-safe queue for audio visualization data

**5. BackendAdapter Interface**

The `BackendAdapter` class bridges between the GUI and our async backend implementation:

*   `backend = BackendAdapter(sink_name=None)`: Creates adapter with optional audio sink
*   `backend.start_continuous_listening(callback_on_speech)`: Starts listening in background thread
*   `backend.pause_listening()`: Temporarily stops audio input
*   `backend.resume_listening()`: Resumes audio input after pause
*   `backend.get_llm_response(transcript: str, context: str | None) -> str`: Gets LLM response
*   `backend.synthesize_and_play(text: str, output_filepath: str | None = None) -> str | None`: Generates and plays speech
*   `backend.cleanup()`: Cleans up resources on exit

**Implementation Details:**

The `BackendAdapter` wraps these components:
1. `BufferedRecorder`: Handles audio recording and VAD using PipeWire/PulseAudio
2. `AudioSessionManager`: Manages speech sessions and coordinates with the recorder
3. `KokoroTTSWrapper`: Synthesizes speech using Kokoro TTS
4. `generate_chat_response`: Interacts with local LLMs via Ollama

The adapter maintains an asyncio event loop in a background thread for processing async operations while keeping the GUI responsive.

**6. Threading Model**

The implementation uses multiple threads for responsiveness:

*   **Main Thread**: Runs the Tkinter UI event loop
*   **Listening Thread**: Runs in the adapter to handle continuous audio processing
*   **LLM Processing Thread**: Handles LLM inference without blocking GUI
*   **TTS Thread**: Manages speech synthesis and playback
*   **Waveform Capture Thread**: Captures audio data for visualization

Thread synchronization occurs through:
*   Queue-based data passing for waveform visualization
*   Thread locks to protect critical sections
*   Tkinter's `after()` method for safe UI updates from background threads
*   The adapter's asyncio event loop for async operations

**7. Dependencies**

*   **Python Libraries:** `tkinter`, `tkinter.ttk`, `tkinterdnd2-universal`, `numpy`, `asyncio`
*   **External Tools:** PulseAudio (`paplay`, `parec`) or PipeWire (`pw-record`)
*   **Backend Dependencies:** Kokoro TTS, Whisper STT, Ollama

**8. Launcher Integration**

The application includes a launcher script that provides:
*   Command-line parameter handling
*   Mode selection between CLI and GUI interfaces
*   Audio device configuration options
*   Seamless integration with the existing CLI mode


**9. Key Implementation Features**

*   **Error Resilience**: Comprehensive error handling with automatic recovery
*   **State Verification**: Checks that listening properly resumes after operations
*   **Resource Management**: Proper cleanup of audio resources and temporary files
*   **Diagnostic Logging**: Detailed logging for troubleshooting
*   **Non-blocking Operations**: All potentially blocking operations run in separate threads