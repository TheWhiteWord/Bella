## Documentation: Voice Assistant GUI Implementation

**Project:** Build a Python Tkinter GUI for a voice assistant based on a provided visual concept.

**Core Technologies:**
*   GUI: Python `tkinter`, `tkinter.ttk`, `tkinterdnd2` (for drag-and-drop)
*   Audio Visualization: `subprocess` module calling `parec` (PulseAudio) or `pw-record` (PipeWire), `numpy` for data processing.
*   Threading: Python `threading` module, `queue` module.
*   Backend Integration: Assumes existing Python backend functions/methods for STT, LLM, TTS, and audio session management (details below).

**1. Overview**

The goal is to create a simple, visually appealing GUI for a voice assistant application. The application should:
*   Continuously listen for user voice input by default.
*   Process user speech via STT, interact with a local LLM, and respond via TTS.
*   Allow users to optionally load context from a `.txt` file via drag-and-drop.
*   Use loaded context in LLM interactions.
*   Provide a function to read the loaded text context aloud using TTS, temporarily pausing the chat interaction loop.
*   Allow downloading the generated audio (`.wav`) of the *read-aloud context*.
*   Display a real-time audio waveform visualization reflecting the audio being currently played (TTS output or read-aloud context).

**2. Core Requirements & Interaction Flow**

*   **Initialization:**
    *   GUI window appears, matching the visual style (see Section 3).
    *   Backend is initialized.
    *   **Continuous Listening Starts:** The application immediately begins listening for voice input via the backend. The display area should indicate "Listening...".
*   **Default Chat Interaction:**
    *   User speaks.
    *   Backend detects speech (VAD), transcribes it (STT).
    *   GUI receives transcript. Display updates to "Thinking...".
    *   GUI sends transcript (and context, if loaded) to the backend's LLM function.
    *   GUI receives LLM response. Display updates to "Speaking...". Waveform becomes active.
    *   GUI sends LLM response to backend's TTS function. TTS audio is played.
    *   Once TTS finishes, waveform stops, display returns to "Listening...". Continuous listening automatically resumes (handled by backend).
*   **Context Loading (Drag & Drop):**
    *   User drags a `.txt` file onto the application window.
    *   GUI accepts the file drop.
    *   GUI reads the file content and stores it internally (`context_content`).
    *   Context Button visually changes to "ON" state (e.g., green).
    *   Record Button (Red Circle) becomes *enabled*.
    *   Continuous listening *continues*. The loaded context is now available for subsequent chat interactions.
*   **Read Context Aloud:**
    *   User clicks the (now enabled) Record Button (Red Circle).
    *   GUI updates display: "Reading Context...".
    *   **Crucially:** GUI signals the backend to *pause* continuous listening.
    *   GUI sends the stored `context_content` to the backend's TTS function, requesting the audio be saved to a known temporary path (e.g., `context_tts.wav`).
    *   Backend synthesizes and plays the audio. The waveform visualization is active during playback.
    *   Once playback finishes, GUI receives confirmation (and the path to the saved `.wav` file).
    *   Download Button becomes *enabled*.
    *   **Crucially:** GUI signals the backend to *resume* continuous listening.
    *   GUI updates display: "Listening...".
*   **Download Context Audio:**
    *   User clicks the (now enabled) Download Button.
    *   GUI opens a "Save As" dialog.
    *   GUI copies the temporary context TTS audio file (e.g., `context_tts.wav`) to the user-selected location.
*   **Context Clearing:**
    *   User clicks the Context Button (Green Button).
    *   GUI clears the internal `context_content` and associated state flags.
    *   Context Button visually changes to "OFF" state (e.g., grey/black).
    *   Record Button (Red Circle) becomes *disabled*.
    *   Download Button becomes *disabled*.
    *   Continuous listening is unaffected.

**3. GUI Specification & Styling (Based on Image)**

*   **Main Window:**
    *   Background: Light green (e.g., `#C8E6C9`).
    *   Size: Approximately 450x280 pixels (adjustable).
    *   Title: "Voice Assistant".
*   **Display Area:**
    *   A rectangular area occupying the top section.
    *   Background: Pinkish/Light Coral (e.g., `#F08080`).
    *   Border: Inner dark brown/sienna border (e.g., `#8B4513`).
    *   Content: Real-time audio waveform visualization (dark red bars, e.g., `#8B0000`) when audio is playing. Otherwise, can display status text ("Listening...", "Thinking...", "Reading...") or be blank. Implement using `tkinter.Canvas`.
*   **Status Label:**
    *   Located below the display area.
    *   Displays text like "Listening...", "Thinking...", "Context loaded", "Drop .txt file...".
    *   Background: Match main window green.
    *   Foreground: Black. Font: Standard size (e.g., Helvetica 10). Use `ttk.Label`.
*   **Button Row Frame:**
    *   A `ttk.Frame` below the status label to hold the buttons horizontally. Background matches window.
*   **Context Button:**
    *   Left-most button.
    *   Appearance:
        *   **OFF:** Dark grey/black background (`#696969`), white text "Context: OFF".
        *   **ON:** Light green background (`#ADFF2F`), black text "Context: ON".
    *   Function: Toggles context off (clears state, disables Record/Download). Use `ttk.Button` with dynamic styling.
*   **Download Button:**
    *   Middle button.
    *   Appearance: Dark grey/black background (`#333333`), white foreground. Can use text "Download" or a Unicode icon (e.g., `📥`).
    *   State: Initially disabled. Enabled only after context has been successfully read aloud. Disabled when context is cleared or a new chat/read operation starts. Use `ttk.Button`.
*   **Record Button:**
    *   Right-most button.
    *   Appearance: Red background (`#FF4136`). Can have text "Read Context" or just be a circle icon (simplest: styled square `ttk.Button`; advanced: Canvas/Image).
    *   State: Initially disabled. Enabled *only* when context is loaded (`is_context_loaded == True`). Disabled otherwise.
    *   Function: When clicked (and enabled), triggers the "Read Context Aloud" sequence.

**4. State Management**

The GUI needs internal variables to track its state:
*   `is_context_loaded` (bool): True if a context file has been successfully loaded.
*   `context_content` (str | None): Stores the text content of the loaded file.
*   `is_reading_context` (bool): True during the "Read Context Aloud" process (TTS playback). Helps ignore potential voice input during this time.
*   `is_processing` (bool): General flag to indicate if a backend operation (LLM, TTS) is in progress, used to disable buttons temporarily.
*   `last_tts_filepath` (str | None): Path to the generated `.wav` file after reading context aloud, used for the download button.
*   `audio_subprocess` (subprocess.Popen | None): Reference to the running `parec`/`pw-record` process for visualization.
*   `audio_queue` (queue.Queue): Thread-safe queue for passing audio data from the capture thread to the GUI thread for visualization.

**5. Backend Interface (Actual Implementation)**

The GUI code will call functions/methods provided by the `BackendAdapter` class, which bridges between the GUI and our existing async-based implementation. The adapter translates between the callback-based approach expected by GUI and the async/await pattern used in our core backend. The interface looks like this:

*   `backend = BackendAdapter(sink_name=None)`: Creates a new backend adapter with optional audio sink name.
*   `backend.start_continuous_listening(callback_on_speech)`: Starts the background listening loop. Calls `callback_on_speech(transcript: str)` whenever speech is detected and transcribed. Runs non-blockingly in its own thread.
*   `backend.pause_listening()`: Temporarily stops the audio input/VAD. Safe to call multiple times.
*   `backend.resume_listening()`: Resumes the audio input/VAD after a pause. Safe to call multiple times.
*   `backend.get_llm_response(transcript: str, context: str | None) -> str`: Sends text to the LLM (optionally including context) and returns the response string. This runs the async `generate_chat_response` function from `chat_manager.py` synchronously for GUI compatibility.
*   `backend.synthesize_and_play(text: str, output_filepath: str | None = None) -> str | None`: Takes text, generates speech using Kokoro TTS, and plays it back immediately. If `output_filepath` is provided, it will save the audio to that path before finishing playback. Returns the `output_filepath` if saving was successful, otherwise `None`.
*   `backend.cleanup()`: Cleans up resources when the application is closed.

**Implementation Details:**

The `BackendAdapter` wraps these existing components:
1. `BufferedRecorder`: Handles audio recording and voice activity detection using PipeWire/PulseAudio
2. `AudioSessionManager`: Manages speech processing sessions and coordinates with the recorder
3. `KokoroTTSWrapper`: Handles text-to-speech generation and playback
4. `generate_chat_response`: Generates responses from the LLM

The adapter maintains an asyncio event loop in a background thread for processing async operations while keeping the GUI responsive. All backend interface methods are designed to be called from the main thread without blocking the GUI.

**6. Waveform Visualization (PipeWire/PulseAudio via `subprocess`)**

*   **Goal:** Visualize the audio being *played* by the application (TTS).
*   **Method:** Use the `subprocess` module to run `parec` or `pw-record` targeting the **monitor source** of the application's output audio sink.
*   **Command:**
    *   Identify the PulseAudio/PipeWire sink name used by the application for TTS output.
    *   Construct the command: `['parec'/'pw-record', '--raw', '--format=s16le', '--rate=44100', '--channels=1', '-d'/'--target', 'YOUR_SINK_NAME.monitor']` (Adjust format, rate, channels, and especially the monitor source name as needed).
*   **Implementation:**
    1.  Start the command using `subprocess.Popen(..., stdout=subprocess.PIPE, stderr=subprocess.PIPE)`. Store the process object.
    2.  Create a dedicated `threading.Thread` to continuously read raw bytes from `process.stdout.read(CHUNK_SIZE)`.
    3.  In the reading thread:
        *   Convert the raw bytes to a NumPy array: `np.frombuffer(raw_bytes, dtype=np.int16)`.
        *   Convert the `int16` array to `float32` between -1.0 and 1.0: `data.astype(np.float32) / 32768.0`.
        *   Put the float NumPy array onto the thread-safe `audio_queue`.
    4.  In the main Tkinter thread:
        *   Have a periodic function (`update_waveform`) scheduled using `root.after(interval_ms, update_waveform)`.
        *   This function reads data from the `audio_queue`.
        *   Process the float audio data: Calculate amplitude values for each bar (e.g., segment the data, find max absolute value or RMS per segment, normalize to canvas height).
        *   Update the `tkinter.Canvas`: Clear old bars (using tags), draw new rectangles (`canvas.create_rectangle`) representing the calculated amplitudes.
*   **Cleanup:** Ensure the `parec`/`pw-record` subprocess is terminated (`process.terminate()`, `process.wait()`, `process.kill()`) when the application closes.

**7. Threading Model**

*   **Main Thread:** Runs the Tkinter event loop (`root.mainloop()`). All GUI updates *must* happen here.
*   **Backend Threads:** The existing backend likely uses its own threads or `asyncio` for continuous listening, STT, LLM, TTS. The GUI should interact with it via the defined interface functions.
*   **Waveform Capture Thread:** A dedicated thread (created by the GUI) reads `stdout` from the `parec`/`pw-record` subprocess.
*   **Communication:**
    *   **Backend -> GUI (Speech Input):** Backend calls the `callback_on_speech` provided by the GUI. This callback *must* use `root.after(0, ...)` to schedule the actual processing (`_process_chat_interaction`) in the main thread.
    *   **Capture Thread -> GUI (Waveform Data):** The capture thread puts processed audio data (`numpy` array) onto a `queue.Queue`. The GUI's `update_waveform` function (running in main thread via `root.after`) reads from this queue.
    *   **GUI -> Backend (LLM/TTS Requests):** GUI calls backend functions. If these are blocking, the GUI must run them in a `threading.Thread` to avoid freezing. If they are `async`, manage the `asyncio` loop appropriately (e.g., run loop in a dedicated thread, use `asyncio.run_coroutine_threadsafe`).

**8. Dependencies**

*   **Python Libraries:** `tkinter`, `tkinter.ttk`, `tkinterdnd2-universal` (or similar), `numpy`.
*   **External Tools:** `parec` (from PulseAudio utils) or `pw-record` (from PipeWire utils) must be installed and in the system's PATH.

**9. Key Implementation Notes**

*   **Error Handling:** Implement `try...except` blocks around file operations, subprocess calls, backend interactions, and audio processing. Provide user feedback via the status label on errors.
*   **Configuration:** Consider making the PulseAudio/PipeWire monitor source name configurable, as it can vary between systems.
*   **Resource Cleanup:** Ensure threads are properly joined (if necessary) and the audio capture subprocess is terminated when the application window is closed.
*   **Responsiveness:** Use threading and the `queue`/`after` pattern correctly to keep the GUI responsive during backend operations and audio processing.

---