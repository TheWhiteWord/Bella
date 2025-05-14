"""
Test script to verify integration between SystemAudioListener and VoiceVisualizerWindow.

- Ensures that intensity updates from system audio are received and visualized.
- Prints debug output for both listener and visualizer.

Usage:
    conda activate bella
    python test_visualizer_integration.py
"""


import sys
import time
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import your modules (adjust import paths as needed)
from voice_visualizer import VoiceVisualizerWindow
from system_audio_listener import SystemAudioListener


def main():
    app = QApplication(sys.argv)


    # Create the visualizer window
    visualizer = VoiceVisualizerWindow()
    visualizer.show()

    # Create the system audio listener (force correct monitor source)
    monitor_source = "alsa_output.pci-0000_30_00.6.analog-stereo.monitor"

    def audio_chunk_callback(audio_chunk):
        # Compute intensity (RMS)
        if len(audio_chunk) == 0:
            intensity = 0.0
        else:
            rms = np.sqrt(np.mean(np.square(audio_chunk)))
            # Normalize: typical float32 PCM is in [-1, 1], so clamp to [0, 1]
            intensity = min(max(rms, 0.0), 1.0)
        # Sensitivity adjustment: higher = more sensitive
        sensitivity = 2.0  # Try 1.5 to 3.0 for more/less sensitivity
        intensity = intensity ** (1 / sensitivity)
        intensity = min(max(intensity, 0.0), 1.0)
        print(f"[Test] Intensity from listener: {intensity}")
        visualizer.set_voice_intensity(intensity)

    listener = SystemAudioListener(monitor_source=monitor_source, callback=audio_chunk_callback)
    listener.start()

    # --- Run Kokoro TTS test automatically ---
    import asyncio
    import threading
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../audio/kokoro_tts')))
    from kokoro_tts import KokoroTTSWrapper
    
    def run_tts():
        async def tts_task():
            tts = KokoroTTSWrapper()
            test_text = "Hello, this is an automated test of the Bella Kokoro TTS system. The visualizer should react to this speech."
            await tts.generate_speech(test_text)
        asyncio.run(tts_task())

    tts_thread = threading.Thread(target=run_tts, daemon=True)
    tts_thread.start()

    # Optionally, stop after some time
    def stop_all():
        print("[Test] Stopping listener and closing visualizer.")
        listener.stop()
        visualizer.close()
        app.quit()

    # Uncomment to auto-stop after 30 seconds
    # QTimer.singleShot(30000, stop_all)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
