"""
System Audio Listener for Bella Visualizer

Captures system audio from a PulseAudio/PipeWire monitor source (e.g., 'alsa_output.pci-0000_30_00.6.analog-stereo.monitor')
and feeds audio chunks to a callback (e.g., AudioProcessor.add_audio_chunk).

Does NOT use PortAudio. Uses parec for compatibility.
"""
import subprocess
import numpy as np
import threading

class SystemAudioListener:
    def __init__(self, monitor_source, sample_rate=48000, chunk_size=1024, callback=None):
        self.monitor_source = monitor_source
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.callback = callback  # Function to call with each audio chunk (numpy array)
        self._stop = threading.Event()
        self.thread = None

    def start(self):
        self._stop.clear()
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=1)

    def _listen(self):
        cmd = [
            'parec',
            '--device', self.monitor_source,
            '--format=float32le',
            '--rate', str(self.sample_rate),
            '--channels=1',
            '--latency-msec=10'
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        try:
            while not self._stop.is_set():
                data = proc.stdout.read(self.chunk_size * 4)  # 4 bytes per float32
                if not data:
                    break
                audio_chunk = np.frombuffer(data, dtype=np.float32)
                if self.callback and len(audio_chunk) > 0:
                    self.callback(audio_chunk)
        finally:
            proc.terminate()
