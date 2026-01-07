import sys
import os
import time

print("Starting import test...", flush=True)

try:
    print("Importing PyQt5...", flush=True)
    from PyQt5.QtWidgets import QApplication
    print("PyQt5 imported.", flush=True)
except Exception as e:
    print(f"Failed to import PyQt5: {e}", flush=True)

try:
    print("Importing voice_visualizer...", flush=True)
    sys.path.insert(0, os.path.abspath("src/ui"))
    import voice_visualizer
    print("voice_visualizer imported.", flush=True)
except Exception as e:
    print(f"Failed to import voice_visualizer: {e}", flush=True)

print("Import test complete.", flush=True)
