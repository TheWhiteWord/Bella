import sounddevice as sd
import numpy as np
import logging
import sys

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_audio_device(device_index=None):
    """Test if we can play a simple sine wave through the audio device."""
    try:
        # Get device info
        devices = sd.query_devices()
        if device_index is not None:
            device_info = devices[device_index]
            print(f"\nTesting device {device_index}: {device_info['name']}")
            print(f"Sample rate: {device_info['default_samplerate']}")
            sd.default.device = (None, device_index)
            sample_rate = int(device_info['default_samplerate'])
        else:
            sample_rate = 44100
            print("\nTesting default audio device")
        
        # Generate a simple sine wave
        duration = 2  # seconds
        frequency = 440  # Hz (A4 note)
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * frequency * t)
        
        # Play the test tone
        print(f"Playing {frequency}Hz test tone for {duration} seconds...")
        sd.play(samples, samplerate=sample_rate)
        sd.wait()
        print("Test tone completed")
        
        return True
        
    except Exception as e:
        print(f"Error testing audio device: {e}")
        return False

def list_audio_devices():
    """List all available audio devices."""
    devices = sd.query_devices()
    print("\nAvailable audio devices:")
    print("-" * 50)
    for i, dev in enumerate(devices):
        if dev['max_output_channels'] > 0:  # Only show output devices
            print(f"{i}: {dev['name']}")
            print(f"   Outputs: {dev['max_output_channels']}")
            print(f"   Sample rate: {dev['default_samplerate']}")
            print(f"   Host API: {sd.query_hostapis(dev['hostapi'])['name']}")
            print("-" * 50)

if __name__ == "__main__":
    list_audio_devices()
    
    while True:
        try:
            choice = input("\nEnter device number to test (-1 for default, q to quit): ").strip()
            if choice.lower() == 'q':
                break
                
            device_idx = int(choice)
            if device_idx == -1:
                device_idx = None
                
            if test_audio_device(device_idx):
                again = input("\nDid you hear the test tone? (y/n): ").strip().lower()
                if again == 'y':
                    print("Great! Audio output is working on this device.")
                else:
                    print("No sound heard. Try another device.")
            
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
            continue