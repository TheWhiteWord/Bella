import sounddevice as sd
import sys

def test_audio_devices():
    """Test audio device configuration and status"""
    print("\n=== Audio Device Diagnostic Test ===\n")
    
    try:
        # Get default devices
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        print(f"Default input device: {default_input}")
        print(f"Default output device: {default_output}")
        
        print("\nAll available devices:")
        print("-" * 60)
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            # Print detailed device info
            print(f"\nDevice {i}: {dev['name']}")
            print(f"  Max Channels (in/out): {dev['max_input_channels']}/{dev['max_output_channels']}")
            print(f"  Sample Rates: {dev['default_samplerate']}")
            print(f"  Latency (in/out): {dev.get('default_low_input_latency', 'N/A')}/{dev.get('default_low_output_latency', 'N/A')}")
            print(f"  Host API: {dev['hostapi']}")
            
            # If it's the default input device, try to validate it
            if i == default_input and dev['max_input_channels'] > 0:
                print("\nTesting default input device...")
                try:
                    with sd.InputStream(device=i, channels=1, samplerate=44100):
                        print("  ✅ Input device test successful")
                except Exception as e:
                    print(f"  ❌ Input device test failed: {str(e)}")
        
        print("\nHost APIs:")
        print("-" * 60)
        apis = sd.query_hostapis()
        for i, api in enumerate(apis):
            print(f"{i}: {api['name']}")
            
    except Exception as e:
        print(f"\n❌ Error during device query: {str(e)}")
        print(f"Error type: {type(e).__name__}")

if __name__ == "__main__":
    print("Running audio device diagnostics...")
    test_audio_devices()