"""Basic recorder test script."""
import os
import asyncio
from recorder import AudioRecorder
import sounddevice as sd

async def main():
    """Test basic recording functionality."""
    print("\nBasic Recording Test")
    print("==================")
    
    # Initialize recorder
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")
    recorder = AudioRecorder(config_path)
    
    try:
        while True:
            print("\nReady to record!")
            print("Press Enter to START recording...")
            
            # Record audio
            audio_data = await recorder.record_async()
            
            if audio_data is not None:
                print("\nPlayback of recording...")
                sd.play(audio_data, recorder.sample_rate)
                sd.wait()
                print("Playback complete!")
            else:
                print("\nNo audio was recorded!")
            
            # Ask to try again
            again = input("\nWould you like to record again? (y/n): ").lower().strip()
            if again != 'y':
                break
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nError during test: {str(e)}")
    finally:
        print("\nTest complete!")

if __name__ == "__main__":
    asyncio.run(main())