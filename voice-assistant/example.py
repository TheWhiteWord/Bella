from pipeline.manager import PipelineManager
import time

def main():
    # Initialize the pipeline
    pipeline = PipelineManager()
    
    print("\nPhi-4 Multimodal Voice Assistant Test")
    print("=====================================")
    print("This is a basic test of audio-to-text conversion.")
    print("Press Enter to start recording...")
    print("Press Enter again to stop recording.\n")
    
    try:
        while True:
            input("[Press Enter to start] ")
            
            # Record and process voice input
            print("\nProcessing...")
            response = pipeline.process_voice_input(save_audio=True)
            
            if response:
                print("\nPhi-4 Response:", response)
            print("\n" + "="*50 + "\n")
            
            try_again = input("\nTry another? (y/n): ").lower()
            if try_again != 'y':
                break
            
    except Exception as e:
        print(f"\nError in demo: {str(e)}")
    finally:
        del pipeline

if __name__ == "__main__":
    main()