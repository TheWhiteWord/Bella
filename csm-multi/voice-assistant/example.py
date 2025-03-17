from pipeline.manager import PipelineManager
import asyncio

async def main_async():
    """Run the voice assistant demo."""
    # Initialize the pipeline
    print("\nInitializing Voice Assistant...")
    pipeline = PipelineManager()
    
    print("\nVoice Assistant")
    print("==============")
    print("Press Enter to start recording, and Enter again to stop.\n")
    
    try:
        while True:
            # Record voice input
            print("Ready to record!")
            input("[Press Enter to start recording] ")
            
            print("\nRecording... Press Enter to stop")
            audio_data = await pipeline.process_voice_input_async(save_audio=True)
            
            if audio_data is not None:
                print("\nRecording complete!")
                print("\nAudio data captured successfully")
                # Note: Additional processing can be added here
                
            else:
                print("\nError: Failed to record audio")
                
            print("\n" + "="*50 + "\n")
            
            # Ask to try again
            try_again = input("Would you like to record again? (y/n): ").lower()
            if try_again != 'y':
                break
            print("\n")
            
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError in demo: {str(e)}")
    finally:
        print("\nCleaning up resources...")
        del pipeline
        print("Goodbye!")

def main():
    """Synchronous entry point"""
    asyncio.run(main_async())

if __name__ == "__main__":
    main()