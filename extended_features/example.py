import os
from pathlib import Path
from generator import load_csm_1b
from extended_features import MultiSpeakerGenerator, ConversationManager

def main():
    # Load the base generator
    model_path = os.path.join("models", "ckpt.pt")
    base_generator = load_csm_1b(model_path, "cuda")
    print("Model loaded successfully!")

    # Create multi-speaker generator
    multi_generator = MultiSpeakerGenerator(base_generator)
    
    # Create conversation manager
    conversation = ConversationManager(multi_generator)

    print("\nMulti-speaker voice generation demo")
    print("Special commands:")
    print("  $CLEAR$ - Clear conversation context")
    print("  $SWAP$ - Switch to next speaker")
    print("  $BACK$ - Switch to previous speaker")
    print("Use || to separate different speakers in your input")
    print("Example: 'Hello there!||Hi, how are you?||I'm doing great!'")
    print("\nEnter 'quit' to exit\n")

    while True:
        try:
            text = input("Enter text: ").strip()
            
            if text.lower() == 'quit':
                break
                
            if not text:
                continue

            output_file = conversation.process_input(
                text,
                output_filename="output.wav"
            )
            
            if output_file:
                print(f"Audio generated and saved to: {output_file}")
                
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()