from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment, CACHE_DIR
import torchaudio
import os

print("Loading model...")
model_path = os.path.join(CACHE_DIR, "ckpt.pt")
generator = load_csm_1b(model_path, "cuda")
print("Model loaded successfully!")

reference_path = "reference.wav"
context_segments = []
conversation_history = []

if os.path.exists(reference_path):
    print(f"Using {reference_path} as reference audio")
    def load_audio(audio_path):
        audio_tensor, sample_rate = torchaudio.load(audio_path)
        audio_tensor = torchaudio.functional.resample(
            audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
        )
        return audio_tensor
        
    reference_audio = load_audio(reference_path)
    reference_text = "This is Sesame. I say hi. And pwee. And more!"
    context_segments = [
        Segment(text=reference_text, speaker=0, audio=reference_audio)
    ]
    conversation_history.append({"role": "assistant", "content": reference_text})
    print("Reference audio loaded and added to context")
else:
    print("No reference audio.wav found, starting without context")

while True:
    try:
        text = input("Enter text: ")
        if text == "$CLEAR$":
            print("Clearing context...")
            context_segments = []
            conversation_history = []
            if os.path.exists(reference_path):
                reference_audio = load_audio(reference_path)
                context_segments = [
                    Segment(text=reference_text, speaker=0, audio=reference_audio)
                ]
                conversation_history.append({"role": "assistant", "content": reference_text})
            print("Cleared.")
            continue

        print(f"Generating audio for: '{text}'")
        audio = generator.generate(
            text=text,
            speaker=0,
            context=context_segments,
            max_audio_length_ms=25_000,
        )
        
        torchaudio.save("output.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        context_segments.append(
            Segment(text=text, speaker=0, audio=audio)
        )
        
        if len(context_segments) > 5:
            context_segments = context_segments[-5:]
        
        conversation_history.append({"role": "user", "content": text})
        
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Try again.")
