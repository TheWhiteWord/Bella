import os
import io
import requests
import torch
import torch.amp  # Explicitly import torch.amp
from PIL import Image
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import gc
from urllib.request import urlopen
import warnings
import torchvision

# Disable all warnings initially
warnings.filterwarnings('ignore')

# Environment settings for warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ['PYTORCH_NO_WARN_ON_EXTENSION_NOT_FOUND'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Enhanced memory management settings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'max_split_size_mb:128,'  # Prevent splitting large blocks
    'garbage_collection_threshold:0.8,'  # Start GC when 80% memory used
    'max_non_split_rounding_mb:512'  # Allow more flexible block reuse
)

def get_model_path():
    """Locate the Phi-4 model in the local structure"""
    CACHE_DIR = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(CACHE_DIR, "microsoft--Phi-4-multimodal-instruct")
    
    if os.path.exists(model_path):
        return model_path
    return None

def load_audio(url):
    """Load and process audio file with error handling"""
    try:
        audio_data = urlopen(url).read()
        audio, samplerate = sf.read(io.BytesIO(audio_data))
        return audio, samplerate
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

def process_image(processor, model, image, prompt, max_new_tokens=500):
    """Process single image with memory optimization"""
    torch.cuda.empty_cache()
    gc.collect()
    
    try:
        inputs = processor(text=prompt, images=image, return_tensors='pt')
        cuda_inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda'):  # Updated autocast usage
            generate_ids = model.generate(
                **cuda_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=0.9,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return response
        
    except torch.cuda.OutOfMemoryError:
        print("Warning: CUDA out of memory, attempting recovery...")
        torch.cuda.empty_cache()
        gc.collect()
        return "Error: Out of memory"
        
    finally:
        if 'cuda_inputs' in locals():
            del cuda_inputs
        if 'generate_ids' in locals():
            del generate_ids
        torch.cuda.empty_cache()
        gc.collect()

def main():
    try:
        # Get model path
        model_path = get_model_path()
        if not model_path:
            raise ValueError("Could not find Phi-4 model in the models directory")

        print(f"Loading model from: {model_path}")

        # Initialize processor and model with optimized settings
        processor = AutoProcessor.from_pretrained(
            model_path, 
            trust_remote_code=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.float16,  # Use fp16 for reduced memory usage
            trust_remote_code=True,
            _attn_implementation='flash_attention_2',
            use_cache=False,  # Disable KV cache for memory savings
            low_cpu_mem_usage=True,
        ).cuda()

        # Set model to evaluation mode and disable gradients
        model.eval()
        torch.set_grad_enabled(False)

        # Load generation config
        generation_config = GenerationConfig.from_pretrained(model_path)

        # Define prompt structure
        user_prompt = '<|user|>'
        assistant_prompt = '<|assistant|>'
        prompt_suffix = '<|end|>'

        # Part 1: Image Processing
        print("\n--- IMAGE PROCESSING ---")
        image_url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
        prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
        print(f'>>> Prompt\n{prompt}')

        # Download and process image
        image = Image.open(requests.get(image_url, stream=True).raw)
        response = process_image(processor, model, image, prompt)
        print(f'>>> Response\n{response}')

        # Part 2: Audio Processing
        print("\n--- AUDIO PROCESSING ---")
        audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
        speech_prompt = "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the original transcript and the translation."
        prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
        print(f'>>> Prompt\n{prompt}')

        # Process audio
        audio, samplerate = load_audio(audio_url)
        if audio is not None:
            try:
                with torch.amp.autocast(device_type='cuda'):  # Updated autocast usage
                    inputs = processor(text=prompt, audios=[(audio, samplerate)], return_tensors='pt').to('cuda:0')
                    generate_ids = model.generate(
                        **inputs,
                        max_new_tokens=1000,
                        generation_config=generation_config,
                    )
                    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                    response = processor.batch_decode(
                        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
                    print(f'>>> Response\n{response}')
            except Exception as e:
                print(f"Error processing audio: {e}")
            finally:
                torch.cuda.empty_cache()
                gc.collect()
        else:
            print("Failed to load audio file")

    except Exception as e:
        print(f"Error in main execution: {e}")
    finally:
        # Final cleanup
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()