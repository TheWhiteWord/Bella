import os
import torch
import torch.amp  # Explicitly import torch.amp
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import gc
import warnings
import torchaudio

class Phi4Handler:
    def __init__(self, config_path="../config/settings.yaml"):
        # Disable all warnings initially
        warnings.filterwarnings('ignore')
        
        # Environment settings for warnings
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
        os.environ['PYTORCH_NO_WARN_ON_EXTENSION_NOT_FOUND'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Enhanced memory management settings
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            'max_split_size_mb:128,'
            'garbage_collection_threshold:0.8,'
            'max_non_split_rounding_mb:512'
        )
        
        self.model_path = "../phi-4-multi/models/microsoft--Phi-4-multimodal-instruct"
        self.load_model()
        
    def load_model(self):
        """Initialize the Phi-4 model with CPU offloading"""
        print("Loading Phi-4 Multimodal model...")
        try:
            # Clean up any existing CUDA memory
            torch.cuda.empty_cache()
            gc.collect()
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # First load model to CPU
            print("Loading model to CPU first...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="cpu",
                torch_dtype=torch.float16,  # Use fp16 for reduced memory usage
                trust_remote_code=True,
                _attn_implementation='flash_attention_2',
                use_cache=False,  # Disable KV cache for memory savings
                low_cpu_mem_usage=True,
            )
            
            print("Moving model to CUDA...")
            # Now move to CUDA in a controlled way
            self.model = self.model.to("cuda")
            
            # Set model to evaluation mode and disable gradients
            self.model.eval()
            torch.set_grad_enabled(False)
            
            # Load generation config
            self.generation_config = GenerationConfig.from_pretrained(self.model_path)
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
            
    def _preprocess_audio(self, audio_data, sample_rate):
        """Preprocess audio for better transcription"""
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(dim=0)
            
        # Normalize audio
        if audio_data.abs().max() > 0:
            audio_data = audio_data / audio_data.abs().max()
            
        # Apply noise gate to remove silence
        noise_threshold = 0.02
        audio_data[audio_data.abs() < noise_threshold] = 0
            
        # Trim silence from start and end
        start = 0
        end = len(audio_data)
        for i in range(len(audio_data)):
            if audio_data[i].abs() > noise_threshold:
                start = max(0, i - int(0.1 * sample_rate))  # Keep 0.1s before speech
                break
        for i in range(len(audio_data) - 1, -1, -1):
            if audio_data[i].abs() > noise_threshold:
                end = min(len(audio_data), i + int(0.1 * sample_rate))  # Keep 0.1s after speech
                break
        audio_data = audio_data[start:end]
        
        return audio_data
            
    def process_audio(self, audio_data, sample_rate):
        """Process audio input with improved preprocessing"""
        try:
            # Clean up memory before processing
            torch.cuda.empty_cache()
            gc.collect()
            
            # Preprocess audio
            audio_data = self._preprocess_audio(audio_data, sample_rate)
            
            # Use more explicit prompt
            user_prompt = '<|user|>'
            assistant_prompt = '<|assistant|>'
            prompt_suffix = '<|end|>'
            speech_prompt = "Transcribe this audio to text accurately. Write out all spoken words."
            prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'
            
            # Prepare inputs with memory optimization
            with torch.amp.autocast(device_type='cuda'):
                inputs = self.processor(
                    text=prompt,
                    audios=[(audio_data, sample_rate)],
                    return_tensors='pt'
                ).to('cuda')
                
                # Generate response with exact same settings as test
                generate_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    generation_config=self.generation_config,
                )
                
                # Remove prompt from generated output
                generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                
                # Decode response
                response = self.processor.batch_decode(
                    generate_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                
            return response.strip()
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return None
            
        finally:
            if 'inputs' in locals():
                del inputs
            if 'generate_ids' in locals():
                del generate_ids
            torch.cuda.empty_cache()
            gc.collect()
            
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            # Final cleanup
            if hasattr(self, 'model'):
                self.model.to('cpu')
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            torch.cuda.empty_cache()
            gc.collect()
        except:
            pass