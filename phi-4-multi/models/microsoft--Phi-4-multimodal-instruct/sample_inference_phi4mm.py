import os
import requests
import torch
from PIL import Image
import soundfile
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
import gc

# Set CUDA launch blocking for debugging
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Enhanced memory management settings
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'max_split_size_mb:128,'
    'garbage_collection_threshold:0.8,'
    'max_non_split_rounding_mb:512'
)

def process_images_sequentially(images, prompt):
    all_responses = []
    
    for i, image in enumerate(images):
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create single-image prompt
        single_prompt = f'{user_prompt}<|image_{i+1}|>Describe this slide.{prompt_suffix}{assistant_prompt}'
        
        try:
            inputs = processor(text=single_prompt, images=[image], return_tensors='pt')
            
            # Move inputs to GPU in a controlled manner
            cuda_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    cuda_inputs[k] = v.cuda()
                else:
                    cuda_inputs[k] = v
            
            with torch.inference_mode():
                generate_ids = model.generate(
                    **cuda_inputs,
                    max_new_tokens=250,
                    generation_config=generation_config,
                )
                
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response = processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            all_responses.append(response)
            
        except Exception as e:
            print(f"Warning: Error processing image {i+1}: {str(e)}")
            torch.cuda.empty_cache()
            gc.collect()
            continue
            
        finally:
            # Clear CUDA cache
            if 'cuda_inputs' in locals():
                del cuda_inputs
            if 'generate_ids' in locals():
                del generate_ids
            torch.cuda.empty_cache()
            gc.collect()
    
    # Combine responses
    final_response = " Slide-by-slide summary: " + " ".join(all_responses)
    return final_response

model_path = './'

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
print(processor.tokenizer)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use fp16 for reduced memory usage
    _attn_implementation='flash_attention_2',
    use_cache=False,
    device_map='auto',
    low_cpu_mem_usage=True,
).cuda()

model.eval()  # Ensure model is in eval mode

print("model.config._attn_implementation:", model.config._attn_implementation)

generation_config = GenerationConfig.from_pretrained(model_path, 'generation_config.json')

user_prompt = '<|user|>'
assistant_prompt = '<|assistant|>'
prompt_suffix = '<|end|>'
 
#################################################### text-only ####################################################
prompt = f'{user_prompt}what is the answer for 1+1? Explain it.{prompt_suffix}{assistant_prompt}'
print(f'>>> Prompt\n{prompt}')
inputs = processor(prompt, images=None, return_tensors='pt').to('cuda:0')

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(f'>>> Response\n{response}')

#################################################### vision (single-turn) ####################################################
# single-image prompt
prompt = f'{user_prompt}<|image_1|>What is shown in this image?{prompt_suffix}{assistant_prompt}'
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
print(f'>>> Prompt\n{prompt}')
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

#################################################### vision (multi-turn) ####################################################
# chat template
chat = [
    {'role': 'user', 'content': f'<|image_1|>What is shown in this image?'},
    {
        'role': 'assistant',
        'content': "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
    },
    {'role': 'user', 'content': 'What is so special about this image'},
]
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith('<|endoftext|>'):
    prompt = prompt.rstrip('<|endoftext|>')

print(f'>>> Prompt\n{prompt}')

inputs = processor(prompt, [image], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

########################### vision (multi-frame) ################################
images = []
placeholder = ''
for i in range(1, 5):
    url = f'https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg'
    images.append(Image.open(requests.get(url, stream=True).raw))
    placeholder += f'<|image_{i}|>'

messages = [
    {'role': 'user', 'content': placeholder + 'Summarize the deck of slides.'},
]

prompt = processor.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f'>>> Prompt\n{prompt}')

# Process images sequentially
response = process_images_sequentially(images, prompt)
print(f'>>> Response\n{response}')

# NOTE: Please prepare the audio file 'examples/what_is_the_traffic_sign_in_the_image.wav'
#       and audio file 'examples/what_is_shown_in_this_image.wav' before running the following code
#       Basically you can record your own voice for the question "What is the traffic sign in the image?" in "examples/what_is_the_traffic_sign_in_the_image.wav".
#       And you can record your own voice for the question "What is shown in this image?" in "examples/what_is_shown_in_this_image.wav".

AUDIO_FILE_1 = 'examples/what_is_the_traffic_sign_in_the_image.wav'
AUDIO_FILE_2 = 'examples/what_is_shown_in_this_image.wav'

if not os.path.exists(AUDIO_FILE_1):
    raise FileNotFoundError(f'Please prepare the audio file {AUDIO_FILE_1} before running the following code.')
########################## vision-speech ################################
prompt = f'{user_prompt}<|image_1|><|audio_1|>{prompt_suffix}{assistant_prompt}'
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
print(f'>>> Prompt\n{prompt}')
image = Image.open(requests.get(url, stream=True).raw)
audio = soundfile.read(AUDIO_FILE_1)
inputs = processor(text=prompt, images=[image], audios=[audio], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

########################## speech only ################################
speech_prompt = "Based on the attached audio, generate a comprehensive text transcription of the spoken content."
prompt = f'{user_prompt}<|audio_1|>{speech_prompt}{prompt_suffix}{assistant_prompt}'

print(f'>>> Prompt\n{prompt}')
audio = soundfile.read(AUDIO_FILE_1)
inputs = processor(text=prompt, audios=[audio], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

if not os.path.exists(AUDIO_FILE_2):
    raise FileNotFoundError(f'Please prepare the audio file {AUDIO_FILE_2} before running the following code.')
########################### speech only (multi-turn) ################################
audio_1 = soundfile.read(AUDIO_FILE_2)
audio_2 = soundfile.read(AUDIO_FILE_1)
chat = [
    {'role': 'user', 'content': f'<|audio_1|>Based on the attached audio, generate a comprehensive text transcription of the spoken content.'},
    {
        'role': 'assistant',
        'content': "What is shown in this image.",
    },
    {'role': 'user', 'content': f'<|audio_2|>Based on the attached audio, generate a comprehensive text transcription of the spoken content.'},
]
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith('<|endoftext|>'):
    prompt = prompt.rstrip('<|endoftext|>')

print(f'>>> Prompt\n{prompt}')

inputs = processor(text=prompt, audios=[audio_1, audio_2], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')

#################################################### vision-speech (multi-turn) ####################################################
# chat template
audio_1 = soundfile.read(AUDIO_FILE_2)
audio_2 = soundfile.read(AUDIO_FILE_1)
chat = [
    {'role': 'user', 'content': f'<|image_1|><|audio_1|>'},
    {
        'role': 'assistant',
        'content': "The image depicts a street scene with a prominent red stop sign in the foreground. The background showcases a building with traditional Chinese architecture, characterized by its red roof and ornate decorations. There are also several statues of lions, which are common in Chinese culture, positioned in front of the building. The street is lined with various shops and businesses, and there's a car passing by.",
    },
    {'role': 'user', 'content': f'<|audio_2|>'},
]
url = 'https://www.ilankelman.org/stopsigns/australia.jpg'
image = Image.open(requests.get(url, stream=True).raw)
prompt = processor.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
# need to remove last <|endoftext|> if it is there, which is used for training, not inference. For training, make sure to add <|endoftext|> in the end.
if prompt.endswith('<|endoftext|>'):
    prompt = prompt.rstrip('<|endoftext|>')

print(f'>>> Prompt\n{prompt}')

inputs = processor(text=prompt, images=[image], audios=[audio_1, audio_2], return_tensors='pt').to('cuda:0')
generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    generation_config=generation_config,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1] :]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'>>> Response\n{response}')
