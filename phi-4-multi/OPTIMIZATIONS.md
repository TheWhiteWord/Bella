# Phi-4-MM Model Optimizations

This document outlines the key optimizations and fixes applied to make the Phi-4-MM model work efficiently, particularly for handling multiple images and mixed-modal inputs.

## Memory Management Optimizations

### 1. CUDA Memory Settings
```python
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
    'max_split_size_mb:128,'  # Prevent splitting large blocks
    'garbage_collection_threshold:0.8,'  # Start GC when 80% memory used
    'max_non_split_rounding_mb:512'  # Allow more flexible block reuse
)
```

### 2. Model Initialization
```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use fp16 for reduced memory usage
    _attn_implementation='flash_attention_2',
    use_cache=False,  # Disable KV cache for memory savings
    device_map='auto',
    low_cpu_mem_usage=True,
).cuda()
```

## Multi-Image Processing Optimizations

### 1. Sequential Processing
- Images are processed one at a time rather than in batch to avoid OOM errors
- Memory is cleared between each image processing step
- Results are concatenated after individual processing

### 2. Controlled GPU Memory Usage
```python
def process_images_sequentially(images, prompt):
    all_responses = []
    
    for i, image in enumerate(images):
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
        
        # Process single image
        try:
            inputs = processor(text=single_prompt, images=[image], return_tensors='pt')
            cuda_inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            with torch.inference_mode():
                generate_ids = model.generate(
                    **cuda_inputs,
                    max_new_tokens=250,  # Reduced token limit
                    generation_config=generation_config,
                )
                
            # Process response and cleanup
            response = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            all_responses.append(response)
            
        finally:
            # Clear CUDA cache
            if 'cuda_inputs' in locals(): del cuda_inputs
            if 'generate_ids' in locals(): del generate_ids
            torch.cuda.empty_cache()
            gc.collect()
```

## Error Handling and Debugging

### 1. CUDA Synchronization
- Added `CUDA_LAUNCH_BLOCKING=1` for synchronous operations
- Helps identify the exact location of CUDA errors

### 2. Gradient Management
- Model set to evaluation mode with `model.eval()`
- Using `torch.inference_mode()` for inference
- Disabled gradient calculation to save memory

## Image Processing Optimizations

### 1. Memory-Efficient Image Loading
- Images are processed one at a time
- Tensors are moved to GPU only when needed
- Explicit cleanup after processing

### 2. Batch Size Control
- Default batch size reduced to 1 for image processing
- Dynamic batch sizing based on available memory
- Sequential processing for multiple images

## Mixed Modal Processing

### 1. Input Mode Handling
```python
if len(image_inputs) > 0 and len(audio_inputs) > 0:
    input_mode = InputMode.VISION_SPEECH
elif len(image_inputs) > 0:
    input_mode = InputMode.VISION
elif len(audio_inputs) > 0:
    input_mode = InputMode.SPEECH
else:
    input_mode = InputMode.LANGUAGE
```

### 2. Resource Management
- Proper cleanup between different modal inputs
- Memory monitoring for mixed inputs
- Sequential processing for multi-modal inputs

## Future Optimizations

1. **Dynamic Memory Management**
   - Implement dynamic batch sizing based on available GPU memory
   - Add memory monitoring for automatic resource adjustment

2. **Performance Improvements**
   - Consider implementing gradient checkpointing for larger models
   - Explore quantization options for reduced memory usage
   - Investigate optimal attention implementations

3. **Error Recovery**
   - Add automatic retry mechanisms for OOM errors
   - Implement graceful degradation for resource-intensive operations

## Configuration Notes

For optimal performance, configure your environment with:

```bash
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:128,garbage_collection_threshold:0.8,max_non_split_rounding_mb:512'
```

Remember to monitor GPU memory usage when processing multiple images or running mixed-modal operations. Consider using smaller batch sizes or reducing model precision if memory issues persist.