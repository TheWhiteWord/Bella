import torch
import time
import sys
import os
from pathlib import Path

# Add parent directory to path to import from utils
sys.path.append(str(Path(__file__).parents[2]))
from utils.generator import load_csm_1b

def main():
    # Enable optimal CUDA settings
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
        
    print("\nCUDA Settings:")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading CSM model...")
    generator = None
    try:
        with torch.cuda.device(device):
            generator = load_csm_1b(device=device)
            # Ensure model is loaded before timing
            torch.cuda.synchronize()
    
        # Test texts of different lengths
        test_texts = [
            "This is a short test.",
            "This is a medium length test of the optimized TTS system.",
            "This is a longer test of the text to speech system with optimizations enabled. Let's see how it performs with more content.",
        ]
        
        print("\nWarming up model...")
        # Warmup run
        with torch.inference_mode():
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                generator.generate(
                    text=test_texts[0],
                    speaker=0,
                    context=[],
                    temperature=0.7,
                )
        torch.cuda.synchronize()
        
        print("\nRunning speed tests...")
        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}: Length {len(text)} chars")
            torch.cuda.empty_cache()
            
            # Run multiple iterations to get average speed
            times = []
            num_iterations = 3
            
            for j in range(num_iterations):
                torch.cuda.synchronize()  # Ensure previous iteration is complete
                start_time = time.time()
                
                with torch.inference_mode():
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        generator.generate(
                            text=text,
                            speaker=0,
                            context=[],
                            temperature=0.7,
                        )
                
                torch.cuda.synchronize()  # Wait for generation to complete
                end_time = time.time()
                times.append(end_time - start_time)
                print(f"  Iteration {j+1}/{num_iterations}: {times[-1]:.2f}s")
                
                # Memory cleanup after each iteration
                torch.cuda.empty_cache()
            
            avg_time = sum(times) / len(times)
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Memory Usage: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            print(f"  Max Memory: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
            print(f"  Chars/second: {len(text)/avg_time:.1f}")
    
    finally:
        # Memory cleanup
        if generator is not None:
            generator._model.cleanup_tensors()
            del generator
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()