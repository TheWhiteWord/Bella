# Hardware Optimization (NVIDIA CUDA)

To achieve maximum performance on NVIDIA hardware, specific build and runtime flags must be utilized.

## Build-Time Optimizations

When compiling llama.cpp, ensure the following CMake flags are used:

*   `-DGGML_CUDA=ON`: Enables the CUDA backend.
*   `-DGGML_CUDA_GRAPHS=ON`: Reduces CPU overhead by capturing sequences of CUDA kernels.
*   `-DGGML_CUDA_FA_ALL_QUANTS=ON`: Enables Flash Attention for all quantization types (not just F16).

## Runtime Optimizations

### 1. Flash Attention (`--flash-attn on`)
Drastically reduces memory usage and increases speed for long-context sequences (8k+ tokens).

### 2. KV Cache Quantization
Reduces the VRAM footprint of the "memory" of the conversation.
*   `--cache-type-k q8_0`
*   `--cache-type-v q8_0`
*   *Note: Using `q4_0` saves even more VRAM but may slightly impact long-term coherence.*

### 3. GPU Offloading (`--n-gpu-layers 99`)
Ensures all model tensors are stored in VRAM. If a model is too large, reduce this number until it fits.

### 4. CPU Threading
Match `-t` (threads) to your physical CPU cores (not logical threads) for the best performance during the "Prompt Processing" phase.
