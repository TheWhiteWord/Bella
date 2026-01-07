# llama.cpp Documentation Index

This directory contains comprehensive guides for migrating from Ollama to llama.cpp, configuring high-performance servers, and optimizing for multi-agent creative writing workflows.

## 📚 Documentation Guides

### 1. [Migration Guide: Ollama to llama.cpp](01_Migration_Guide_Ollama_to_LlamaCPP.md)
*   **Focus:** Architectural differences, core concept mapping, and the basic migration workflow.
*   **Use when:** Transitioning an existing project or understanding why native llama.cpp is superior for agents.

### 2. [Server Configuration and Router Mode](02_Server_Configuration_and_Router_Mode.md)
*   **Focus:** `llama-server` arguments, experimental Router Mode, and on-demand model loading.
*   **Use when:** Setting up your background service or managing multiple models on one port.

### 3. [Model Presets and INI Configuration](03_Model_Presets_and_INI_Configuration.md)
*   **Focus:** Using `models.ini` to tailor VRAM, context, and sampling settings per model.
*   **Use when:** Fine-tuning hardware allocation for different agents (e.g., Summarizer vs. Writer).

### 4. [Concurrency and the Slot System](04_Concurrency_and_Slot_System.md)
*   **Focus:** Understanding `--parallel` slots, continuous batching, and the $C_{total}$ formula.
*   **Use when:** Designing systems where multiple agents need to generate text simultaneously.

### 5. [Hardware Optimization (CUDA)](05_Hardware_Optimization_CUDA.md)
*   **Focus:** Flash Attention, KV Cache Quantization, and GPU offloading flags.
*   **Use when:** Maximizing performance on NVIDIA RTX 40-series hardware.

### 6. [API Integration (Python)](06_API_Integration_Python.md)
*   **Focus:** OpenAI-compatible client setup, handling load-time timeouts, and streaming.
*   **Use when:** Writing or refactoring Python code to communicate with the server.

### 7. [Creative Writing Optimization](07_Creative_Writing_Optimization.md)
*   **Focus:** Sampling parameters (Min-P, Temp), long-context management, and story-writing workflows.
*   **Use when:** Tuning models for fictional prose, narrative depth, and stylistic flair.

---

## 🚀 Quick Start Commands

*   **Build llama.cpp:** `bash build_llama.sh`
*   **Start Router Server:** `bash start_server.sh`
*   **Interactive Chat:** `bash chat.sh`
*   **Test API Connection:** `python3 test_api.py`
