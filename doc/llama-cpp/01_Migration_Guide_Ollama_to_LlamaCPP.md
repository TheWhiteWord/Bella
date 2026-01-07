# Migration Guide: Ollama to llama.cpp

This guide outlines the architectural advantages and practical steps for migrating local LLM projects from the Ollama wrapper to the native llama.cpp engine.

## Why Migrate?

| Feature | Ollama | llama.cpp (Native) |
| :--- | :--- | :--- |
| **Architecture** | Go Wrapper / C++ Engine | Pure C++ (Zero Overhead) |
| **Concurrency** | Limited/Sequential | Native "Slots" system for true parallelism |
| **Resource Control** | Abstracted/Automated | Granular (VRAM, Threads, KV Cache) |
| **Performance** | 13% - 80% slower due to layers | Maximum hardware utilization |
| **API** | Custom & OpenAI-Compatible | OpenAI & Anthropic-Compatible |

## Core Concepts for Migration

### 1. From "Modelfiles" to "Presets"
In Ollama, you define model behavior in a `Modelfile`. In llama.cpp, you use a `models.ini` file to define presets for the Router Server.

### 2. From `ollama run` to `llama-cli`
For interactive terminal use, `llama-cli` provides a similar experience but with more control over sampling.

### 3. The "Slot" System
Ollama handles multiple requests by queuing or attempting to load multiple model instances. llama.cpp uses **Slots** (`--parallel`), which partitions the KV cache of a single model instance to handle multiple agents simultaneously in a single batch.

## Migration Workflow

1.  **Download GGUF:** Obtain the GGUF version of your model from Hugging Face.
2.  **Configure Presets:** Add the model to your `models.ini` with tailored `ctx-size` and `parallel` settings.
3.  **Update API Base URL:** Change your code's API endpoint from `http://localhost:11434/v1` to `http://localhost:8080/v1`.
4.  **Enable Continuous Batching:** Ensure `--cont-batching` is active to allow agents to generate in parallel.
