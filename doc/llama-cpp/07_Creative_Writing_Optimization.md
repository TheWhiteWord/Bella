# Creative Writing Optimization

Fictional prose generation requires different sampling strategies than technical or chat-based tasks.

## Recommended Sampling Parameters

| Parameter | Value | Why? |
| :--- | :--- | :--- |
| **Temperature** | `0.8 - 1.2` | Higher values increase stylistic flair. |
| **Min-P** | `0.05 - 0.1` | Removes low-probability "noise" without killing creativity. |
| **Top-P** | `0.9 - 0.95` | Standard nucleus sampling. |
| **Repetition Penalty** | `1.1 - 1.15` | Prevents the model from getting stuck in loops. |

## Long-Context Management

For novels or long stories, context management is vital.

1.  **Context Shifting:** llama.cpp automatically "shifts" the window when full.
2.  **Keep Tokens (`--n-keep`):** Use this to ensure the "System Prompt" or "Story Bible" is never forgotten during a shift.
3.  **KV Quantization:** Always use `q8_0` or `q4_0` for the cache to allow for 16k+ context windows on consumer GPUs.

## Multi-Agent Prose Workflows

A common high-performance setup for writing:
*   **Agent 1 (Plotter):** Small model (3B-7B), low context, low temperature.
*   **Agent 2 (Writer):** Large model (8B-30B), high context, high temperature.
*   **Agent 3 (Editor):** Large model, high context, medium temperature.

By using **Router Mode** and **Slots**, all three can work on the same GPU simultaneously.
