# Server Configuration and Router Mode

The `llama-server` is the primary interface for hosting models. For multi-project environments, **Router Mode** is the recommended deployment strategy.

## Starting the Router Server

To start the server in Router Mode, point it to a directory of models without specifying a single `-m` flag.

```bash
./llama-server \
    --models-dir /path/to/your/models \
    --models-preset /path/to/models.ini \
    --host 0.0.0.0 \
    --port 8080
```

## Key Server Arguments

| Argument | Description |
| :--- | :--- |
| `--models-dir` | Directory containing `.gguf` files. Scans 1 level deep. |
| `--models-preset` | Path to an `.ini` file defining specific model settings. |
| `--models-max` | Max number of different models to keep in VRAM (default: 4). |
| `--parallel` / `-np` | Number of concurrent request slots per model instance. |
| `--cont-batching` | **CRITICAL:** Enables simultaneous processing of multiple requests. |
| `--ctx-size` / `-c` | Total context tokens allocated (divided among slots). |
| `--n-gpu-layers` / `-ngl` | Number of layers to offload to GPU (99 = all). |

## Router Mode Behavior

1.  **On-Demand Loading:** Models are loaded into VRAM only when an API request mentions their name.
2.  **LRU Eviction:** If VRAM is full, the server unloads the "Least Recently Used" model to make room for a new one.
3.  **Inheritance:** Child model processes inherit global flags (like `--flash-attn`) from the main router process unless overridden in `models.ini`.
