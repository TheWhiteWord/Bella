# Model Presets and INI Configuration

The `models.ini` file allows you to define specific hardware and sampling parameters for each model, ensuring that a small summarizer doesn't waste VRAM while a large writer gets the context it needs.

## File Structure

### Global Defaults (`[*]`)
Settings under this section apply to every model loaded by the router.

```ini
[*]
n-gpu-layers = 99
flash-attn = on
cache-type-k = q8_0
cache-type-v = q8_0
```

### Model-Specific Sections
The section name must match the filename (without `.gguf`) or the folder name.

```ini
[DeepHermes_8B]
ctx-size = 16384
parallel = 4
temp = 0.8

[Summarizer-3B]
ctx-size = 2048
parallel = 1
```

## Common Preset Keys

| Key | CLI Equivalent | Purpose |
| :--- | :--- | :--- |
| `ctx-size` | `-c` | Sets the context window for this model. |
| `parallel` | `-np` | Sets the number of concurrent agents. |
| `n-gpu-layers` | `-ngl` | Controls GPU offloading. |
| `cache-type-k` | `--cache-type-k` | Quantizes K-cache (e.g., `q8_0`, `q4_0`). |
| `cache-type-v` | `--cache-type-v` | Quantizes V-cache (e.g., `q8_0`, `q4_0`). |
| `load-on-startup` | N/A | If `true`, loads the model immediately. |

## Precedence Rules
1.  **Command-line arguments** passed to the router (highest).
2.  **Model-specific options** in the `.ini`.
3.  **Global options** (`[*]`) in the `.ini`.
