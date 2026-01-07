# Concurrency and the Slot System

llama.cpp uses a "Slot" architecture to handle multiple agents or users simultaneously without the overhead of loading multiple model instances.

## How Slots Work

When a model is loaded with `--parallel N`, the total context window (`--ctx-size`) is partitioned into `N` discrete units called **Slots**.

**The Formula:**
$$C_{total} = N_{slots} \times C_{slot}$$

*   If you set `-c 16384` and `-np 4`, each agent gets **4,096 tokens** of context.
*   If an agent exceeds its slot size, the server will use "Context Shifting" to discard old tokens.

## Continuous Batching

Without continuous batching, the server processes requests in groups. If one agent is slow, others must wait.

**With `--cont-batching`:**
*   New requests are inserted into the inference cycle as soon as a slot is free.
*   Multiple agents generate tokens in the same GPU forward pass.
*   This maximizes GPU utilization and minimizes "Time to First Token" (TTFT).

## Multi-Agent Best Practices

1.  **Calculate VRAM:** Ensure your `ctx-size` doesn't push the model out of VRAM.
2.  **Match Slots to Agents:** If your system has 3 agents (Plotter, Writer, Editor), set `--parallel 3` or higher.
3.  **Use KV Quantization:** Use `--cache-type-k q8_0` to fit larger context windows or more slots into the same VRAM.
