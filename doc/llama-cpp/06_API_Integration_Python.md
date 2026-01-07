# API Integration (Python)

llama.cpp provides a REST API that is fully compatible with the OpenAI Chat Completions standard.

## Basic Setup

Use the standard `openai` Python library.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none" # Required but ignored
)
```

## Handling Router Mode Latency

The first time a model is requested in Router Mode, it must load from disk. Increase your timeout to prevent crashes.

```python
completion = client.chat.completions.create(
    model="DeepHermes_8B",
    messages=[{"role": "user", "content": "Hello!"}],
    timeout=120.0 # 2 minutes for the first load
)
```

## Streaming Responses

Streaming is supported via the standard protocol.

```python
stream = client.chat.completions.create(
    model="DeepHermes_8B",
    messages=[{"role": "user", "content": "Write a story..."}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Endpoints

*   `/v1/chat/completions`: Standard chat interface.
*   `/v1/completions`: Legacy completion interface.
*   `/slots`: (llama.cpp specific) Returns the status of all parallel slots.
*   `/health`: Returns server status.
