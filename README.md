# AiAssist Python SDK

Simple, lightweight Python client for the AiAssist AI API.

**Only dependency: httpx** - No heavy frameworks, no Redis, no server components.

## Installation

```bash
pip install aiassist
```

## Quick Start

### Async Client

```python
from aiassist import AiAssistClient

async def main():
    client = AiAssistClient(api_key="aai_your_key")
    
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
    
    await client.close()

# Or use as context manager
async with AiAssistClient(api_key="aai_your_key") as client:
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
```

### Sync Client

```python
from aiassist import SyncAiAssistClient

client = SyncAiAssistClient(api_key="aai_your_key")

response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming

```python
async for chunk in await client.chat.completions.create(
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True
):
    delta = chunk.choices[0].get("delta", {})
    if delta.get("content"):
        print(delta["content"], end="", flush=True)
```

### Workspaces (Managed Conversations)

```python
# Create a workspace for a user
workspace = await client.workspaces.create(
    client_id="user_123",
    initial_message="I need help with my project"
)

# Send messages
message = await client.workspaces.send_message(
    workspace_id=workspace.id,
    content="Can you help me with Python?"
)

# Get conversation history
messages = await client.workspaces.get_messages(workspace.id)
```

### White-Label Integration

Proxy AiAssist through your own API:

```python
from fastapi import FastAPI, Request
from aiassist import AiAssistClient

app = FastAPI()
client = AiAssistClient(api_key="aai_your_api_key")

@app.post("/v1/chat")
async def chat(request: Request):
    data = await request.json()
    
    response = await client.chat.completions.create(
        messages=data["messages"],
        temperature=data.get("temperature", 0.7)
    )
    
    return {
        "response": response.choices[0].message.content,
        "model": response.model
    }
```

## Configuration

```python
client = AiAssistClient(
    api_key="aai_your_key",
    base_url="https://api.aiassist.net",  # Default
    timeout=30.0,                          # Request timeout
    max_retries=3                          # Retry on failures
)
```

## Error Handling

```python
from aiassist import (
    AiAssistClient,
    AuthenticationError,
    RateLimitError,
    APIError
)

try:
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}]
    )
except AuthenticationError:
    print("Invalid API key")
except RateLimitError:
    print("Too many requests - slow down")
except APIError as e:
    print(f"API error: {e}")
```

## Available Models Examples (actual models based on your provider)

- `llama-3.3-70b-versatile` - Best quality
- `llama-3.1-8b-instant` - Fast responses  
- `mixtral-8x7b-32768` - Long context

## Links

- [Documentation](https://aiassist.net/developer-doc)
- [API Reference](https://aiassist.net/dos)
- [GitHub](https://github.com/aiassistsecure/python-client)

## License

MIT License - Interchained LLC
