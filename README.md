# AiAssist Python SDK

**Enterprise-grade AI orchestration client for Python.**

Connect to 12 AI providers with a single, unified API. Supports BYOK (Bring Your Own Key), shadow mode, human-in-the-loop, streaming, and managed workspaces.

```python
pip install aiassist
```

## Features

- **12 Providers** - OpenAI, Anthropic, Groq, Gemini, Mistral, xAI Grok, Together AI, OpenRouter, DeepSeek, Fireworks, Perplexity, PIN
- **OpenAI-Compatible** - Drop-in replacement for existing OpenAI code
- **Shadow Mode** - AI drafts responses for human approval (Enterprise)
- **Human-in-the-Loop** - Seamless handoff between AI and humans
- **Managed Workspaces** - Stateful conversations with context injection
- **Streaming** - Real-time token-by-token responses
- **Lightweight** - Only dependency is `httpx`

---

## Quick Start

### Async Client (Recommended)

```python
import asyncio
from aiassist import AiAssistClient

async def main():
    async with AiAssistClient(
        api_key="aai_your_api_key",
        base_url="https://api.aiassist.net"
    ) as client:
        
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            model="gpt-4o"
        )
        
        print(response.choices[0].message.content)

asyncio.run(main())
```

### Sync Client

```python
from aiassist import SyncAiAssistClient

with SyncAiAssistClient(api_key="aai_xxx") as client:
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}],
        model="claude-3-5-sonnet-20241022"
    )
    print(response.choices[0].message.content)
```

---

## Multi-Provider Support

Use any model from any provider - just pass the model string:

```python
# OpenAI
await client.chat.completions.create(messages=msgs, model="gpt-4o")
await client.chat.completions.create(messages=msgs, model="gpt-4o-mini")

# Anthropic
await client.chat.completions.create(messages=msgs, model="claude-3-5-sonnet-20241022")
await client.chat.completions.create(messages=msgs, model="claude-3-haiku-20240307")

# Groq (ultra-fast)
await client.chat.completions.create(messages=msgs, model="llama-3.3-70b-versatile")
await client.chat.completions.create(messages=msgs, model="mixtral-8x7b-32768")

# Google Gemini
await client.chat.completions.create(messages=msgs, model="gemini-1.5-pro")
await client.chat.completions.create(messages=msgs, model="gemini-1.5-flash")

# Mistral
await client.chat.completions.create(messages=msgs, model="mistral-large-latest")

# xAI Grok
await client.chat.completions.create(messages=msgs, model="grok-2")

# Together AI
await client.chat.completions.create(messages=msgs, model="meta-llama/Llama-3.3-70B-Instruct-Turbo")

# OpenRouter
await client.chat.completions.create(messages=msgs, model="openrouter/auto")

# DeepSeek
await client.chat.completions.create(messages=msgs, model="deepseek-chat")

# Fireworks
await client.chat.completions.create(messages=msgs, model="accounts/fireworks/models/llama-v3p3-70b-instruct")

# Perplexity
await client.chat.completions.create(messages=msgs, model="llama-3.1-sonar-large-128k-online")

# PIN (P2P Network)
await client.chat.completions.create(messages=msgs, model="pin:operator-alias/llama3")
```

---

## Streaming

```python
stream = await client.chat.completions.create(
    messages=[{"role": "user", "content": "Write a poem about AI."}],
    model="gpt-4o",
    stream=True
)

async for chunk in stream:
    if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
        print(chunk.choices[0]["delta"]["content"], end="", flush=True)
```

---

## Workspaces (Managed Conversations)

Workspaces provide stateful, managed conversations with context injection:

```python
# Create workspace with system prompt and context
result = await client.workspaces.create(
    client_id="customer_12345",
    initial_message="I need help with my order",
    system_prompt="You are a customer support agent for ACME Inc.",
    context={
        "customer_name": "John Doe",
        "customer_tier": "premium",
        "recent_orders": ["ORD-001", "ORD-002"]
    }
)

workspace_id = result.workspace.id
print(f"Mode: {result.workspace.mode}")  # ai, shadow, or takeover

# Send messages
response = await client.workspaces.send_message(
    workspace_id,
    "What's the status of my last order?"
)

for msg in response.responses:
    print(f"Agent: {msg.content}")

# Get conversation history
messages = await client.workspaces.get_messages(workspace_id)

# Resume existing conversation
existing = await client.workspaces.get_by_client_id("customer_12345")
if existing.exists:
    workspace_id = existing.workspace.id

# End conversation
await client.workspaces.end_conversation(workspace_id)
```

---

## Shadow Mode (Enterprise)

Shadow mode requires manager approval before AI responses reach end users:

```python
response = await client.workspaces.send_message(workspace_id, "I want a refund")

if response.pending_approval:
    # Response is queued for manager review
    print("Your message is being reviewed...")
    draft = response.responses[0].content if response.responses else ""
    
elif response.mode == "takeover":
    # Human agent has taken over
    print("A team member will respond shortly...")
    
else:
    # Normal AI response (approved or AI mode)
    for msg in response.responses:
        print(f"Agent: {msg.content}")
```

---

## Typing Previews

Send real-time typing previews for manager dashboards:

```python
# As user types (debounce in your UI)
await client.workspaces.send_typing_preview(workspace_id, "I need help with")
await client.workspaces.send_typing_preview(workspace_id, "I need help with my order")

# Then send actual message
response = await client.workspaces.send_message(workspace_id, "I need help with my order")
```

---

## List Available Models

```python
models = await client.models.list()
for model in models:
    print(f"- {model['id']}")
```

---

## Configuration

```python
client = AiAssistClient(
    api_key="aai_your_key",                 # Required - your API key
    base_url="https://api.aiassist.net",    # Your AiAssist instance URL
    timeout=30.0,                           # Request timeout (seconds)
    max_retries=3                           # Retry count for failures
)
```

---

## Error Handling

```python
from aiassist import (
    AiAssistClient,
    AiAssistError,
    AuthenticationError,
    RateLimitError,
    APIError
)

try:
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-4o"
    )
except AuthenticationError:
    # Invalid or expired API key
    print("Check your API key")
    
except RateLimitError:
    # Too many requests - implement backoff
    print("Rate limited - retry with exponential backoff")
    
except APIError as e:
    # General API error
    print(f"Error {e.status_code}: {e}")
```

---

## White-Label Integration

Proxy AiAssist through your own API for white-label deployments:

```python
from fastapi import FastAPI, Request, HTTPException
from aiassist import AiAssistClient, AiAssistError

app = FastAPI()
client = AiAssistClient(api_key="aai_xxx", base_url="https://api.aiassist.net")

@app.post("/api/chat")
async def chat(request: Request):
    data = await request.json()
    
    try:
        response = await client.chat.completions.create(
            messages=data["messages"],
            model=data.get("model", "gpt-4o"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1024)
        )
        
        return {
            "message": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "tokens": response.usage.total_tokens
            } if response.usage else None
        }
        
    except AiAssistError as e:
        raise HTTPException(status_code=e.status_code or 500, detail=str(e))


@app.post("/api/workspaces")
async def create_workspace(request: Request):
    data = await request.json()
    
    result = await client.workspaces.create(
        client_id=data["user_id"],
        initial_message=data.get("message"),
        system_prompt=data.get("system_prompt"),
        context=data.get("context")
    )
    
    return {
        "workspace_id": result.workspace.id,
        "mode": result.workspace.mode,
        "messages": [
            {"role": m.role, "content": m.content}
            for m in result.messages
        ]
    }
```

---

## Response Objects

### ChatCompletion

```python
response.id              # "chatcmpl-xxx"
response.model           # "gpt-4o"
response.choices[0].message.role     # "assistant"
response.choices[0].message.content  # "Hello! How can I help?"
response.choices[0].finish_reason    # "stop"
response.usage.prompt_tokens         # 10
response.usage.completion_tokens     # 15
response.usage.total_tokens          # 25
```

### Workspace

```python
workspace.id        # "ws_xxx"
workspace.mode      # "ai" | "shadow" | "takeover"
workspace.status    # "active"
workspace.metadata  # {...}
```

### SendMessageResponse

```python
response.user_message    # WorkspaceMessage
response.responses       # List[WorkspaceMessage]
response.mode            # "ai" | "shadow" | "takeover"
response.pending_approval  # True if awaiting manager approval
```

---

## Full Examples

See [`examples.py`](./examples.py) for complete working examples including:

- Basic chat completions
- Multi-provider usage
- Streaming responses
- Workspace lifecycle
- Shadow mode handling
- Customer support bot

---

## Requirements

- Python 3.8+
- httpx

---

## Links

- [Documentation](https://aiassist.net/developer-docs)
- [API Reference](https://aiassist.net/docs)
- [Examples](./examples.py)

---

## License

MIT License - Interchained LLC

Copyright (c) 2024-2026 Interchained LLC. All rights reserved.
