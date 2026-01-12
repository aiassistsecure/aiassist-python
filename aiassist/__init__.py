"""
AiAssist Python SDK v1 - Lightweight API Client

Simple, lightweight Python client for the AiAssist AI API.
Only dependency: httpx

Usage:
    from aiassist import AiAssistClient

    client = AiAssistClient(api_key="aai_your_key")
    
    response = await client.chat.completions.create(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)
"""

from .client import (
    AiAssistClient,
    SyncAiAssistClient,
    AiAssistError,
    AuthenticationError,
    RateLimitError,
    APIError,
    ChatCompletion,
    ChatCompletionChunk,
    Message,
    Choice,
    Usage,
    Model,
    Workspace,
    WorkspaceMessage,
)

__version__ = "1.0.0"
__all__ = [
    "AiAssistClient",
    "SyncAiAssistClient",
    "AiAssistError",
    "AuthenticationError",
    "RateLimitError",
    "APIError",
    "ChatCompletion",
    "ChatCompletionChunk",
    "Message",
    "Choice",
    "Usage",
    "Model",
    "Workspace",
    "WorkspaceMessage",
]
