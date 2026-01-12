"""
AiAssist API Client - Lightweight SDK for API integration.

Full feature parity with React and Vanilla SDKs including:
- Shadow mode support (pending_approval)
- Human-in-the-loop mode detection
- Typing preview
- Conversation management

Example:
    from aiassist import AiAssistClient
    
    client = AiAssistClient(api_key="aai_your_key")
    
    # Create workspace with system prompt
    result = await client.workspaces.create(
        initial_message="Hello!",
        system_prompt="You are helpful.",
        context={"user_tier": "premium"}
    )
    
    # Send message and check mode
    response = await client.workspaces.send_message(
        result.workspace.id,
        "Help me"
    )
    print(f"Mode: {response.mode}")  # ai, shadow, takeover
    print(f"Pending approval: {response.pending_approval}")
"""

import json
import httpx
from typing import Optional, List, Dict, Any, AsyncIterator, Union, Literal
from dataclasses import dataclass, field


WorkspaceMode = Literal["ai", "shadow", "takeover"]
MessageRole = Literal["user", "ai", "human", "system"]


def normalize_role(role: str) -> MessageRole:
    """Normalize role strings to standard MessageRole."""
    if role in ("assistant", "ai"):
        return "ai"
    if role == "human":
        return "human"
    if role == "system":
        return "system"
    return "user"


@dataclass
class Message:
    role: str
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class Choice:
    index: int
    message: Message
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletion":
        choices = [
            Choice(
                index=c["index"],
                message=Message(
                    role=c["message"]["role"],
                    content=c["message"]["content"]
                ),
                finish_reason=c.get("finish_reason")
            )
            for c in data.get("choices", [])
        ]
        usage = None
        if data.get("usage"):
            usage = Usage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"]
            )
        return cls(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=choices,
            usage=usage
        )


@dataclass
class ChatCompletionChunk:
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatCompletionChunk":
        return cls(
            id=data["id"],
            object=data["object"],
            created=data["created"],
            model=data["model"],
            choices=data.get("choices", [])
        )


@dataclass
class Workspace:
    id: str
    mode: WorkspaceMode
    status: str = "active"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class WorkspaceMessage:
    id: str
    role: MessageRole
    content: str
    created_at: Optional[str] = None
    workspace_id: Optional[str] = None


@dataclass
class WorkspaceCreateResponse:
    workspace: Workspace
    messages: List[WorkspaceMessage]


@dataclass
class SendMessageResponse:
    user_message: WorkspaceMessage
    responses: List[WorkspaceMessage]
    mode: WorkspaceMode
    pending_approval: bool = False


@dataclass
class WorkspaceByClientResponse:
    workspace: Optional[Workspace]
    messages: List[WorkspaceMessage]
    exists: bool


class AiAssistError(Exception):
    """Base exception for AiAssist API errors."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        error_code: Optional[str] = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code


class AuthenticationError(AiAssistError):
    """Raised when API key is invalid or missing."""
    pass


class RateLimitError(AiAssistError):
    """Raised when rate limit is exceeded."""
    pass


class APIError(AiAssistError):
    """Raised for general API errors."""
    pass


class ChatCompletions:
    """Chat completions API - OpenAI compatible."""
    
    def __init__(self, client: "AiAssistClient"):
        self._client = client
    
    async def create(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs
    ) -> Union[ChatCompletion, AsyncIterator[ChatCompletionChunk]]:
        """Create a chat completion."""
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        if model:
            payload["model"] = model
        
        if stream:
            return self._stream_response(payload)
        
        response = await self._client._request("POST", "/v1/chat/completions", json=payload)
        return ChatCompletion.from_dict(response)
    
    async def _stream_response(
        self,
        payload: Dict[str, Any]
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completion chunks."""
        async with self._client._stream("POST", "/v1/chat/completions", json=payload) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield ChatCompletionChunk.from_dict(chunk)
                    except json.JSONDecodeError:
                        continue


class Chat:
    """Chat API namespace."""
    
    def __init__(self, client: "AiAssistClient"):
        self.completions = ChatCompletions(client)


class Workspaces:
    """Workspaces API for managed conversations with full feature support."""
    
    def __init__(self, client: "AiAssistClient"):
        self._client = client
    
    async def create(
        self,
        initial_message: Optional[str] = None,
        client_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WorkspaceCreateResponse:
        """
        Create a new workspace with optional system prompt and context.
        
        Args:
            initial_message: Optional first message from user
            client_id: Unique identifier for the end user
            system_prompt: System instructions for the AI
            context: Custom context data for AI decisions
            metadata: Custom metadata to attach
            
        Returns:
            WorkspaceCreateResponse with workspace and initial messages
        """
        payload: Dict[str, Any] = {}
        if initial_message:
            payload["initial_message"] = initial_message
        if client_id:
            payload["client_id"] = client_id
        if system_prompt:
            payload["system_prompt"] = system_prompt
        if context:
            payload["context"] = context
        if metadata:
            payload["metadata"] = metadata
        
        response = await self._client._request("POST", "/api/workspaces", json=payload)
        
        ws_data = response.get("workspace", response)
        messages_data = response.get("messages", [])
        
        return WorkspaceCreateResponse(
            workspace=Workspace(
                id=ws_data["id"],
                mode=ws_data.get("mode", "ai"),
                status=ws_data.get("status", "active"),
                metadata=ws_data.get("metadata")
            ),
            messages=[
                WorkspaceMessage(
                    id=m["id"],
                    role=normalize_role(m["role"]),
                    content=m["content"],
                    created_at=m.get("created_at")
                )
                for m in messages_data
            ]
        )
    
    async def get(self, workspace_id: str) -> Workspace:
        """Get workspace by ID."""
        response = await self._client._request("GET", f"/api/workspaces/{workspace_id}")
        ws_data = response.get("workspace", response)
        return Workspace(
            id=ws_data["id"],
            mode=ws_data.get("mode", "ai"),
            status=ws_data.get("status", "active"),
            metadata=ws_data.get("metadata")
        )
    
    async def get_by_client_id(self, client_id: str) -> WorkspaceByClientResponse:
        """Get workspace by client ID."""
        try:
            response = await self._client._request("GET", f"/api/workspaces/by-client/{client_id}")
            ws_data = response.get("workspace")
            messages_data = response.get("messages", [])
            
            return WorkspaceByClientResponse(
                workspace=Workspace(
                    id=ws_data["id"],
                    mode=ws_data.get("mode", "ai"),
                    status=ws_data.get("status", "active")
                ) if ws_data else None,
                messages=[
                    WorkspaceMessage(
                        id=m["id"],
                        role=normalize_role(m["role"]),
                        content=m["content"],
                        created_at=m.get("created_at")
                    )
                    for m in messages_data
                ],
                exists=response.get("exists", ws_data is not None)
            )
        except APIError as e:
            if e.status_code == 404:
                return WorkspaceByClientResponse(workspace=None, messages=[], exists=False)
            raise
    
    async def send_message(
        self,
        workspace_id: str,
        content: str
    ) -> SendMessageResponse:
        """
        Send a message to a workspace.
        
        Returns user message, AI/human responses, mode, and shadow mode approval status.
        """
        response = await self._client._request(
            "POST",
            f"/api/workspaces/{workspace_id}/messages",
            json={"content": content}
        )
        
        user_msg = response.get("user_message", {})
        responses_data = response.get("responses", [])
        
        return SendMessageResponse(
            user_message=WorkspaceMessage(
                id=user_msg.get("id", ""),
                role="user",
                content=user_msg.get("content", content),
                created_at=user_msg.get("created_at"),
                workspace_id=workspace_id
            ),
            responses=[
                WorkspaceMessage(
                    id=r["id"],
                    role=normalize_role(r["role"]),
                    content=r["content"],
                    created_at=r.get("created_at"),
                    workspace_id=workspace_id
                )
                for r in responses_data
            ],
            mode=response.get("mode", "ai"),
            pending_approval=response.get("pending_approval", False)
        )
    
    async def get_messages(self, workspace_id: str) -> List[WorkspaceMessage]:
        """Get all messages in a workspace."""
        response = await self._client._request("GET", f"/api/workspaces/{workspace_id}/messages")
        messages_data = response.get("messages", response) if isinstance(response, dict) else response
        return [
            WorkspaceMessage(
                id=msg["id"],
                role=normalize_role(msg["role"]),
                content=msg["content"],
                created_at=msg.get("created_at"),
                workspace_id=workspace_id
            )
            for msg in messages_data
        ]
    
    async def send_typing_preview(self, workspace_id: str, text: str) -> None:
        """Send typing preview (fails silently)."""
        try:
            await self._client._request(
                "POST",
                f"/api/workspaces/{workspace_id}/typing",
                json={"text": text}
            )
        except Exception:
            pass
    
    async def end_conversation(self, workspace_id: str) -> None:
        """End a conversation (fails silently)."""
        try:
            await self._client._request(
                "POST",
                f"/api/workspaces/{workspace_id}/end",
                json={}
            )
        except Exception:
            pass


class Models:
    """Models API."""
    
    def __init__(self, client: "AiAssistClient"):
        self._client = client
    
    async def list(self) -> List[Dict[str, Any]]:
        """List available models for your plan."""
        response = await self._client._request("GET", "/v1/models")
        return response.get("data", [])


@dataclass
class AiAssistClient:
    """
    AiAssist API client with full feature parity.
    
    Supports:
    - Shadow mode (pending_approval indicator)
    - Human-in-the-loop mode detection
    - Typing preview
    - Conversation lifecycle management
    """
    
    api_key: str
    base_url: str = "https://api.aiassist.net"
    timeout: float = 30.0
    max_retries: int = 3
    
    chat: Chat = field(init=False)
    workspaces: Workspaces = field(init=False)
    models: Models = field(init=False)
    
    _http_client: Optional[httpx.AsyncClient] = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        self.chat = Chat(self)
        self.workspaces = Workspaces(self)
        self.models = Models(self)
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "aiassist-python/1.0.0"
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=self.timeout
            )
        return self._http_client
    
    async def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an HTTP request with error handling and retries."""
        client = await self._get_client()
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await client.request(method, path, **kwargs)
                
                if response.status_code == 401:
                    raise AuthenticationError(
                        "Invalid API key",
                        status_code=401,
                        error_code="invalid_api_key"
                    )
                
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    if attempt < self.max_retries - 1:
                        import asyncio
                        await asyncio.sleep(retry_after)
                        continue
                    raise RateLimitError(
                        "Rate limit exceeded",
                        status_code=429,
                        error_code="rate_limit_exceeded"
                    )
                
                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    raise APIError(
                        error_data.get("detail", f"HTTP {response.status_code}"),
                        status_code=response.status_code,
                        error_code=error_data.get("code")
                    )
                
                return response.json()
                
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise APIError(f"Connection error: {e}")
        
        raise APIError(f"Max retries exceeded: {last_error}")
    
    def _stream(
        self,
        method: str,
        path: str,
        **kwargs
    ):
        """Create a streaming request context."""
        return _StreamContext(self, method, path, **kwargs)
    
    async def close(self):
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()


class _StreamContext:
    """Context manager for streaming responses."""
    
    def __init__(self, client: AiAssistClient, method: str, path: str, **kwargs):
        self._client = client
        self._method = method
        self._path = path
        self._kwargs = kwargs
        self._stream_ctx: Any = None
        self._response: Any = None
    
    async def __aenter__(self):
        http_client = await self._client._get_client()
        self._stream_ctx = http_client.stream(
            self._method,
            self._path,
            **self._kwargs
        )
        self._response = await self._stream_ctx.__aenter__()
        return self._response
    
    async def __aexit__(self, *args):
        if self._stream_ctx:
            await self._stream_ctx.__aexit__(*args)


class SyncAiAssistClient:
    """Synchronous wrapper for AiAssistClient."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.aiassist.net",
        timeout: float = 30.0
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=base_url,
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "aiassist-python/1.0.0"
            },
            timeout=timeout
        )
        self.chat = _SyncChat(self)
    
    def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        response = self._client.request(method, path, **kwargs)
        
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key", status_code=401)
        if response.status_code == 429:
            raise RateLimitError("Rate limit exceeded", status_code=429)
        if response.status_code >= 400:
            error_data = response.json() if response.content else {}
            raise APIError(
                error_data.get("detail", f"HTTP {response.status_code}"),
                status_code=response.status_code
            )
        
        return response.json()
    
    def close(self):
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class _SyncChat:
    def __init__(self, client: SyncAiAssistClient):
        self.completions = _SyncChatCompletions(client)


class _SyncChatCompletions:
    def __init__(self, client: SyncAiAssistClient):
        self._client = client
    
    def create(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> ChatCompletion:
        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        if model:
            payload["model"] = model
        
        response = self._client._request("POST", "/v1/chat/completions", json=payload)
        return ChatCompletion.from_dict(response)
