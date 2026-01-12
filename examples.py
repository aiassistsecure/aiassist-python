"""
AiAssist Python Client - Full Example Usage

Demonstrates all features of the SDK:
- Chat completions (OpenAI-compatible)
- Streaming responses
- Workspace management
- Shadow mode detection
- Human-in-the-loop handling
- Typing previews
- Conversation lifecycle

Works with all 12 providers:
  OpenAI, Anthropic, Groq, Gemini, Mistral, xAI Grok,
  Together AI, OpenRouter, DeepSeek, Fireworks, Perplexity, PIN
"""

import asyncio
from aiassist import (
    AiAssistClient,
    SyncAiAssistClient,
    AiAssistError,
    AuthenticationError,
    RateLimitError,
)


# =============================================================================
# BASIC CHAT COMPLETION
# =============================================================================

async def basic_chat():
    """Simple chat completion - works like OpenAI SDK."""
    async with AiAssistClient(
        api_key="aai_your_api_key",
        base_url="https://api.aiassist.net"
    ) as client:
        
        response = await client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is Python?"}
            ],
            model="gpt-4o",  # Or any model from your configured providers
            temperature=0.7,
            max_tokens=500
        )
        
        print(response.choices[0].message.content)
        print(f"Tokens used: {response.usage.total_tokens}")


# =============================================================================
# MULTI-PROVIDER EXAMPLES
# =============================================================================

async def multi_provider_chat():
    """Use different providers by specifying different models."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        # OpenAI
        openai_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from OpenAI"}],
            model="gpt-4o"
        )
        
        # Anthropic
        anthropic_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Anthropic"}],
            model="claude-3-5-sonnet-20241022"
        )
        
        # Groq (ultra-fast)
        groq_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Groq"}],
            model="llama-3.3-70b-versatile"
        )
        
        # Google Gemini
        gemini_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Gemini"}],
            model="gemini-1.5-pro"
        )
        
        # Mistral
        mistral_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Mistral"}],
            model="mistral-large-latest"
        )
        
        # xAI Grok
        grok_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Grok"}],
            model="grok-2"
        )
        
        # Together AI
        together_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Together"}],
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        
        # OpenRouter (any model)
        openrouter_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from OpenRouter"}],
            model="openrouter/auto"
        )
        
        # DeepSeek
        deepseek_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from DeepSeek"}],
            model="deepseek-chat"
        )
        
        # Fireworks
        fireworks_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Fireworks"}],
            model="accounts/fireworks/models/llama-v3p3-70b-instruct"
        )
        
        # Perplexity
        perplexity_response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello from Perplexity"}],
            model="llama-3.1-sonar-large-128k-online"
        )


# =============================================================================
# STREAMING RESPONSES
# =============================================================================

async def streaming_chat():
    """Stream responses token by token."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        stream = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Write a haiku about coding."}
            ],
            model="gpt-4o",
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].get("delta", {}).get("content"):
                print(chunk.choices[0]["delta"]["content"], end="", flush=True)
        
        print()  # Newline at end


# =============================================================================
# WORKSPACE MANAGEMENT (Managed Conversations)
# =============================================================================

async def workspace_conversation():
    """Create and manage a workspace conversation."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        # Create workspace with system prompt and context
        result = await client.workspaces.create(
            initial_message="Hi, I need help with my order.",
            client_id="customer_12345",  # Your user ID
            system_prompt="You are a helpful customer support agent.",
            context={
                "customer_tier": "premium",
                "recent_orders": ["ORD-001", "ORD-002"],
                "account_status": "active"
            },
            metadata={
                "source": "mobile_app",
                "session_id": "sess_abc123"
            }
        )
        
        workspace = result.workspace
        print(f"Created workspace: {workspace.id}")
        print(f"Mode: {workspace.mode}")  # ai, shadow, or takeover
        
        # Print initial messages (including AI response if in AI mode)
        for msg in result.messages:
            print(f"[{msg.role}]: {msg.content}")


async def workspace_send_message():
    """Send messages and handle different modes."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        workspace_id = "ws_existing_id"
        
        # Send a message
        response = await client.workspaces.send_message(
            workspace_id,
            "What's the status of my recent order?"
        )
        
        # Check the mode
        print(f"Mode: {response.mode}")
        
        if response.mode == "ai":
            # Fully autonomous - AI handled it
            for msg in response.responses:
                print(f"AI: {msg.content}")
                
        elif response.mode == "shadow":
            # Shadow mode - AI drafted, awaiting manager approval
            if response.pending_approval:
                print("Response is pending manager approval")
                for msg in response.responses:
                    print(f"[DRAFT] AI: {msg.content}")
            else:
                # Already approved
                for msg in response.responses:
                    print(f"AI (approved): {msg.content}")
                    
        elif response.mode == "takeover":
            # Human takeover - waiting for human agent
            print("Conversation handed to human agent")


# =============================================================================
# SHADOW MODE HANDLING (Enterprise)
# =============================================================================

async def shadow_mode_workflow():
    """Complete shadow mode workflow with pending approval detection."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        # Get or create workspace
        result = await client.workspaces.get_by_client_id("customer_456")
        
        if not result.exists:
            create_result = await client.workspaces.create(
                client_id="customer_456",
                system_prompt="You are a sales agent. Be helpful but professional."
            )
            workspace_id = create_result.workspace.id
        else:
            workspace_id = result.workspace.id
        
        # Send message
        response = await client.workspaces.send_message(
            workspace_id,
            "I want to cancel my subscription"
        )
        
        # Handle based on mode and approval status
        if response.pending_approval:
            # Show draft to user with "pending" indicator
            print("Your message is being reviewed...")
            draft = response.responses[0].content if response.responses else ""
            # Optionally show draft preview to end user
            
        elif response.mode == "takeover":
            # Human is handling
            print("A team member will respond shortly...")
            
        else:
            # Normal AI response
            for msg in response.responses:
                print(f"Agent: {msg.content}")


# =============================================================================
# TYPING PREVIEWS
# =============================================================================

async def typing_preview():
    """Send typing previews for real-time feedback."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        workspace_id = "ws_xxx"
        
        # As user types, send previews (debounce in your UI)
        await client.workspaces.send_typing_preview(workspace_id, "I need help")
        await asyncio.sleep(0.5)
        await client.workspaces.send_typing_preview(workspace_id, "I need help with my")
        await asyncio.sleep(0.5)
        await client.workspaces.send_typing_preview(workspace_id, "I need help with my order")
        
        # Then send the actual message
        response = await client.workspaces.send_message(
            workspace_id,
            "I need help with my order"
        )


# =============================================================================
# CONVERSATION LIFECYCLE
# =============================================================================

async def conversation_lifecycle():
    """Full conversation lifecycle management."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        # Create
        result = await client.workspaces.create(
            client_id="user_789",
            initial_message="Hello!"
        )
        workspace_id = result.workspace.id
        
        # Converse
        await client.workspaces.send_message(workspace_id, "First question")
        await client.workspaces.send_message(workspace_id, "Follow up")
        
        # Get all messages
        messages = await client.workspaces.get_messages(workspace_id)
        for msg in messages:
            print(f"[{msg.created_at}] {msg.role}: {msg.content}")
        
        # End conversation
        await client.workspaces.end_conversation(workspace_id)


# =============================================================================
# LIST AVAILABLE MODELS
# =============================================================================

async def list_models():
    """List all models available for your plan."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        models = await client.models.list()
        
        for model in models:
            print(f"- {model['id']}: {model.get('owned_by', 'unknown')}")


# =============================================================================
# ERROR HANDLING
# =============================================================================

async def error_handling():
    """Proper error handling patterns."""
    async with AiAssistClient(api_key="aai_xxx") as client:
        
        try:
            response = await client.chat.completions.create(
                messages=[{"role": "user", "content": "Hello"}],
                model="gpt-4o"
            )
            print(response.choices[0].message.content)
            
        except AuthenticationError:
            print("Invalid API key - check your credentials")
            
        except RateLimitError as e:
            print(f"Rate limited - retry after backoff")
            # Implement exponential backoff
            
        except AiAssistError as e:
            print(f"API error: {e} (status: {e.status_code})")


# =============================================================================
# SYNCHRONOUS CLIENT (for non-async code)
# =============================================================================

def sync_example():
    """Use the synchronous client for simpler scripts."""
    with SyncAiAssistClient(
        api_key="aai_xxx",
        base_url="https://your-instance.com"
    ) as client:
        
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Hello!"}
            ],
            model="gpt-4o"
        )
        
        print(response.choices[0].message.content)


# =============================================================================
# REAL-WORLD: CUSTOMER SUPPORT BOT
# =============================================================================

async def customer_support_bot():
    """Production-ready customer support integration."""
    
    async with AiAssistClient(
        api_key="aai_xxx",
        base_url="https://api.yourdomain.com",
        timeout=60.0,
        max_retries=3
    ) as client:
        
        customer_id = "cust_12345"
        
        # Check for existing conversation
        existing = await client.workspaces.get_by_client_id(customer_id)
        
        if existing.exists:
            workspace_id = existing.workspace.id
            print(f"Resuming conversation {workspace_id}")
            # Show previous messages
            for msg in existing.messages[-5:]:  # Last 5
                print(f"{msg.role}: {msg.content}")
        else:
            # Create new workspace with rich context
            result = await client.workspaces.create(
                client_id=customer_id,
                system_prompt="""You are a helpful customer support agent for ACME Inc.
                
Guidelines:
- Be friendly and professional
- If you can't help, offer to escalate
- Never make promises about refunds without manager approval
- Always verify customer identity before sharing account details""",
                context={
                    "customer_name": "John Doe",
                    "plan": "enterprise",
                    "tenure_months": 24,
                    "open_tickets": 0,
                    "last_purchase": "2024-12-15"
                }
            )
            workspace_id = result.workspace.id
            print(f"New conversation started: {workspace_id}")
        
        # Conversation loop
        while True:
            user_input = input("\nYou: ").strip()
            if not user_input or user_input.lower() in ["quit", "exit"]:
                await client.workspaces.end_conversation(workspace_id)
                print("Conversation ended.")
                break
            
            response = await client.workspaces.send_message(workspace_id, user_input)
            
            if response.pending_approval:
                print("\n[Pending manager review...]")
            elif response.mode == "takeover":
                print("\n[Connecting you to a team member...]")
            else:
                for msg in response.responses:
                    print(f"\nAgent: {msg.content}")


# =============================================================================
# RUN EXAMPLES
# =============================================================================

if __name__ == "__main__":
    print("=== AiAssist Python Client Examples ===\n")
    
    # Uncomment to run:
    # asyncio.run(basic_chat())
    # asyncio.run(streaming_chat())
    # asyncio.run(workspace_conversation())
    # asyncio.run(shadow_mode_workflow())
    # asyncio.run(list_models())
    # asyncio.run(customer_support_bot())
    # sync_example()
    
    print("See examples.py for full usage patterns.")
