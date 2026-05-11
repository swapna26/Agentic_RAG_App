"""
Chat Router - OpenWebUI Compatible Endpoints

This module provides chat endpoints that are compatible with OpenWebUI,
enabling seamless integration with the frontend interface. It handles
chat completions using the CrewAI-powered RAG service.

Key Features:
- OpenWebUI compatibility
- CrewAI agent integration
- Conversation memory management
- Streaming and non-streaming responses
- Phoenix observability integration

Version: 1.0.0
"""

import json
import time
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import structlog

# Import Phoenix for tracing
try:
    from opentelemetry import trace
    TRACING_AVAILABLE = True
except ImportError:
    TRACING_AVAILABLE = False

logger = structlog.get_logger()

# Create router for chat endpoints
router = APIRouter(tags=["chat"])

# Pydantic models for OpenWebUI compatibility
class ChatMessage(BaseModel):
    """Individual chat message in a conversation."""
    role: str = Field(..., description="Role of the message sender (user/assistant/system)")
    content: str = Field(..., description="Content of the message")
    name: Optional[str] = Field(None, description="Name of the sender")

class ChatCompletionRequest(BaseModel):
    """Request model for chat completions compatible with OpenWebUI."""
    model: str = Field(..., description="Model to use for completion")
    messages: List[ChatMessage] = Field(..., description="List of conversation messages")
    stream: bool = Field(False, description="Whether to stream the response")
    temperature: Optional[float] = Field(0.1, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(4000, description="Maximum tokens in response")
    top_p: Optional[float] = Field(1.0, description="Top-p sampling parameter")
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")

class ChatCompletionResponse(BaseModel):
    """Response model for chat completions compatible with OpenWebUI."""
    id: str = Field(..., description="Unique identifier for the completion")
    object: str = Field("chat.completion", description="Object type")
    created: int = Field(..., description="Unix timestamp of creation")
    model: str = Field(..., description="Model used for completion")
    choices: List[Dict[str, Any]] = Field(..., description="List of completion choices")
    usage: Dict[str, int] = Field(..., description="Token usage information")

class ConversationHistory(BaseModel):
    conversation_id: str
    messages: List[Dict[str, Any]]

class ConversationResponse(BaseModel):
    success: bool
    conversation_id: str
    message: Optional[str] = None

def get_rag_service(request: Request):
    """Helper function to get RAG service from request."""
    rag_service = getattr(request.app.state, 'rag_service', None)
    if not rag_service:
        try:
            import main
            rag_service = main.rag_service
        except (ImportError, AttributeError):
            pass
    
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    return rag_service

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, http_request: Request):
    """
    OpenWebUI compatible chat completions endpoint with CrewAI integration.

    This is the primary endpoint that OpenWebUI uses for chat functionality.
    It processes user queries using the CrewAI multi-agent system and returns
    intelligent responses with source citations.

    Features:
    - CrewAI multi-agent processing
    - Conversation memory management
    - Source document retrieval
    - Streaming and non-streaming responses
    - Phoenix observability integration

    Args:
        request: Chat completion request with messages and parameters
        http_request: FastAPI request object for service access

    Returns:
        ChatCompletionResponse: Structured response compatible with OpenWebUI
    """
    # Create Phoenix trace for chat completion if tracing is available
    tracer = None
    span = None
    if TRACING_AVAILABLE:
        try:
            tracer = trace.get_tracer(__name__)
            span = tracer.start_span("chat_completions")
        except Exception:
            pass

    try:
        rag_service = get_rag_service(http_request)

        # Extract messages and create/maintain conversation ID
        user_messages = [msg for msg in request.messages if msg.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        latest_message = user_messages[-1].content

        # Generate conversation ID based on request context or create new one
        # In real applications, this would come from the client
        conversation_id = request.model_extra.get('conversation_id') if hasattr(request, 'model_extra') and request.model_extra else str(uuid.uuid4())[:16]

        # Convert OpenWebUI messages to conversation history for memory
        conversation_history = []
        for msg in request.messages[:-1]:  # Exclude the latest message
            conversation_history.append({
                "role": msg.role,
                "content": msg.content
            })

        # Add trace attributes if span is available
        if span:
            span.set_attribute("chat.model", request.model)
            span.set_attribute("chat.message", latest_message[:200])  # Limit message length
            span.set_attribute("chat.stream", request.stream)
            span.set_attribute("chat.conversation_id", conversation_id)
            span.set_attribute("chat.message_count", len(request.messages))

        logger.info("Processing chat completion",
                   model=request.model,
                   message_preview=latest_message[:100] + "..." if len(latest_message) > 100 else latest_message,
                   stream=request.stream,
                   conversation_id=conversation_id,
                   message_count=len(request.messages))
        
        if request.stream:
            response = StreamingResponse(
                stream_chat_response(rag_service, latest_message, conversation_history, request, conversation_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        else:
            response = await non_stream_chat_response(rag_service, latest_message, conversation_history, request, conversation_id)

        # Finalize trace
        if span:
            span.set_attribute("chat.success", True)
            span.set_attribute("chat.response_type", "streaming" if request.stream else "non_streaming")
            span.end()

        return response

    except Exception as e:
        # Record error in trace
        if span:
            span.set_attribute("chat.success", False)
            span.set_attribute("chat.error", str(e))
            span.end()

        logger.error("Chat completion failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")

async def non_stream_chat_response(rag_service, message: str, conversation_history: List[Dict], request: ChatCompletionRequest, conversation_id: str) -> ChatCompletionResponse:
    """Generate non-streaming chat response with RAG and optional CrewAI agents."""
    try:
        # Use the chat engine with conversation memory
        actual_message = message.replace("[AGENTIC_RAG]", "").strip().replace("[SIMPLE_RAG]", "").strip().replace("[CREW_AI]", "").strip()

        # Pass conversation history to the RAG service for context
        result = await rag_service.chat(actual_message, conversation_history, conversation_id)
        
        # Format response with sources as context
        response_content = result["response"]
        
        # Add source information if available
        sources = result.get("sources", [])
        if sources:
            response_content += "\n\n Sources"
            for i, source in enumerate(sources[:3], 1):  # Limit to top 3 sources
                metadata = source.get("metadata", {})

                # Handle case where metadata might be a string (from database JSON)
                if isinstance(metadata, str):
                    try:
                        import json
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                elif metadata is None:
                    metadata = {}

                doc_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("source_document") or f"Document {i}"
                score = source.get("score", 0.0)
                response_content += f"\n{i}. {doc_name} (relevance: {score:.2f})"
        
        # Add processing info
        processing_mode = result.get("metadata", {}).get("processing_mode", "unknown")
        agents_used = result.get("metadata", {}).get("agents_used", [])
        
        if agents_used:
            response_content += f"\n\nProcessed using: {', '.join(agents_used)}"
        
        # Calculate token usage (rough estimate)
        prompt_tokens = len(message.split())
        completion_tokens = len(response_content.split())
        total_tokens = prompt_tokens + completion_tokens
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
        # Log conversation history to Phoenix if available
        try:
            if hasattr(rag_service, 'phoenix_service') and rag_service.phoenix_service:
                # Extract conversation history
                conversation_history = []
                for msg in request.messages:
                    conversation_history.append({
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": time.time()
                    })

                # Log the complete conversation context
                await rag_service.phoenix_service.log_prompt_execution(
                    "openwebui_conversation_context",
                    {
                        "conversation_id": conversation_id,
                        "message_count": len(request.messages),
                        "conversation_history": str(conversation_history)
                    },
                    response_content,
                    {
                        "processing_mode": processing_mode,
                        "sources_count": len(sources),
                        "model": request.model,
                        "temperature": request.temperature,
                        "max_tokens": request.max_tokens
                    }
                )
        except Exception as e:
            logger.warning("Failed to log conversation history to Phoenix", error=str(e))

        logger.info("Non-streaming response generated",
                   response_length=len(result["response"]),
                   source_count=len(sources),
                   processing_mode=processing_mode,
                   conversation_id=conversation_id)

        return response
        
    except Exception as e:
        logger.error("Non-streaming response failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Response generation failed: {str(e)}")

async def stream_chat_response(rag_service, message: str, conversation_history: List[Dict], request: ChatCompletionRequest, conversation_id: str):
    """Generate streaming chat response with RAG."""
    try:
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created = int(time.time())
        
        # Send initial chunk
        initial_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }
            ]
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"
        
        # Get the full response from RAG with conversation history
        result = await rag_service.chat(message, conversation_history, conversation_id)
        response_text = result["response"]
        
        # Add source information
        sources = result.get("sources", [])
        if sources:
            response_text += "\n\n# Sources"
            for i, source in enumerate(sources[:3], 1):
                metadata = source.get("metadata", {})
                doc_name = metadata.get("file_name") or metadata.get("filename") or metadata.get("source_document") or f"Document {i}"
                score = source.get("score", 0.0)
                response_text += f"\n{i}. {doc_name} (relevance: {score:.2f})"
        
        # Stream the response in chunks
        chunk_size = 15  # Words per chunk for smooth streaming
        words = response_text.split()
        
        for i in range(0, len(words), chunk_size):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Add space after chunk unless it's the last chunk
            if i + chunk_size < len(words):
                chunk_text += " "
            
            chunk_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
        
        # Send final chunk
        final_chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
        logger.info("Streaming response completed", 
                   response_length=len(result["response"]),
                   source_count=len(sources),
                   conversation_id=conversation_id)
        
    except Exception as e:
        logger.error("Streaming response failed", error=str(e), exc_info=True)
        
        # Send error chunk
        error_chunk = {
            "id": f"chatcmpl-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": f"\n\nError: {str(e)}"},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"
        yield "data: [DONE]\n\n"

@router.get("/chat/models")
async def get_chat_models():
    """Get available chat models for OpenWebUI."""
    return {
        "object": "list",
        "data": [
            {
                "id": "agentic-rag-ollama",
                "object": "model",
                "created": 1677610602,
                "owned_by": "agentic-rag",
                "permission": [
                    {
                        "id": "modelperm-agentic-rag",
                        "object": "model_permission",
                        "created": 1677610602,
                        "allow_create_engine": False,
                        "allow_sampling": True,
                        "allow_logprobs": True,
                        "allow_search_indices": False,
                        "allow_view": True,
                        "allow_fine_tuning": False,
                        "organization": "*",
                        "group": None,
                        "is_blocking": False
                    }
                ],
                "root": "agentic-rag-ollama",
                "parent": None
            }
        ]
    }

@router.get("/chat/conversations/{conversation_id}", response_model=ConversationHistory)
async def get_conversation(conversation_id: str, request: Request):
    """Get conversation history by ID."""
    try:
        rag_service = get_rag_service(request)
        
        history = await rag_service.get_conversation_history(conversation_id)
        
        return ConversationHistory(
            conversation_id=conversation_id,
            messages=history
        )
        
    except Exception as e:
        logger.error("Failed to get conversation", 
                    conversation_id=conversation_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation: {str(e)}")

@router.delete("/chat/conversations/{conversation_id}", response_model=ConversationResponse)
async def clear_conversation(conversation_id: str, request: Request):
    """Clear conversation history by ID."""
    try:
        rag_service = get_rag_service(request)
        
        success = await rag_service.clear_conversation(conversation_id)
        
        return ConversationResponse(
            success=success,
            conversation_id=conversation_id,
            message="Conversation cleared successfully" if success else "Failed to clear conversation"
        )
        
    except Exception as e:
        logger.error("Failed to clear conversation", 
                    conversation_id=conversation_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to clear conversation: {str(e)}")

@router.post("/completions")
async def completions(request: ChatCompletionRequest, http_request: Request):
    """
    Legacy completions endpoint for backward compatibility.
    
    Some clients may use this endpoint instead of chat/completions.
    """
    return await chat_completions(request, http_request)

@router.get("/chat/health")
async def chat_health(request: Request):
    """Health check endpoint for chat service."""
    try:
        rag_service = get_rag_service(request)
        
        is_healthy = await rag_service.is_healthy()
        stats = await rag_service.get_index_stats()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "chat",
            "rag_available": bool(rag_service),
            "rag_healthy": is_healthy,
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0),
            "embedding_model": stats.get("embedding_model", "unknown"),
            "llm_model": stats.get("llm_model", "unknown"),
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error("Chat health check failed", error=str(e))
        return {
            "status": "error", 
            "service": "chat",
            "error": str(e),
            "timestamp": int(time.time())
        }

@router.get("/chat/stats")
async def chat_stats(request: Request):
    """Get detailed chat service statistics."""
    try:
        rag_service = get_rag_service(request)
        
        stats = await rag_service.get_index_stats()
        
        return {
            "service": "chat",
            "rag_stats": stats,
            "endpoints": {
                "chat_completions": "/api/chat/completions",
                "models": "/api/chat/models", 
                "health": "/api/chat/health",
                "conversations": "/api/chat/conversations/{id}"
            },
            "features": {
                "streaming": True,
                "conversation_memory": True,
                "source_citations": True,
                "rag_enabled": True,
                "openwebui_compatible": True
            },
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error("Failed to get chat stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

# Additional utility endpoints
@router.post("/chat/test")
async def test_chat(message: str, request: Request):
    """Simple test endpoint for chat functionality."""
    try:
        rag_service = get_rag_service(request)
        
        result = await rag_service.query(message, "test-conversation")
        
        return {
            "query": message,
            "response": result["response"],
            "sources_count": len(result.get("sources", [])),
            "metadata": result.get("metadata", {}),
            "timestamp": int(time.time())
        }
        
    except Exception as e:
        logger.error("Chat test failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")