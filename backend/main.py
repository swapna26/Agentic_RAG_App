"""
Agentic RAG Backend - FastAPI Application

This is the main FastAPI application that provides the backend API for the
Agentic RAG system. It integrates CrewAI agents, PostgreSQL vector storage,
and Phoenix observability to deliver intelligent document retrieval and
response generation capabilities.

Key Features:
- CrewAI multi-agent processing (Primary)
- PostgreSQL vector store integration
- OpenWebUI compatibility
- Phoenix observability and tracing
- Conversation memory management
- Rate limiting and error handling

API Endpoints:
- /api/models - Model information for OpenWebUI
- /api/chat/completions - Chat completions endpoint
- /health - Health check endpoint
- /info - System information

Version: 1.0.0
"""

import ssl
import urllib3
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

# Disable SSL verification globally for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context

from config import config
from services.rag_service import RAGService
from services.phoenix_service import PhoenixService
from services.prompt_store import PromptStore

# Import Phoenix for auto-instrumentation
try:
    import phoenix as px
    from phoenix.trace import using_project
    from phoenix.trace.llama_index import LlamaIndexInstrumentor
    from phoenix.trace.openai import OpenAIInstrumentor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    px = None
    using_project = None
    LlamaIndexInstrumentor = None
    OpenAIInstrumentor = None

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global service instances
rag_service = None
prompt_store = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown.
    
    This function handles:
    1. Phoenix service initialization and instrumentation
    2. RAG service initialization with CrewAI agents
    3. Database connection setup
    4. Graceful shutdown and cleanup
    
    Args:
        app: FastAPI application instance
    """
    global rag_service, phoenix_service, prompt_store

    # Startup sequence
    logger.info("Starting Agentic RAG Backend with CrewAI integration")
    try:
        # Initialize Prompt Store first (local JSON storage with versioning)
        prompt_store = PromptStore()
        await prompt_store.initialize()
        app.state.prompt_store = prompt_store
        logger.info("Prompt store initialized")

        # Initialize Phoenix service with prompt store integration
        phoenix_service = PhoenixService(config, prompt_store=prompt_store)
        await phoenix_service.initialize()

        # Set up Phoenix auto-instrumentation
        if PHOENIX_AVAILABLE and phoenix_service.tracing_enabled:
            try:
                # Set Phoenix endpoint for auto-instrumentation
                import os
                from urllib.parse import urlparse
                parsed_url = urlparse(config.phoenix_base_url)
                phoenix_host = parsed_url.hostname or "localhost"
                phoenix_port = parsed_url.port or 6006

                # Configure Phoenix session
                px.connect(endpoint=f"http://{phoenix_host}:{phoenix_port}")

                # Instrument LlamaIndex
                if LlamaIndexInstrumentor:
                    LlamaIndexInstrumentor().instrument()
                    logger.info("LlamaIndex instrumentation enabled")

                # Enable project context
                if using_project:
                    using_project(config.phoenix_project_name).__enter__()
                    logger.info("Phoenix project context enabled", project=config.phoenix_project_name)

                logger.info("Phoenix auto-instrumentation enabled")
            except Exception as e:
                logger.warning("Failed to setup Phoenix auto-instrumentation", error=str(e))

        # Initialize RAG service with Phoenix and PromptStore integration
        rag_service = RAGService(config)
        rag_service.phoenix_service = phoenix_service  # Inject Phoenix service
        rag_service.prompt_store = prompt_store  # Inject PromptStore for CrewAI agents
        await rag_service.initialize()
        
        logger.info("All services initialized successfully")
        yield
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    finally:
        # Shutdown
        if rag_service:
            await rag_service.cleanup()
        if phoenix_service:
            await phoenix_service.cleanup()
        if prompt_store:
            await prompt_store.cleanup()
        logger.info("All services cleanup completed")

# Create FastAPI app
app = FastAPI(
    title="Agentic RAG API",
    description="Intelligent RAG system with CrewAI multi-agent processing and OpenWebUI compatibility",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for OpenWebUI compatibility
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Options handler for CORS preflight
@app.options("/{path:path}")
async def options_handler():
    """Handle CORS preflight requests."""
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        if rag_service and await rag_service.is_healthy():
            return {"status": "healthy", "service": "agentic-rag-backend"}
        else:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "service": "agentic-rag-backend"}
            )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={"status": "error", "error": str(e)}
        )

# OpenWebUI compatible endpoints
@app.get("/api/models")
async def get_models():
    """Get available models (OpenWebUI compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": config.model_name,
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
                "root": config.model_name,
                "parent": None
            }
        ]
    }

@app.get("/api/models/{model_id}")
async def get_model(model_id: str):
    """Get specific model details (OpenWebUI compatible)."""
    if model_id != config.model_name:
        return JSONResponse(
            status_code=404,
            content={"error": {"message": f"Model {model_id} not found", "type": "invalid_request_error"}}
        )
    
    return {
        "id": config.model_name,
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
        "root": config.model_name,
        "parent": None
    }

# Include chat endpoints
from routers.chat import router as chat_router
app.include_router(chat_router, prefix="/api")

# Include document management endpoints
from routers.documents import router as docs_router
app.include_router(docs_router, prefix="/api")

# Include prompt management endpoints
from routers.prompts import router as prompts_router
app.include_router(prompts_router, prefix="/api")

# Include cache management endpoints
from routers.cache import router as cache_router
app.include_router(cache_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config.api_host,
        port=config.api_port,
        reload=True,
        log_level=config.log_level.lower()
    )