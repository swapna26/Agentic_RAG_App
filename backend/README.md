# Agentic RAG Backend

Production-ready FastAPI backend for Retrieval-Augmented Generation with integrated conversation memory and CrewAI agents. Features OpenWebUI compatibility for seamless chat interface integration and intelligent context-aware responses.

## Processing Flow

1) CrewAI (primary): multi-agent retrieval + answer generation
2) Chat engine (fallback): LlamaIndex chat with memory
3) Query engine (final fallback): direct RAG query

Context handling:
- Recent turns are included only when the new question overlaps the prior topic (heuristic)
- New-topic questions are answered standalone (no contamination)

## Endpoints (OpenAI compatible)

- GET `/api/models`
- POST `/api/chat/completions`
- GET `/health`
- GET `/info`
- GET `/api/cache/stats` — cache hit/miss statistics
- GET `/api/cache/config` — current cache configuration
- POST `/api/cache/config` — update cache settings at runtime
- POST `/api/cache/clear` — clear all cached entries

## Configuration (env)

- `DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/agentic_rag`
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL=llama3.2:1b`
- `OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5`
- `SIMILARITY_TOP_K=5`
- `TEMPERATURE=0.1`
- `MODEL_NAME=agentic-rag-ollama`
- `CREW_VERBOSE=true`

## Run

### Local

```
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Docker Compose

```
docker compose --profile backend up -d
# Services: postgres, ollama, rag_backend, openwebui
```

## OpenWebUI


- Model listed as: `agentic-rag-ollama`
- Chat via `/api/chat/completions`

## CrewAI Behavior (output hygiene)

- Uses configured Ollama model: `ollama/<OLLAMA_MODEL>`
- Plain-text outputs: headings/bold/lists removed
- Inline “Sources” removed from answer (sources returned separately)
- No embedded follow-up questions or planner artifacts

## Evaluation (RAGas)

Run in 3 batches to avoid timeouts on Ollama:
1) `answer_similarity`, `context_recall`
2) `faithfulness`, `answer_relevancy`
3) `context_precision`, `answer_correctness`

Merge the three reports into one combined JSON.

## Troubleshooting (current)

- Some LLM-heavy metrics (relevancy, precision, correctness) may time out on Ollama
- Use batched evaluation and merge results
- Ensure models are pulled:
  - `ollama pull llama3.2:1b`
  - `ollama pull nomic-embed-text:v1.5`

## Architecture Overview

The backend implements an intelligent chatbot system with conversation memory and multi-agent RAG processing:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   OpenWebUI     │    │   FastAPI        │    │   PostgreSQL    │
│   Chat Client   │◄──►│   Backend        │◄──►│   + PGVector    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Conversation     │
                       │ Memory Engine    │
                       └──────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
            ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
            │  CrewAI      │ │ Chat Engine  │ │ Query Engine │
            │  Agents      │ │ (Fallback)   │ │ (Final)      │
            │ (Primary)    │ │              │ │              │
            └──────────────┘ └──────────────┘ └──────────────┘
                    │           │           │
                    └───────────┼───────────┘
                                ▼
                       ┌──────────────────┐
                       │   Ollama LLM     │
                       │   + Embeddings   │
                       └──────────────────┘
```

## Key Features

- **CrewAI Multi-Agent Processing**: Primary intelligent processing with specialized agents
- **Conversation Memory**: Automatic context management with LlamaIndex ChatMemoryBuffer
- **Intelligent Fallback**: Graceful degradation from CrewAI → Chat Engine → Query Engine
- **OpenWebUI Integration**: Seamless chat interface with OpenAI API compatibility
- **Context-Aware Responses**: Follow-up questions answered with full conversation context
- **Production Ready**: Comprehensive error handling, logging, and monitoring

## Core Components

### CrewAI Agents (`agents/crew_agents.py`)

**Primary Processing System** with specialized agent roles:

**Agent Team:**
- **Query Analyzer** - Analyzes conversation context and determines search strategy
- **Document Retrieval Specialist** - Context-aware document search using vector store
- **Information Extractor** - Synthesizes responses based on retrieved documents
- **Response Formatter** - Ensures proper formatting and completeness

**Processing Flow:**
1. **Query Analysis** - Understands conversation context and question type
2. **Document Retrieval** - Searches vector store for relevant documents
3. **Information Extraction** - Extracts specific information from documents
4. **Response Formatting** - Formats final response with proper structure

### RAG Service (`services/rag_service.py`)

**Three-Tier Processing Architecture:**

**Tier 1: CrewAI Agents (Primary)**
- Intelligent multi-agent processing
- Context-aware document retrieval
- Specialized agent roles for different tasks
- Processing time: 10-30 seconds

**Tier 2: Chat Engine (Fallback)**
- LlamaIndex conversation memory
- Context-aware responses
- Fast processing: 1-5 seconds
- Used when CrewAI fails

**Tier 3: Query Engine (Final Fallback)**
- Direct RAG without conversation context
- Simple document retrieval
- Fastest processing: 1-2 seconds
- Used when all other methods fail

### Conversation Memory System

**Memory Components:**
- **ChatMemoryBuffer**: LlamaIndex conversation state management
- **PostgreSQL Storage**: Persistent conversation history across sessions
- **Context Population**: Automatic conversion from OpenWebUI format

**Memory Flow:**
```
OpenWebUI Messages → Conversation History → ChatMemoryBuffer → CrewAI Agents → Context-Aware Response
```

### Query Cache (`services/cache_service.py`)

**Two-Layer Caching System** to avoid redundant LLM calls:

**Layer 1 — Exact Cache:**
- SHA256 hash of the query string
- O(1) lookup for identical questions
- Example: "What is procurement?" asked twice → instant cached response

**Layer 2 — Semantic Cache:**
- Cosine similarity of query embeddings (threshold: 0.95)
- Catches similar-but-not-identical questions
- Example: "What is procurement?" and "Explain procurement" → semantic cache hit

**Configuration:**
- TTL: 1 hour (configurable at runtime)
- Max size: 500 entries with LRU eviction
- Enable/disable and clear via `/api/cache/*` endpoints

**Cache Flow:**
```
User Query → Exact Cache (SHA256 hash match?)
                 ├── HIT → Return cached response instantly
                 └── MISS → Semantic Cache (embedding similarity > 0.95?)
                                ├── HIT → Return cached response
                                └── MISS → Full CrewAI pipeline → Cache response
```

### Chat Router (`routers/chat.py`)

**OpenWebUI Compatible Endpoints:**
- **POST /api/chat/completions** - Main chat endpoint with conversation support
- **GET /api/models** - Model listing for OpenWebUI integration
- **GET /health** - System health check

**Features:**
- **Conversation History Extraction**: Automatically extracts message history from OpenWebUI format
- **Memory Integration**: Passes conversation context to RAG service
- **Error Handling**: Comprehensive error responses and fallback mechanisms

## Request Flow

### Complete Processing Flow

```
1. OpenWebUI Request
   │
   ▼
2. Extract Conversation History
   ├── Current Message: "Give me a summary"
   └── Previous Context: [{"role": "user", "content": "What is procurement?"}, ...]
   │
   ▼
3. Populate Chat Memory
   └── ChatMemoryBuffer.put(previous_messages)
   │
   ▼
4. Three-Tier Processing
   ├── Tier 1: CrewAI Agents (Primary)
   │   ├── Query Analyzer: "FOLLOW_UP - search for procurement summary"
   │   ├── Document Retrieval: Search vector store for procurement docs
   │   ├── Information Extractor: Extract summary information
   │   └── Response Formatter: Format 3-line summary
   ├── Tier 2: Chat Engine (Fallback)
   │   └── Context-aware retrieval and response
   └── Tier 3: Query Engine (Final Fallback)
       └── Direct document retrieval
   │
   ▼
5. Intelligent Response
   └── LLM automatically relates current question to previous context
```

### Processing Modes

| Mode | Trigger | Memory | Speed | Use Case |
|------|---------|--------|-------|----------|
| CrewAI Agents | Default | Full | Medium (10-30s) | Complex analysis, intelligent processing |
| Chat Engine | CrewAI fails | Full | Fast (1-5s) | Conversational Q&A |
| Query Engine | All fail | None | Fast (1-2s) | Simple retrieval |

## Getting Started

### Prerequisites

- Docker and Docker Compose
- PostgreSQL with PGVector extension
- Ollama with required models

### Required Ollama Models

```bash
ollama pull llama3.2:1b              # Main LLM
ollama pull nomic-embed-text:v1.5    # Embedding model
```

### Environment Variables

Create `.env` file:

```env
# Database
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/agentic_rag

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2:1b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:v1.5

# Server Settings
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=*

# RAG Settings
SIMILARITY_TOP_K=5
MAX_TOKENS=4000
TEMPERATURE=0.1

# Conversation Memory
CONVERSATION_TOKEN_LIMIT=2000
MEMORY_CLEANUP_INTERVAL=100

# CrewAI Settings
CREW_VERBOSE=true
CREW_MEMORY=false

# Phoenix Observability (Optional)
PHOENIX_BASE_URL=http://localhost:6006
PHOENIX_PROJECT_NAME=agentic_rag_backend
```

### Running the Backend

```bash
# Using Docker Compose
docker-compose up rag_backend

# Local Development
pip install -r requirements.txt
python main.py
```

## API Endpoints

### Chat Completions (OpenWebUI Compatible)

```bash
POST /api/chat/completions
```

**Conversation Example:**
```json
{
  "model": "agentic-rag-ollama",
  "messages": [
    {"role": "user", "content": "What is procurement?"},
    {"role": "assistant", "content": "Procurement is the process of..."},
    {"role": "user", "content": "Give me a 3-line summary"}
  ],
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "agentic-rag-ollama",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "# Procurement Summary\n\n- Procurement is the process of acquiring goods and services\n- It involves planning, sourcing, and contract management\n- Key steps include requirement analysis, supplier selection, and negotiation\n\nSources\n1. Abu Dhabi Procurement Standards.PDF (relevance: 0.71)\n2. Procurement Manual (Business Process).PDF (relevance: 0.69)\n\nProcessed using: retrieval_specialist, response_generator, validator"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 85,
    "total_tokens": 235
  }
}
```

### Model Information

```bash
GET /api/models
```

**Response:**
```json
{
  "object": "list",
  "data": [{
    "id": "agentic-rag-ollama",
    "object": "model",
    "created": 1234567890,
    "owned_by": "agentic-rag",
    "permission": [],
    "root": "agentic-rag-ollama",
    "parent": null
  }]
}
```

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "agentic-rag-backend",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Conversation Examples

### Example 1: Follow-up Questions

**User:** "What are the main procurement policies?"
**Assistant:** *CrewAI agents analyze the question, retrieve relevant documents, and provide detailed procurement policy information*

**User:** "Can you summarize that in 5 bullet points?"
**Assistant:** *Query Analyzer identifies this as a FOLLOW_UP question, Document Retrieval Specialist searches for procurement summary information, Information Extractor creates 5 bullet points*

### Example 2: Context Switching

**User:** "Tell me about data protection requirements"
**Assistant:** *CrewAI agents process the new topic and provide data protection information*

**User:** "How does this relate to procurement?"
**Assistant:** *Query Analyzer identifies the relationship question, agents search for connections between data protection and procurement*

### Example 3: Complex Analysis

**User:** "Compare the approval processes for different procurement amounts"
**Assistant:** *CrewAI agents perform complex analysis across multiple documents to compare approval processes for different procurement thresholds*

## Configuration

### CrewAI Settings

```env
# CrewAI Configuration
CREW_VERBOSE=true          # Enable detailed agent logging
CREW_MEMORY=false          # Disable CrewAI memory (use conversation memory instead)

# Agent Processing
MAX_ITER=3                 # Maximum agent iterations
MAX_EXECUTION_TIME=300     # Agent timeout (seconds)
```

### Memory Management

```env
# Conversation memory settings
MAX_TOKENS=4000                    # Total token limit
CONVERSATION_TOKEN_LIMIT=2000      # Reserve for conversation memory
MEMORY_CLEANUP_INTERVAL=100        # Clean memory after N conversations
```

### Performance Optimization

```python
# Fast responses (CrewAI only)
SIMILARITY_TOP_K=3
CONVERSATION_TOKEN_LIMIT=1000

# Balanced performance
SIMILARITY_TOP_K=5
CONVERSATION_TOKEN_LIMIT=2000

# Maximum context retention
SIMILARITY_TOP_K=10
CONVERSATION_TOKEN_LIMIT=3000
```

## Development

### Testing the System

```bash
# Test chat completions
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "agentic-rag-ollama",
    "messages": [
      {"role": "user", "content": "What is procurement?"},
      {"role": "assistant", "content": "Procurement is..."},
      {"role": "user", "content": "Give me a brief summary"}
    ]
  }'

# Test model listing
curl http://localhost:8000/api/models

# Test health check
curl http://localhost:8000/health
```

### Custom Agent Integration

```python
from services.rag_service import RAGService

# Direct usage
rag_service = RAGService(config)
await rag_service.initialize()

# Chat with conversation history
conversation_history = [
    {"role": "user", "content": "Previous question"},
    {"role": "assistant", "content": "Previous response"}
]

result = await rag_service.chat(
    "Current question",
    conversation_history,
    "conversation_id"
)
```

## Monitoring & Observability

### Processing Metrics

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "event": "query_processed",
  "conversation_id": "conv_123",
  "processing_mode": "crew_ai_primary",
  "agents_used": ["retrieval_specialist", "response_generator", "validator"],
  "response_time_ms": 15000,
  "source_count": 3,
  "memory_token_usage": 1250
}
```

### Agent Performance Tracking

- **Agent Success Rate**: Track which agents complete successfully
- **Processing Time**: Monitor agent execution times
- **Fallback Frequency**: Track when CrewAI falls back to other methods
- **Context Understanding**: Measure conversation context accuracy

## Troubleshooting

### Common Issues

**CrewAI Agents Not Responding:**
- Check Ollama service status
- Verify database connection
- Review agent timeout settings
- Check CREW_VERBOSE logs

**Memory Issues:**
- Reduce `CONVERSATION_TOKEN_LIMIT`
- Check PostgreSQL connection
- Monitor memory usage in logs

**Slow Responses:**
- CrewAI agents take 10-30 seconds (normal)
- Check `MAX_EXECUTION_TIME` setting
- Consider reducing `SIMILARITY_TOP_K`

### Debug Mode

```env
LOG_LEVEL=DEBUG
CREW_VERBOSE=true
MEMORY_DEBUG=true
```

## Key Features

1. **CrewAI Primary Processing**: Intelligent multi-agent system for complex queries
2. **Intelligent Fallback**: Graceful degradation through three processing tiers
3. **Conversation Memory**: Full context awareness for follow-up questions
4. **Query Cache (Exact + Semantic)**: Two-layer caching to skip LLM calls for repeated/similar questions
5. **OpenWebUI Integration**: Seamless chat interface with OpenAI API compatibility
6. **Production Ready**: Comprehensive error handling, logging, and monitoring
7. **Clean Code**: Professional, symbol-free codebase ready for presentations

## Integration

### OpenWebUI Setup

1. **Add Backend URL**: `http://localhost:8000/api`
2. **API Key**: `dummy-key-for-agentic-rag`
3. **Select Model**: `agentic-rag-ollama`
4. **Start Conversation**: Natural back-and-forth chat with intelligent document processing

The system provides a true chatbot experience where you can have natural conversations with your documents, ask follow-up questions, and the AI will maintain context throughout the conversation using intelligent multi-agent processing.

## License

Part of the Agentic RAG System. See main project for license information.

---

**Built for intelligent document conversations with enterprise-grade multi-agent processing and comprehensive memory management**