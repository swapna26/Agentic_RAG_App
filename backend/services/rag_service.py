"""
Agentic RAG Service with CrewAI Integration

This module provides the core RAG (Retrieval Augmented Generation) service with intelligent
agentic capabilities using CrewAI. It serves as the primary interface for processing user
queries with advanced document retrieval and response generation.

Key Features:
- CrewAI agents as primary processing engine
- PostgreSQL vector store integration
- Conversation memory management
- Rate limiting and error handling
- Phoenix observability integration

Version: 1.0.0
"""

import asyncio
import time
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
import structlog
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage

from config import BackendConfig
from services.conversation_service import ConversationService
from services.cache_service import CacheService
from services.guardrails_service import GuardrailsService

logger = structlog.get_logger()


class RAGService:
    """
    Agentic RAG Service with CrewAI Integration
    
    This service provides intelligent document retrieval and response generation using
    CrewAI agents as the primary processing engine. It integrates with PostgreSQL for
    vector storage and conversation management.
    
    Architecture:
    1. CrewAI Agents (Primary) - Intelligent query processing
    2. Chat Engine (Fallback) - LlamaIndex conversation memory
    3. Query Engine (Final Fallback) - Basic RAG functionality
    
    Features:
    - Multi-agent processing with specialized roles
    - Conversation memory and context management
    - Rate limiting and error handling
    - Phoenix observability integration
    - PostgreSQL vector store integration
    """
    
    def __init__(self, config: BackendConfig):
        """Initialize the RAG service with configuration."""
        self.config = config
        
        # Core RAG components
        self.vector_store = None
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.embedding_model = None
        self.llm = None
        self.memory = None
        self.is_initialized = False
        
        # CrewAI agents (Primary processing engine)
        self.crew_agents = None
        
        # External services
        self.phoenix_service = None  # Injected from main.py
        self.conversation_service = None  # PostgreSQL conversation storage
        self.cache_service = None  # Injected from main.py
        self.guardrails_service = None  # Initialized during setup
        
        # Rate limiting configuration
        self.last_request_time = 0
        self.min_request_interval = getattr(config, 'ollama_min_request_interval', 0.1)
        self.request_count = 0
        self.request_window_start = time.time()
        self.max_requests_per_minute = getattr(config, 'ollama_requests_per_minute', 100)
        self.max_retries = getattr(config, 'ollama_retry_attempts', 3)
        self.retry_delay = getattr(config, 'ollama_retry_delay', 2)
    
    async def initialize(self):
        """
        Initialize the RAG service with all required components.
        
        This method sets up:
        1. PostgreSQL vector store connection
        2. Ollama embedding and LLM models
        3. LlamaIndex vector store index
        4. Chat engine with conversation memory
        5. CrewAI agents for intelligent processing
        6. Conversation service for PostgreSQL storage
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Agentic RAG service with CrewAI integration")

            # Initialize Ollama embedding model
            self.embedding_model = OllamaEmbedding(
                model_name=self.config.ollama_embedding_model,
                base_url=self.config.ollama_base_url,
            )

            # Initialize Ollama LLM
            self.llm = Ollama(
                model=self.config.ollama_model,
                base_url=self.config.ollama_base_url,
                temperature=self.config.temperature,
                request_timeout=120.0,
            )

            # Test the Ollama connection
            try:
                test_response = self.llm.complete("Hello")
                logger.info("Ollama API test successful", response_preview=str(test_response)[:50])
            except Exception as e:
                logger.warning("Ollama API test failed, continuing anyway", error=str(e))
            
            # Initialize PostgreSQL vector store with explicit configuration
            # LlamaIndex adds "data_" prefix, so use "llamaindex_vectors_copy" to get "data_llamaindex_vectors_copy"
            self.vector_store = PGVectorStore.from_params(
                database=self._extract_db_name(self.config.database_url),
                host=self._extract_host(self.config.database_url),
                port=self._extract_port(self.config.database_url),
                user=self._extract_user(self.config.database_url),
                password=self._extract_password(self.config.database_url),
                table_name="llamaindex_vectors_copy",
                embed_dim=768,  # nomic-embed-text:v1.5 dimension
                perform_setup=False,  # Don't try to create table - it already exists
                # Enable debugging to see what's happening
                debug=True,
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
            
            # Initialize vector index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )
            
            # Initialize query engine with retrieval settings
            self.query_engine = self.index.as_query_engine(
                llm=self.llm,
                similarity_top_k=self.config.similarity_top_k,
                response_mode="compact",
                verbose=True
            )
            
            # Initialize conversation memory
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=self.config.max_tokens // 2  # Reserve half for response
            )

            # Initialize chat engine with memory
            try:
                self.chat_engine = self.index.as_chat_engine(
                    chat_mode="context",
                    llm=self.llm,
                    memory=self.memory,
                    verbose=True
                )
                logger.info("Chat engine initialized successfully")
            except Exception as e:
                logger.warning("Failed to initialize chat engine, will use query engine fallback", error=str(e))
                self.chat_engine = None

            # Initialize conversation service
            self.conversation_service = ConversationService(self.config)
            await self.conversation_service.initialize()

            # Initialize cache service with embedding model for semantic matching
            if self.cache_service is None:
                self.cache_service = CacheService(
                    embedding_model=self.embedding_model,
                    ttl_seconds=3600,
                    max_size=500,
                    semantic_threshold=0.95,
                )
            else:
                # Cache was injected externally, set embedding model
                self.cache_service.embedding_model = self.embedding_model

            # Initialize guardrails service
            if self.guardrails_service is None:
                self.guardrails_service = GuardrailsService()
            logger.info("Guardrails service initialized")

            # Initialize CrewAI agents
            await self._initialize_crew_agents()

            self.is_initialized = True
            logger.info("RAG service initialized successfully with CrewAI agents and conversation storage")
            
        except Exception as e:
            logger.error("Failed to initialize RAG service", error=str(e))
            raise
    
    async def _initialize_crew_agents(self):
        """Initialize CrewAI agents."""
        try:
            from agents.crew_agents import RAGCrew
            
            # Initialize full CrewAI setup
            self.crew_agents = RAGCrew(self, self.config)
            
            # Crew agents initialized above
            
            logger.info("CrewAI agents initialized successfully")
            
        except Exception as e:
            logger.warning("Failed to initialize CrewAI agents", error=str(e))
            # Continue without agents - fall back to regular RAG
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        current_time = time.time()
        
        # Reset request count every minute
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Check requests per minute limit
        if self.request_count >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_window_start)
            logger.warning("Rate limit reached, waiting", wait_time=wait_time)
            await asyncio.sleep(wait_time)
            self.request_count = 0
            self.request_window_start = time.time()
        
        # Check minimum interval between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last
            logger.info("Rate limiting: waiting between requests", wait_time=wait_time)
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _extract_db_name(self, db_url: str) -> str:
        """Extract database name from database URL."""
        return db_url.split('/')[-1]
    
    def _extract_host(self, db_url: str) -> str:
        """Extract host from database URL."""
        return db_url.split('@')[1].split(':')[0]
    
    def _extract_port(self, db_url: str) -> int:
        """Extract port from database URL."""
        parts = db_url.split('@')[1].split(':')
        return int(parts[1].split('/')[0]) if len(parts) > 1 else 5432
    
    def _extract_user(self, db_url: str) -> str:
        """Extract user from database URL."""
        return db_url.split('://')[1].split(':')[0]
    
    def _extract_password(self, db_url: str) -> str:
        """Extract password from database URL."""
        return db_url.split('://')[1].split('@')[0].split(':')[1]

    def _is_greeting(self, message: str) -> bool:
        """Check if the message is a greeting."""
        message_lower = message.lower().strip()

        # Common greeting patterns
        greeting_patterns = [
            r'^(hi|hello|hey|hiya|howdy)[\s\.,!]*$',
            r'^(good\s+(morning|afternoon|evening|day)|good\s*day)[\s\.,!]*$',
            r'^(what\'s\s+up|whats\s+up|sup)[\s\.,!]*$',
            r'^(how\s+(are\s+you|r\s+u)|how\s+you\s+doing)[\s\.,!]*$',
            r'^(greetings?|salutations?)[\s\.,!]*$',
            r'^(nice\s+to\s+meet\s+you)[\s\.,!]*$'
        ]

        return any(re.match(pattern, message_lower) for pattern in greeting_patterns)

    # Topic change detection removed - now handled at router level for better consistency

    def _get_greeting_response(self) -> str:
        """Generate a friendly greeting response."""
        import random

        greetings = [
            "Hello! I'm your Agentic RAG Assistant. I can help you find information from the documents in my knowledge base. What would you like to know?",
            "Hi there! I'm here to assist you with questions about the documents I have access to. How can I help you today?",
            "Hey! Nice to meet you. I'm an AI assistant specialized in retrieving and analyzing information from various documents. What can I help you with?",
            "Hello! I'm ready to help you explore the knowledge base and answer your questions. What would you like to learn about?",
            "Hi! I'm your document assistant powered by advanced AI. I can search through documents and provide detailed answers. What's on your mind?"
        ]

        return random.choice(greetings)

    async def _populate_memory_from_history(self, conversation_history: List[Dict]):
        """Populate ChatMemoryBuffer with conversation history."""
        if not conversation_history:
            return

        try:
            # Clear existing memory
            self.memory.reset()

            # Add messages to memory
            for msg in conversation_history:
                chat_msg = ChatMessage(
                    role=msg["role"],
                    content=msg["content"]
                )
                self.memory.put(chat_msg)

            logger.info("Populated memory with conversation history",
                       message_count=len(conversation_history))
        except Exception as e:
            logger.error("Failed to populate memory from history", error=str(e))

    async def _get_conversation_context(self, conversation_id: str) -> str:
        """Get conversation context for the given conversation ID using PostgreSQL."""
        if not self.conversation_service:
            return ""

        try:
            return await self.conversation_service.get_conversation_context(
                conversation_id,
                max_messages=10,  # Last 5 exchanges
                include_sources=False
            )
        except Exception as e:
            logger.error("Failed to get conversation context", error=str(e))
            return ""

    def _should_include_context(self, question: str, conversation_history: List[Dict]) -> bool:
        """Heuristically decide whether to include prior context for this question.

        Uses lightweight lexical overlap with the most recent user message to avoid
        contaminating new-topic queries with old context.
        """
        if not conversation_history:
            return False

        # Look for an explicit reset/new topic signal
        lower_q = (question or "").lower()
        explicit_reset_phrases = ["new topic", "start over", "ignore previous", "no relation", "unrelated"]
        if any(p in lower_q for p in explicit_reset_phrases):
            return False

        # Find last user message content
        last_user = next((m for m in reversed(conversation_history) if m.get("role") == "user" and m.get("content")), None)
        if not last_user:
            return False

        import re
        def tokenize(text: str) -> set:
            tokens = re.findall(r"[a-zA-Z0-9]+", (text or "").lower())
            # Remove trivial short tokens
            return {t for t in tokens if len(t) > 2}

        q_tokens = tokenize(question)
        prev_tokens = tokenize(last_user.get("content", ""))
        if not q_tokens or not prev_tokens:
            return False

        overlap = len(q_tokens & prev_tokens) / max(1, len(q_tokens | prev_tokens))
        # Include context only if sufficient overlap (threshold tuned small)
        return overlap >= 0.2

    def _clean_conversation_context(self, raw_context: str) -> str:
        """Clean conversation context to remove meta-commentary and focus on substance."""
        if not raw_context:
            return ""

        lines = raw_context.split('\n')
        cleaned_lines = []

        # Patterns that indicate meta-commentary rather than actual answers
        meta_commentary_patterns = [
            r'^(Thoughtful|Excellent|Great|Fantastic|Well done|This is a|Perfect|Outstanding)',
            r'(comprehensive|well-structured|detailed|thorough) response',
            r'(analysis|breakdown|explanation) you\'ve provided',
            r'quality.*response',
            r'addresses.*question.*well',
            r'clear.*comprehensive.*answer',
            r'excellent.*coverage'
        ]

        import re

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Skip lines that are meta-commentary
            is_meta_commentary = False
            for pattern in meta_commentary_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    is_meta_commentary = True
                    break

            # Also skip very short responses that look like feedback
            if len(line) < 20 and any(word in line.lower() for word in ['great', 'excellent', 'good', 'nice', 'perfect']):
                is_meta_commentary = True

            if not is_meta_commentary:
                cleaned_lines.append(line)

        cleaned_context = '\n'.join(cleaned_lines)

        # Limit context length to prevent overwhelming the agents
        if len(cleaned_context) > 2000:
            # Take the most recent 2000 characters
            cleaned_context = cleaned_context[-2000:]
            # Try to start at a complete message boundary
            user_pos = cleaned_context.find('User:')
            if user_pos > 0:
                cleaned_context = cleaned_context[user_pos:]

        return cleaned_context

    async def _add_to_conversation(self, conversation_id: str, role: str, content: str,
                                 sources: Optional[List[Dict]] = None,
                                 processing_mode: Optional[str] = None,
                                 response_time_ms: Optional[int] = None):
        """Add a message to the conversation history using PostgreSQL."""
        if not self.conversation_service:
            return

        try:
            await self.conversation_service.add_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                sources=sources,
                processing_mode=processing_mode,
                model_used=self.config.ollama_model if role == "assistant" else None,
                response_time_ms=response_time_ms
            )
        except Exception as e:
            logger.error("Failed to add message to conversation", error=str(e))

    async def _manual_vector_search(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform manual vector similarity search as fallback."""
        try:
            import asyncpg

            # Get embedding for the query
            query_embedding = self.embedding_model.get_text_embedding(question)
            query_vector = f"[{','.join(map(str, query_embedding))}]"

            # Connect to database
            conn = await asyncpg.connect(self.config.database_url)

            try:
                # Search with cosine similarity
                sql = """
                SELECT
                    text,
                    metadata_,
                    node_id,
                    1 - (embedding <=> $1::vector) as similarity
                FROM data_llamaindex_vectors_copy
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """

                results = await conn.fetch(sql, query_vector, top_k)

                sources = []
                for row in results:
                    # Create a source entry
                    source_info = {
                        "content": row['text'][:200] + "..." if len(row['text']) > 200 else row['text'],
                        "score": float(row['similarity']),
                        "metadata": row['metadata_'] if row['metadata_'] else {},
                        "node_id": row['node_id'],
                        "text": row['text']  # Full text for context
                    }
                    sources.append(source_info)

                return sources

            finally:
                await conn.close()

        except Exception as e:
            logger.error("Manual vector search failed", error=str(e))
            return []

    async def chat(self, question: str, conversation_history: List[Dict], conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Process user query using intelligent agentic RAG system.
        
        This method implements a three-tier processing approach:
        1. CrewAI Agents (Primary) - Intelligent multi-agent processing
        2. Chat Engine (Fallback) - LlamaIndex conversation memory
        3. Query Engine (Final Fallback) - Basic RAG functionality
        
        Args:
            question (str): User's question or query
            conversation_history (List[Dict]): Previous conversation messages
            conversation_id (Optional[str]): Unique conversation identifier
            
        Returns:
            Dict[str, Any]: Response containing answer, sources, and metadata
            
        Raises:
            RuntimeError: If service not initialized
        """
        if not self.is_initialized:
            raise RuntimeError("RAG service not initialized")

        try:
            logger.info("Processing chat message", question=question[:100],
                       history_length=len(conversation_history))

            # Check if this is a greeting and handle it without agents
            if self._is_greeting(question):
                logger.info("Detected greeting message, responding directly")
                greeting_response = self._get_greeting_response()

                # Add to conversation history if conversation_id provided
                if conversation_id:
                    await self._add_to_conversation(
                        conversation_id, "user", question, processing_mode="greeting_request"
                    )
                    await self._add_to_conversation(
                        conversation_id, "assistant", greeting_response, processing_mode="greeting_response"
                    )

                result = {
                    "response": greeting_response,
                    "sources": [],
                    "metadata": {
                        "model": self.config.ollama_model,
                        "conversation_id": conversation_id,
                        "source_count": 0,
                        "processing_mode": "greeting_response"
                    }
                }

                # Log greeting interaction to Phoenix
                if self.phoenix_service:
                    await self.phoenix_service.log_chat_interaction(
                        conversation_id or "unknown",
                        question,
                        greeting_response,
                        [],
                        result["metadata"]
                    )

                return result

            # Run input guardrails before any processing
            if self.guardrails_service:
                guardrail_result = self.guardrails_service.check_input(question)
                if not guardrail_result.passed:
                    logger.warning("Query blocked by guardrails",
                                 guardrail=guardrail_result.guardrail_name,
                                 reason=guardrail_result.message)
                    return {
                        "response": guardrail_result.message,
                        "sources": [],
                        "metadata": {
                            "model": self.config.ollama_model,
                            "conversation_id": conversation_id,
                            "source_count": 0,
                            "processing_mode": "guardrail_blocked",
                            "guardrail": guardrail_result.guardrail_name,
                        }
                    }

            # Check cache before expensive processing
            if self.cache_service:
                cached = await self.cache_service.get(question)
                if cached:
                    logger.info("Serving response from cache", question=question[:80])
                    if conversation_id:
                        await self._add_to_conversation(
                            conversation_id, "user", question, processing_mode="cached"
                        )
                        await self._add_to_conversation(
                            conversation_id, "assistant", cached["response"],
                            sources=cached.get("sources", []),
                            processing_mode="cached_response"
                        )
                    return cached

            # Populate memory with conversation history
            await self._populate_memory_from_history(conversation_history)

            # Apply rate limiting for non-greeting queries
            await self._check_rate_limit()

            # Start tracing span for the query
            trace_span = None
            if self.phoenix_service:
                trace_span = self.phoenix_service.create_trace_span(
                    "rag_chat",
                    {
                        "query": question[:100],
                        "conversation_id": conversation_id,
                        "history_length": len(conversation_history)
                    }
                )

            # Use CrewAI agents as primary service
            if self.crew_agents:
                try:
                    logger.info("Using CrewAI agents for intelligent processing")

                    # Build enriched query with recent conversation context (plain text, no markdown)
                    enriched_query = question
                    if conversation_history and self._should_include_context(question, conversation_history):
                        # Take last few turns to provide context
                        recent = conversation_history[-6:]
                        ctx_lines = []
                        for msg in recent:
                            role = msg.get("role", "user").lower()
                            content = (msg.get("content", "") or "").strip()
                            if not content:
                                continue
                            if role == "user":
                                ctx_lines.append(f"User: {content}")
                            elif role == "assistant":
                                ctx_lines.append(f"Assistant: {content}")
                            else:
                                ctx_lines.append(f"{role.capitalize()}: {content}")
                        ctx_text = "\n".join(ctx_lines)
                        enriched_query = (
                            "Previous conversation context:\n"
                            f"{ctx_text}\n\n"
                            "Current question: " + question
                        )

                    # Process query with CrewAI agents using enriched context
                    result = await self.crew_agents.process_query(enriched_query)

                    # Add to conversation history if conversation_id provided
                    if conversation_id:
                        await self._add_to_conversation(
                            conversation_id, "user", question, processing_mode="crew_ai_primary"
                        )
                        await self._add_to_conversation(
                            conversation_id, "assistant", result['response'],
                            sources=result.get('sources', []),
                            processing_mode="crew_ai_primary_response"
                        )

                    # Ensure metadata is properly set
                    if "metadata" not in result:
                        result["metadata"] = {}
                    result["metadata"].update({
                        "model": self.config.ollama_model,
                        "conversation_id": conversation_id,
                        "source_count": len(result.get('sources', [])),
                        "processing_mode": "crew_ai_primary",
                        "has_conversation_context": len(conversation_history) > 0
                    })

                    # Log to Phoenix if available
                    if self.phoenix_service:
                        await self.phoenix_service.log_chat_interaction(
                            conversation_id or "unknown",
                            question,
                            result["response"],
                            result.get('sources', []),
                            result.get("metadata", {})
                        )

                    # Run output guardrails before returning
                    if self.guardrails_service:
                        output_result = self.guardrails_service.check_output(
                            result["response"],
                            result.get("sources", []),
                            question,
                        )
                        result["response"] = output_result.details.get(
                            "cleaned_response", result["response"]
                        )
                        warnings = output_result.details.get("warnings", [])
                        if warnings:
                            result["metadata"]["guardrail_warnings"] = warnings

                    # Store in cache for future identical/similar queries
                    if self.cache_service:
                        await self.cache_service.put(question, result)

                    return result

                except Exception as e:
                    logger.warning("CrewAI agents failed, falling back to chat engine", error=str(e))

            # Fallback to chat engine with conversation memory if available
            if self.chat_engine:
                try:
                    logger.info("Using chat engine as fallback with conversation memory")

                    # Use the chat engine that maintains conversation context
                    response = await asyncio.to_thread(
                        self.chat_engine.chat,
                        question
                    )

                    # Extract source information
                    sources = []
                    if hasattr(response, 'source_nodes') and response.source_nodes:
                        for node in response.source_nodes:
                            source_info = {
                                "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                                "score": float(node.score) if hasattr(node, 'score') else 1.0,
                                "metadata": node.metadata
                            }
                            sources.append(source_info)

                    response_text = str(response.response)

                    # Add to conversation history if conversation_id provided
                    if conversation_id:
                        await self._add_to_conversation(
                            conversation_id, "user", question, processing_mode="chat_engine_fallback"
                        )
                        await self._add_to_conversation(
                            conversation_id, "assistant", response_text,
                            sources=sources,
                            processing_mode="chat_engine_fallback_response"
                        )

                    result = {
                        "response": response_text,
                        "sources": sources,
                        "metadata": {
                            "model": self.config.ollama_model,
                            "conversation_id": conversation_id,
                            "source_count": len(sources),
                            "processing_mode": "chat_engine_fallback",
                            "has_conversation_context": len(conversation_history) > 0
                        }
                    }

                    # Log to Phoenix if available
                    if self.phoenix_service:
                        await self.phoenix_service.log_chat_interaction(
                            conversation_id or "unknown",
                            question,
                            result["response"],
                            sources,
                            result.get("metadata", {})
                        )

                    return result

                except Exception as e:
                    logger.warning("Chat engine fallback also failed", error=str(e))

            # Final fallback to regular query engine
            logger.info("Using fallback query engine")
            response = await asyncio.to_thread(
                self.query_engine.query,
                question
            )

            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for node in response.source_nodes:
                    source_info = {
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(node.score) if hasattr(node, 'score') else 1.0,
                        "metadata": node.metadata
                    }
                    sources.append(source_info)

            response_text = str(response.response)

            result = {
                "response": response_text,
                "sources": sources,
                "metadata": {
                    "model": self.config.ollama_model,
                    "conversation_id": conversation_id,
                    "source_count": len(sources),
                    "processing_mode": "fallback_query_engine",
                    "has_conversation_context": len(conversation_history) > 0
                }
            }

            # Log to Phoenix if available
            if self.phoenix_service:
                await self.phoenix_service.log_chat_interaction(
                    conversation_id or "unknown",
                    question,
                    result["response"],
                    sources,
                    result.get("metadata", {})
                )

            # End tracing span
            if trace_span:
                trace_span.set_attribute("response.length", len(result["response"]))
                trace_span.set_attribute("sources.count", len(sources))
                trace_span.end()

            logger.info("Chat processed successfully",
                       response_length=len(result["response"]),
                       source_count=len(sources))

            return result

        except Exception as e:
            logger.error("Chat processing failed", error=str(e))
            raise

    async def query(self, question: str, conversation_id: Optional[str] = None, use_agents: bool = True) -> Dict[str, Any]:
        """Legacy query method - redirects to chat with empty history."""
        return await self.chat(question, [], conversation_id)

    async def query_legacy(self, question: str, conversation_id: Optional[str] = None, use_agents: bool = True) -> Dict[str, Any]:
        """Original query method preserved for backward compatibility."""
        if not self.is_initialized:
            raise RuntimeError("RAG service not initialized")

        try:
            logger.info("Processing query", question=question[:100], use_agents=use_agents)

            # Check if this is a greeting and handle it without agents
            if self._is_greeting(question):
                logger.info("Detected greeting message, responding directly")
                greeting_response = self._get_greeting_response()

                # Add to conversation history if conversation_id provided
                if conversation_id:
                    await self._add_to_conversation(
                        conversation_id, "user", question, processing_mode="greeting_request"
                    )
                    await self._add_to_conversation(
                        conversation_id, "assistant", greeting_response, processing_mode="greeting_response"
                    )

                result = {
                    "response": greeting_response,
                    "sources": [],
                    "metadata": {
                        "model": self.config.ollama_model,
                        "conversation_id": conversation_id,
                        "source_count": 0,
                        "processing_mode": "greeting_response"
                    }
                }

                # Log greeting interaction to Phoenix
                if self.phoenix_service:
                    await self.phoenix_service.log_chat_interaction(
                        conversation_id or "unknown",
                        question,
                        greeting_response,
                        [],
                        result["metadata"]
                    )

                return result

            # Apply rate limiting for non-greeting queries
            await self._check_rate_limit()

            # Start tracing span for the query
            trace_span = None
            if self.phoenix_service:
                trace_span = self.phoenix_service.create_trace_span(
                    "rag_query",
                    {
                        "query": question[:100],
                        "use_agents": use_agents,
                        "conversation_id": conversation_id
                    }
                )

            # Get conversation context
            conversation_context = ""
            if conversation_id:
                conversation_context = await self._get_conversation_context(conversation_id)

            # Use CrewAI agents if available and requested
            if use_agents and self.crew_agents:
                try:
                    # Prepare enriched query with conversation context
                    enriched_query = question
                    if conversation_context:
                        enriched_query = f"Previous conversation context:\n{conversation_context}\n\nCurrent question: {question}"
                        logger.info("Adding conversation context to query", context_length=len(conversation_context))

                    result = await self.crew_agents.process_query(enriched_query)

                    # Track response time for metrics
                    response_time_ms = int((time.time() - self.last_request_time) * 1000)

                    # Add to conversation history if conversation_id provided
                    if conversation_id:
                        await self._add_to_conversation(
                            conversation_id, "user", question, processing_mode="crew_ai_query"
                        )
                        await self._add_to_conversation(
                            conversation_id, "assistant", result['response'],
                            sources=result.get('sources', []),
                            processing_mode="crew_ai_response",
                            response_time_ms=response_time_ms
                        )

                        # Also add to LlamaIndex memory for consistency
                        self.memory.put(f"Human: {question}")
                        self.memory.put(f"Assistant: {result['response']}")

                    # Update metadata to indicate context was used
                    if conversation_context:
                        result["metadata"]["has_conversation_context"] = True
                        result["metadata"]["context_length"] = len(conversation_context)

                    return result

                except Exception as e:
                    logger.warning("CrewAI processing failed, falling back to regular RAG", error=str(e))
                    # Fall through to regular RAG processing

            # Regular RAG processing with retry logic
            for attempt in range(self.max_retries):
                try:
                    # Get response from query engine
                    logger.info("Attempting LlamaIndex query", attempt=attempt + 1)
                    response = await asyncio.to_thread(
                        self.query_engine.query,
                        question
                    )

                    logger.info("LlamaIndex query completed",
                               response_type=type(response).__name__,
                               has_source_nodes=hasattr(response, 'source_nodes'),
                               source_nodes_count=len(response.source_nodes) if hasattr(response, 'source_nodes') and response.source_nodes else 0)

                    break  # Success, exit retry loop

                except Exception as e:
                    logger.warning("Ollama request failed",
                                 attempt=attempt + 1,
                                 max_retries=self.max_retries,
                                 error=str(e))

                    if attempt < self.max_retries - 1:
                        wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.info("Retrying after backoff", wait_time=wait_time)
                        await asyncio.sleep(wait_time)
                    else:
                        # Last attempt failed, return a helpful message
                        return {
                            "response": "I apologize, but I'm currently experiencing issues connecting to the language model. Please try again in a moment.",
                            "sources": [],
                            "metadata": {
                                "model": self.config.ollama_model,
                                "conversation_id": conversation_id,
                                "error": "connection_failed",
                                "retry_after": 30
                            }
                        }

            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes') and response.source_nodes:
                logger.info("Processing source nodes", count=len(response.source_nodes))
                for i, node in enumerate(response.source_nodes):
                    source_info = {
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(node.score) if hasattr(node, 'score') else 1.0,
                        "metadata": node.metadata
                    }
                    sources.append(source_info)
                    logger.info(f"Source {i+1}", score=source_info["score"], content_preview=source_info["content"][:50])
            else:
                logger.warning("No source nodes found in LlamaIndex response")

            response_text = str(response.response)

            # Add to memory if conversation_id provided
            if conversation_id:
                self.memory.put(f"Human: {question}")
                self.memory.put(f"Assistant: {response_text}")

            result = {
                "response": response_text,
                "sources": sources,
                "metadata": {
                    "model": self.config.ollama_model,
                    "conversation_id": conversation_id,
                    "source_count": len(sources),
                    "processing_mode": "regular_rag"
                }
            }

            # Log to Phoenix if available
            if self.phoenix_service:
                # Log complete chat interaction
                await self.phoenix_service.log_chat_interaction(
                    conversation_id or "unknown",
                    question,
                    result["response"],
                    sources,
                    result.get("metadata", {})
                )

                # Log document retrieval
                if sources:
                    await self.phoenix_service.log_document_retrieval(
                        question,
                        sources,
                        0.0,  # We don't track retrieval time separately yet
                        {"use_agents": use_agents}
                    )

                # Log basic prompt execution
                await self.phoenix_service.log_prompt_execution(
                    "rag_query_complete",
                    {"query": question},
                    result["response"],
                    {
                        "conversation_id": conversation_id,
                        "use_agents": use_agents,
                        "source_count": len(sources),
                        "processing_mode": result["metadata"]["processing_mode"]
                    }
                )

            # End tracing span
            if trace_span:
                trace_span.set_attribute("response.length", len(result["response"]))
                trace_span.set_attribute("sources.count", len(sources))
                trace_span.end()

            logger.info("Query processed successfully",
                       response_length=len(result["response"]),
                       source_count=len(sources))

            return result

        except Exception as e:
            logger.error("Query processing failed", error=str(e))
            raise
    
    async def query_with_full_agents(self, question: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Query using full CrewAI agent workflow (slower but more thorough)."""
        if not self.is_initialized or not self.crew_agents:
            return await self.query(question, conversation_id, use_agents=False)

        try:
            # Check if this is a greeting first
            if self._is_greeting(question):
                logger.info("Detected greeting message, responding directly")
                greeting_response = self._get_greeting_response()

                if conversation_id:
                    await self._add_to_conversation(conversation_id, "user", question)
                    await self._add_to_conversation(conversation_id, "assistant", greeting_response)

                result = {
                    "response": greeting_response,
                    "sources": [],
                    "metadata": {
                        "model": self.config.ollama_model,
                        "conversation_id": conversation_id,
                        "source_count": 0,
                        "processing_mode": "greeting_response"
                    }
                }

                # Log greeting interaction to Phoenix
                if self.phoenix_service:
                    await self.phoenix_service.log_chat_interaction(
                        conversation_id or "unknown",
                        question,
                        greeting_response,
                        [],
                        result["metadata"]
                    )

                return result

            await self._check_rate_limit()

            logger.info("Processing query with full CrewAI agents", question=question[:100])

            # Get and properly format conversation context to prevent agent confusion
            conversation_context = ""
            use_context = False
            topic_changed = False  # Initialize topic_changed variable


            if conversation_id:
                raw_context = await self._get_conversation_context(conversation_id)

                # Since topic change detection is already handled at router level,
                # we always use the conversation context if it exists
                conversation_context = self._clean_conversation_context(raw_context) if raw_context else ""
                use_context = bool(conversation_context)
                topic_changed = False  # Topic changes handled at router level


            # Always pass full conversation context to agents - let them handle conversation analysis
            enriched_query = question
            if conversation_context:
                enriched_query = f"""## Full Conversation History
{conversation_context}

## Current User Question/Request
{question}

INSTRUCTIONS FOR QUERY ANALYSER:
- Analyze the FULL conversation history above
- Determine if the current question is:
  a) A follow-up question (clarification, different format, more details about same topic)
  b) A completely new question (different topic/domain)
- Provide appropriate search strategy for the Document Retriever

INSTRUCTIONS FOR DOCUMENT RETRIEVER:
- For follow-ups: Consider context from previous questions in this conversation
- For new questions: Focus search on the new topic independently
- Use the Query Analyser's guidance for search strategy

INSTRUCTIONS FOR RESPONSE GENERATOR:
- For follow-ups: Reference previous context appropriately and provide the requested format/details
- For new questions: Provide comprehensive answer about the new topic
- Always base answers on retrieved documents"""
                logger.info("CONTEXT SUCCESS: Passing full conversation context to agents",
                          context_length=len(conversation_context),
                          context_preview=conversation_context[:100])
            else:
                logger.info("CONTEXT MISSING: No conversation context available for multi-message request",
                          conversation_id=conversation_id,
                          question=question[:50])

            import time
            start_time = time.time()
            result = await self.crew_agents.process_query(enriched_query)
            execution_time = time.time() - start_time

            # Add to conversation history if conversation_id provided
            if conversation_id:
                logger.info("Saving conversation to database",
                           conversation_id=conversation_id,
                           question=question[:50],
                           response_length=len(result['response']))
                await self._add_to_conversation(conversation_id, "user", question)
                await self._add_to_conversation(conversation_id, "assistant", result['response'])

                # Also add to LlamaIndex memory for consistency
                self.memory.put(f"Human: {question}")
                self.memory.put(f"Assistant: {result['response']}")

            # Update metadata to indicate context usage and topic change detection
            if use_context and conversation_context:
                result["metadata"]["has_conversation_context"] = True
                result["metadata"]["context_length"] = len(conversation_context)
                result["metadata"]["topic_changed"] = False
            else:
                result["metadata"]["has_conversation_context"] = False
                result["metadata"]["context_length"] = 0
                result["metadata"]["topic_changed"] = topic_changed if conversation_id else False

            # Log to Phoenix if available
            if self.phoenix_service:
                # Log complete chat interaction
                await self.phoenix_service.log_chat_interaction(
                    conversation_id or "unknown",
                    question,
                    result["response"],
                    result.get("sources", []),
                    result.get("metadata", {})
                )

                # Log agent workflow
                agents_used = result.get("metadata", {}).get("agents_used", [])
                await self.phoenix_service.log_agent_workflow(
                    "full_crewai_agents",
                    question,
                    agents_used,
                    execution_time,
                    result,
                    {
                        "conversation_id": conversation_id,
                        "has_conversation_context": bool(conversation_context),
                        "context_length": len(conversation_context) if conversation_context else 0
                    }
                )

                # Log document retrieval if sources available
                if result.get("sources"):
                    await self.phoenix_service.log_document_retrieval(
                        question,
                        result["sources"],
                        0.0,  # CrewAI doesn't track retrieval time separately
                        {"workflow_type": "full_agents"}
                    )

            return result
            
        except Exception as e:
            logger.error("Full CrewAI processing failed", error=str(e))
            # Fallback to regular processing
            return await self.query(question, conversation_id, use_agents=False)
    
    async def stream_chat(self, question: str, conversation_history: List[Dict], conversation_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream chat response for real-time UI updates."""
        if not self.is_initialized:
            raise RuntimeError("RAG service not initialized")

        try:
            logger.info("Processing streaming chat", question=question[:100])

            # For now, we'll get the full response and stream it
            # In production, you'd implement proper streaming with the LLM
            result = await self.chat(question, conversation_history, conversation_id)
            
            # Check if we got an error response
            if result.get("metadata", {}).get("error") == "quota_exceeded":
                yield result["response"]
                return
            
            # Yield response in chunks
            response_text = result["response"]
            chunk_size = 50  # Characters per chunk
            
            for i in range(0, len(response_text), chunk_size):
                chunk = response_text[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.05)  # Small delay for streaming effect
            
        except Exception as e:
            logger.error("Streaming chat failed", error=str(e))
            yield f"Error: {str(e)}"

    async def stream_query(self, question: str, conversation_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Legacy stream method for backward compatibility."""
        async for chunk in self.stream_chat(question, [], conversation_id):
            yield chunk

    async def get_conversation_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get conversation history from PostgreSQL."""
        try:
            if self.conversation_service:
                return await self.conversation_service.get_conversation_history(conversation_id)
            else:
                # Fallback to LlamaIndex memory
                messages = self.memory.get_all()
                history = []

                for i in range(0, len(messages), 2):
                    if i + 1 < len(messages):
                        history.append({
                            "user": messages[i].content.replace("Human: ", ""),
                            "assistant": messages[i + 1].content.replace("Assistant: ", ""),
                            "timestamp": messages[i].additional_kwargs.get("timestamp", "")
                        })

                return history

        except Exception as e:
            logger.error("Failed to get conversation history", error=str(e))
            return []
    
    async def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history from PostgreSQL."""
        try:
            if self.conversation_service:
                result = await self.conversation_service.clear_conversation(conversation_id)
                if result:
                    # Also clear LlamaIndex memory
                    self.memory.clear()
                logger.info("Conversation cleared", conversation_id=conversation_id, success=result)
                return result
            else:
                # Fallback to clearing only LlamaIndex memory
                self.memory.clear()
                logger.info("Conversation cleared (memory only)", conversation_id=conversation_id)
                return True
        except Exception as e:
            logger.error("Failed to clear conversation", error=str(e))
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            return {
                "status": "healthy",
                "total_documents": 5,  # From your indexer
                "total_chunks": 482,   # From your indexer
                "embedding_model": self.config.ollama_embedding_model,
                "llm_model": self.config.ollama_model,
                "crew_agents_enabled": bool(self.crew_agents),
                "phoenix_service_enabled": bool(self.phoenix_service),
                "rate_limit_info": {
                    "requests_this_minute": self.request_count,
                    "max_requests_per_minute": self.max_requests_per_minute,
                    "min_request_interval": self.min_request_interval
                }
            }
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            return {"status": "error", "error": str(e)}
    
    async def is_healthy(self) -> bool:
        """Check if the service is healthy."""
        try:
            if not self.is_initialized:
                return False
            
            # Just return True for now to avoid triggering quota limits in health checks
            # In production, you might want a lightweight health check
            return True
            
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def get_available_prompts(self) -> List[Dict[str, Any]]:
        """Get available prompts from Phoenix service."""
        try:
            if self.phoenix_service:
                prompts = await self.phoenix_service.get_available_prompts()
                return [
                    {
                        "id": prompt.id,
                        "name": prompt.name,
                        "description": prompt.description,
                        "variables": prompt.variables,
                        "tags": prompt.tags
                    }
                    for prompt in prompts
                ]
            return []
        except Exception as e:
            logger.error("Failed to get available prompts", error=str(e))
            return []
    
    async def render_prompt_with_phoenix(self, prompt_id: str, variables: Dict[str, str]) -> Optional[str]:
        """Render a prompt using Phoenix service."""
        try:
            if self.phoenix_service:
                return await self.phoenix_service.render_prompt(prompt_id, variables)
            return None
        except Exception as e:
            logger.error("Failed to render prompt", prompt_id=prompt_id, error=str(e))
            return None
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.vector_store:
                # Close database connections if needed
                pass
            self.is_initialized = False
            logger.info("RAG service cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))