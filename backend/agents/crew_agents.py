"""
CrewAI Agents for Intelligent RAG Processing

This module implements a multi-agent system using CrewAI for intelligent document
retrieval and response generation. The agents work collaboratively to provide
comprehensive and accurate answers to user queries.

Agent Architecture:
1. Document Retrieval Agent - Specialized in finding relevant documents
2. Analysis Agent - Analyzes and processes retrieved information
3. Response Generation Agent - Creates comprehensive responses


Version: 1.0.0
"""

import os
from typing import Dict, Any, List
import structlog
from urllib.parse import urlparse
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from config import BackendConfig
from services.fraud_detection_service import FraudDetectionService

# Explicitly disable OpenAI for CrewAI to prevent API key errors
os.environ['OPENAI_API_KEY'] = ''
os.environ['OPENAI_API_BASE'] = ''

logger = structlog.get_logger()


class RAGCrew:
    """
    CrewAI Multi-Agent System for Intelligent RAG Processing

    This class orchestrates a team of specialized AI agents that work collaboratively
    to provide comprehensive and accurate responses to user queries. Each agent has
    a specific role in the document retrieval and response generation process.

    Agent Team:
    1. Document Retrieval Agent - Finds relevant documents from vector store
    2. Analysis Agent - Analyzes and processes retrieved information
    3. Response Generation Agent - Creates comprehensive, well-structured responses

    Features:
    - Multi-agent collaboration with specialized roles
    - PostgreSQL vector store integration
    - Ollama LLM integration for local processing
    - Prompt Management integration via PromptStore
    - Comprehensive error handling and logging
    """

    def __init__(self, rag_service, config):
        """
        Initialize the RAG Crew with specialized agents.

        Args:
            rag_service: Reference to the main RAG service
            config: Configuration object with database and model settings
        """
        self.rag_service = rag_service
        self.config = config
        self.crew = None

        # Get prompt_store and phoenix_service from rag_service if available
        self.prompt_store = getattr(rag_service, 'prompt_store', None) or \
                           (getattr(rag_service, 'phoenix_service', None) and
                            getattr(rag_service.phoenix_service, 'prompt_store', None))
        self.phoenix_service = getattr(rag_service, 'phoenix_service', None)

        # Configure Ollama LLM for CrewAI agents
        self.llm = LLM(
            model=f"ollama/{self.config.ollama_model}",
            api_base=self.config.ollama_base_url,
            temperature=config.temperature,
            max_tokens=1500,
            timeout=300,
            max_retries=2
        )

        # Initialize the agent team
        self._initialize_agents()

    def _create_document_retrieval_tool(self):
        """Create a document retrieval tool using the @tool decorator."""

        @tool("Document Retrieval Tool")
        def document_retrieval_tool(query: str) -> str:
            """Retrieves relevant context from a collection of policy and standards documents. Use this tool to search for information in policy documents, manuals, and standards.

            Args:
                query: The search query to find relevant documents
            """
            try:
                logger.info("Document retrieval tool called", query=query[:100])

                # Validate we have a proper query string
                if not query or not query.strip():
                    return "Error: Search query cannot be empty."

                search_query = query.strip()

                # Check if we got a placeholder description instead of real query
                placeholder_queries = [
                    "The search query to find relevant documents", 
                    "Search query",
                    "query",
                    "search"
                ]
                if search_query.lower() in [p.lower() for p in placeholder_queries]:
                    return "Error: Please provide a specific search query."
                
                # Use your RAG service's existing database configuration
                DATABASE_URL = self.config.database_url
                db_url_parts = urlparse(DATABASE_URL)

                logger.info("Using RAG service database connection", 
                           host=db_url_parts.hostname,
                           port=db_url_parts.port,
                           database=db_url_parts.path.lstrip('/'),
                           user=db_url_parts.username)
                
                # Initialize the vector store with your configuration
                vector_store = PGVectorStore.from_params(
                    host=db_url_parts.hostname,
                    port=db_url_parts.port,
                    database=db_url_parts.path.lstrip('/'),
                    user=db_url_parts.username,
                    password=db_url_parts.password,
                    table_name="llamaindex_vectors_copy",
                    embed_dim=768,
                )

                # Initialize Ollama embedding model using your config
                embed_model = OllamaEmbedding(
                    model_name=self.config.ollama_embedding_model,
                    base_url=self.config.ollama_base_url,
                )

                # Create a LlamaIndex VectorStoreIndex object from the vector store
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    embed_model=embed_model
                )

                # Use retriever directly for document retrieval
                retriever = index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k,
                    verbose=True
                )

                # Query the index to retrieve nodes directly
                retrieved_nodes = retriever.retrieve(search_query)
                
                if not retrieved_nodes:
                    return f"No relevant documents found for query: '{search_query}'. Please try different keywords or check if documents are properly indexed."
                
                # Format the retrieved context with source metadata - Keep concise for Gemma2:1b
                formatted_chunks = []
                for i, node in enumerate(retrieved_nodes, 1):
                    content = node.text[:800]  # Limit content size
                    
                    # Extract source file information from metadata
                    source_info = "Unknown source"
                    page_info = ""
                    
                    if hasattr(node, 'metadata') and node.metadata:
                        file_name = node.metadata.get('file_name', 'Unknown file')
                        source_info = f"Source: {file_name}"
                        
                        page_num = node.metadata.get('page_label', '')
                        if page_num:
                            page_info = f" (Page {page_num})"
                    
                    score = getattr(node, 'score', 0.0)
                    formatted_chunk = f"""DOCUMENT {i}:
{source_info}{page_info} | Score: {score:.2f}
Content: {content}

"""
                    formatted_chunks.append(formatted_chunk)
                
                # Limit total response size for smaller model
                context = "\n".join(formatted_chunks)[:4000]
                
                return context
                
            except Exception as e:
                logger.error("Document retrieval failed", error=str(e))
                return f"Error retrieving documents: {str(e)}. Please check your database connection and try again."
        
        return document_retrieval_tool

    def _create_fraud_detection_tool(self):
        """Create a fraud detection tool using the @tool decorator."""
        fraud_service = getattr(self.rag_service, 'fraud_detection_service', None)

        @tool("Fraud Detection Tool")
        def fraud_detection_tool(transaction_data: str) -> str:
            """Analyzes a credit card transaction for fraud. Pass transaction features as JSON with keys: V1-V28 (PCA features), Amount_scaled, Time_scaled.

            Args:
                transaction_data: JSON string with transaction features
            """
            import json
            try:
                if not fraud_service or not fraud_service.is_loaded:
                    return "Error: Fraud detection model is not available."

                data = json.loads(transaction_data)
                result = fraud_service.predict(data)

                if result["is_fraud"]:
                    return (
                        f"FRAUD DETECTED - This transaction is likely fraudulent. "
                        f"Fraud probability: {result['fraud_probability']:.1%}. "
                        f"Model version: {result['model_version']}."
                    )
                else:
                    return (
                        f"LEGITIMATE - This transaction appears legitimate. "
                        f"Fraud probability: {result['fraud_probability']:.1%}. "
                        f"Model version: {result['model_version']}."
                    )
            except json.JSONDecodeError:
                return "Error: Invalid JSON. Provide transaction features as valid JSON."
            except ValueError as e:
                return f"Error: {str(e)}"
            except Exception as e:
                logger.error("Fraud detection failed", error=str(e))
                return f"Error running fraud detection: {str(e)}"

        return fraud_detection_tool

    def _clean_response(self, response: str) -> str:
        """Clean the response to remove exposed thought processes and unwanted content."""
        import re

        # Remove common thought process patterns
        patterns_to_remove = [
            r"Thought:.*?(?=\n|$)",
            r"Action:.*?(?=\n|$)",
            r"Action Input:.*?(?=\n|$)",
            r"Observation:.*?(?=\n|$)",
            r"Final Answer:.*?(?=\n|$)",
            r"I have ensured.*?(?=\n|$)",
            r"meets the required format.*?(?=\n|$)",
        ]

        cleaned = response
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove excessive bold formatting - convert **text** to plain text
        # Keep only one level of bold for important headings
        cleaned = re.sub(r'\*\*(.*?)\*\*', r'\1', cleaned)

        # Remove any remaining asterisks used for emphasis
        cleaned = re.sub(r'\*([^*\n]+)\*', r'\1', cleaned)

        # Strip all Markdown headings (avoid bold rendering in OpenWebUI)
        cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)

        # Strip common list markers to keep plain text (no bullets)
        cleaned = re.sub(r'^\s*[-*]\s+', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)

        # Remove prompt-leak style lines and validator/planner artifacts
        cleaned = re.sub(r'^\s*Your\s+final\s+answer\s+must.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*Current\s*Task:.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*Review\s+the\s+draft\s+response.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*(Questions|Next\s*steps)\s*:.*$', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*I\s+(now\s+)?can\s+(give|provide).*$','', cleaned, flags=re.IGNORECASE | re.MULTILINE)
        cleaned = re.sub(r'^\s*(Here\s+is|Let\s+me)\b.*$','', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove embedded "Sources" section from the model output (we return sources separately)
        cleaned = re.sub(r'^\s*#?\s*Sources:?\s*$[\s\S]*', '', cleaned, flags=re.IGNORECASE | re.MULTILINE)

        # Remove multiple newlines and clean up
        cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
        cleaned = cleaned.strip()

        # If response is too short or generic, flag it
        generic_phrases = ['data protection requirements', 'privacy compliance measures', 'security protocols']
        if len(cleaned) < 100 or any(phrase in cleaned.lower() for phrase in generic_phrases):
            logger.warning("Response appears generic or too short", length=len(cleaned))

        return cleaned

    def _get_prompt_text(self, prompt_id: str) -> str:
        """Get active prompt template from PromptStore. Returns empty string if unavailable."""
        if self.prompt_store:
            template = self.prompt_store.get_active_template(prompt_id)
            if template:
                version = self.prompt_store.get_prompt(prompt_id).active_version
                print(f"[PromptStore] Using managed prompt: {prompt_id} (version {version})")
                logger.info("Using managed prompt", prompt_id=prompt_id, version=version)
                return template
            else:
                print(f"[PromptStore] No template found for: {prompt_id}, using fallback")
        else:
            print(f"[PromptStore] prompt_store not available, using fallback for: {prompt_id}")
        return ""

    def _initialize_agents(self):
        """Initialize CrewAI agents with prompts from PromptStore when available."""

        # Create retrieval tool using decorator approach
        retrieval_tool = self._create_document_retrieval_tool()

        # Load managed prompts (fall back to hardcoded if PromptStore unavailable)
        query_analysis_prompt = self._get_prompt_text("query_analysis")
        document_retrieval_prompt = self._get_prompt_text("document_retrieval")
        response_generation_prompt = self._get_prompt_text("response_generation")
        response_validation_prompt = self._get_prompt_text("response_validation")

        # --- QUERY ANALYSIS AGENT ---
        query_backstory = query_analysis_prompt if query_analysis_prompt else \
            """You are an expert conversation analyst who understands the full context of user interactions.

            Your key responsibilities:
            1. CONVERSATION ANALYSIS: Examine the full conversation history to understand context
            2. QUESTION TYPE DETECTION: Determine if current question is:
               - Follow-up (clarification, different format, more details about same topic)
               - New question (completely different topic/domain)
            3. SEARCH STRATEGY: Provide specific guidance for document retrieval:
               - For follow-ups: "FOLLOW_UP - search for [specific terms] related to [previous topic]"
               - For new questions: "NEW_TOPIC - search for [new topic terms]"

            Examples:
            - "What is procurement?" → "NEW_TOPIC - search for procurement definition, processes"
            - "Give me a 5-line summary" (after procurement question) → "FOLLOW_UP - search for procurement summary, key points"
            - "What are HR policies?" (after procurement question) → "NEW_TOPIC - search for HR policies, human resources"

            Always provide clear, specific search guidance for the Document Retriever.

            STRICT OUTPUT RULES:
            - Do NOT include suggested questions, next steps, or prompts for the user.
            - Do NOT include FOLLOW_UP/NEW_TOPIC markers in the final user answer (internal guidance only).
            - Keep output short and purely as internal guidance for retrieval."""

        self.query_agent = Agent(
            role="Query Analyzer",
            goal="Analyze conversations to understand question context and determine optimal search strategy",
            backstory=query_backstory,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=2,
            max_execution_time=600
        )

        # --- DOCUMENT RETRIEVAL AGENT ---
        retrieval_backstory = document_retrieval_prompt if document_retrieval_prompt else \
            """You are an expert document retrieval specialist who adapts search strategy based on conversation context.

            Your responsibilities:
            1. FOLLOW QUERY ANALYSER GUIDANCE: Use the search strategy provided by the Query Analyzer
            2. CONTEXT-AWARE SEARCH:
               - For FOLLOW_UP questions: Search using context from previous conversation
               - For NEW_TOPIC questions: Focus on the new topic independently
            3. SMART RETRIEVAL: Use the most relevant search terms to find specific information

            Always use the Document Retrieval Tool with appropriate search terms based on the guidance."""

        self.retrieval_agent = Agent(
            role="Document Retrieval Specialist",
            goal="Intelligently retrieve documents based on conversation context and query analysis guidance",
            backstory=retrieval_backstory,
            tools=[retrieval_tool],
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=3,
            max_execution_time=600
        )

        # --- RESPONSE GENERATION AGENT ---
        response_backstory = response_generation_prompt if response_generation_prompt else \
            """You are an expert at reading documents and extracting SPECIFIC information. You MUST:
            - Read the actual document content provided by the retrieval agent
            - Extract SPECIFIC details, facts, and information from those documents
            - Answer based ONLY on what is actually written in the documents
            - NEVER write generic answers or make assumptions
            - NEVER include your thought process or reasoning in the final answer

            FORMAT STRICTLY AS PLAIN TEXT ONLY:
            - Do NOT use Markdown at all (no headings, no bold, no italics)
            - Write in short paragraphs separated by blank lines

            IMPORTANT:
            - Do NOT ask follow-up questions or propose next steps in the answer.
            - Do NOT start with meta phrases like "I can", "I now can", "Here is", or "Let me".
            - Only answer the user's question."""

        self.response_agent = Agent(
            role="Information Extractor",
            goal="Extract specific information from retrieved documents to answer user questions",
            backstory=response_backstory,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=600
        )

        # --- RESPONSE VALIDATION AGENT ---
        validation_backstory = response_validation_prompt if response_validation_prompt else \
            """Ensure responses:
            - Answer only what the user asked
            - Contain NO follow-up questions, NO suggested next steps
            - Are in plain text (no Markdown, no headings, no bullets)
            - Are concise and specific"""

        self.validation_agent = Agent(
            role="Response Formatter",
            goal="Format responses properly and ensure they answer the user's question",
            backstory=validation_backstory,
            llm=self.llm,
            verbose=self.config.crew_verbose,
            allow_delegation=False,
            max_iter=1,
            max_execution_time=600
        )

        # --- FRAUD DETECTION AGENT ---
        fraud_service = getattr(self.rag_service, 'fraud_detection_service', None)
        if fraud_service and fraud_service.is_loaded:
            fraud_tool = self._create_fraud_detection_tool()
            self.fraud_agent = Agent(
                role="Fraud Detection Analyst",
                goal="Analyze credit card transactions to detect potential fraud using machine learning",
                backstory="""You are a fraud detection specialist. You analyze credit card transaction
                data using a trained machine learning model. When given transaction features, you use
                the Fraud Detection Tool to classify them as fraudulent or legitimate. You explain the
                results clearly, including the fraud probability score.""",
                tools=[fraud_tool],
                llm=self.llm,
                verbose=self.config.crew_verbose,
                allow_delegation=False,
                max_iter=2,
                max_execution_time=300
            )
            logger.info("Fraud detection agent initialized")
        else:
            self.fraud_agent = None
            logger.info("Fraud detection agent not available (model not loaded)")

    def create_crew(self, query: str) -> Crew:
        """Create a crew for processing a specific query."""
        
        # All queries go through the information retrieval workflow
        
        # Task 1: Intelligent conversation analysis and search strategy
        query_task = Task(
            description=f"""Analyze the following input and provide search strategy:

{query}

Your task:
1. If there's conversation history, analyze it to understand context
2. Determine if the current question is:
   - FOLLOW_UP: Clarification, different format, or more details about the same topic
   - NEW_TOPIC: Completely different subject or domain
3. Provide specific search guidance:
   - For FOLLOW_UP: "FOLLOW_UP - search for [topic] [specific request like summary/details]"
   - For NEW_TOPIC: "NEW_TOPIC - search for [new topic terms]"

Output format: "[TYPE] - search for [specific terms]"
Examples:
- "FOLLOW_UP - search for procurement key points summary"
- "NEW_TOPIC - search for HR policies human resources"
""",
            agent=self.query_agent,
            expected_output="Search strategy with question type and specific search terms"
        )
        
        # Task 2: Context-aware document retrieval
        retrieval_task = Task(
            description=f"""Based on the Query Analyzer's guidance, retrieve relevant documents.

Original query: "{query}"

Instructions:
1. Use the search strategy provided by the Query Analyzer
2. Extract the search terms from their guidance
3. Use the Document Retrieval Tool with appropriate search terms
4. For FOLLOW_UP questions: Consider previous context in search
5. For NEW_TOPIC questions: Focus on the new topic

Use the tool to retrieve the most relevant documents.""",
            agent=self.retrieval_agent,
            expected_output="Retrieved document content relevant to the search strategy",
            context=[query_task]
        )
        
        # Task 3: Context-aware response generation
        response_task = Task(
            description=f"""Generate a response using the Query Analyzer's guidance and retrieved documents.

Original query: "{query}"

Instructions:
1. Check the Query Analyzer's determination (FOLLOW_UP vs NEW_TOPIC)
2. For FOLLOW_UP questions:
   - Reference previous conversation context appropriately
   - Provide the specific format/details requested (e.g., "5-line summary")
   - Use information from retrieved documents about the same topic
3. For NEW_TOPIC questions:
   - Provide comprehensive answer about the new topic
   - Focus on the new subject independently

STRICT REQUIREMENTS:
1. Read the ACTUAL document content from the retrieval task
2. Extract SPECIFIC information from those documents
3. Answer based ONLY on what is written in the documents
4. If documents don't contain the information, say so explicitly
5. Use format: # title, then plain text with - bullet points
6. DO NOT include "Thought:" or reasoning in your answer
7. DO NOT write generic responses
8. DO NOT use asterisks (*) or double asterisks (**) for formatting
9. DO NOT make text bold - use plain text only
10. Write in simple markdown with # for headings and - for lists

FORMATTING RULES:
- Use ONLY ONE # heading at the very top
- Use plain text sections (not additional # headings)
- Use - for bullet points
- NO bold formatting with ** or *
- NO italic formatting
- NO multiple headings - only one # at the start

Answer with SPECIFIC details from the actual documents using plain text formatting.""",
            agent=self.response_agent,
            expected_output="Context-aware answer based on actual document content, following conversation guidance",
            context=[query_task, retrieval_task]
        )

        # Task 4: Validation and formatting (ensures no follow-up questions / next steps)
        validation_task = Task(
            description="""Review the draft response and ensure it:
1) Answers only the user's question
2) Contains NO follow-up questions, NO suggested next steps
3) Is plain text (no Markdown, no headings, no bullets)
4) Is concise and specific

Return only the final cleaned answer text.""",
            agent=self.validation_agent,
            expected_output="Final cleaned answer without follow-up prompts or extra questions",
            context=[response_task]
        )

        # Create and return crew with timeout
        crew = Crew(
            agents=[self.query_agent, self.retrieval_agent, self.response_agent, self.validation_agent],
            tasks=[query_task, retrieval_task, response_task, validation_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            max_execution_time=600
        )
        
        return crew
    
    def create_fraud_crew(self, query: str) -> Crew:
        """Create a crew for processing fraud detection queries."""
        if not self.fraud_agent:
            raise RuntimeError("Fraud detection agent not available")

        fraud_task = Task(
            description=f"""Analyze the following fraud detection request:

{query}

Instructions:
1. Extract the transaction data (JSON with V1-V28, Amount_scaled, Time_scaled)
2. Use the Fraud Detection Tool to classify the transaction
3. Report whether the transaction is fraudulent or legitimate
4. Include the fraud probability score in your response""",
            agent=self.fraud_agent,
            expected_output="Fraud detection result with probability score"
        )

        return Crew(
            agents=[self.fraud_agent],
            tasks=[fraud_task],
            process=Process.sequential,
            verbose=False,
            memory=False,
            max_execution_time=300
        )

    async def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query using the multi-agent CrewAI system.
        
        This method orchestrates the complete workflow:
        1. Creates a specialized crew for the query
        2. Executes the multi-agent workflow
        3. Extracts and cleans the final response
        4. Retrieves relevant source documents
        5. Returns structured response with metadata
        
        Args:
            query (str): User's question or query
            
        Returns:
            Dict[str, Any]: Structured response containing:
                - response: Generated answer
                - sources: List of source documents
                - metadata: Processing information
                
        Raises:
            Exception: If processing fails
        """
        try:
            logger.info("Starting CrewAI processing", query=query[:100])

            # Log prompt versions being used (for Phoenix tracing)
            prompt_versions = {}
            if self.prompt_store:
                for pid in ["query_analysis", "document_retrieval", "response_generation", "response_validation"]:
                    entry = self.prompt_store.get_prompt(pid)
                    if entry:
                        prompt_versions[pid] = entry.active_version
                logger.info("CrewAI using managed prompts", versions=prompt_versions)

            # Trace prompt usage to Phoenix
            if self.phoenix_service and prompt_versions:
                import asyncio
                for pid, version in prompt_versions.items():
                    try:
                        asyncio.get_event_loop().run_until_complete(
                            self.phoenix_service.log_prompt_execution(
                                prompt_id=pid,
                                variables={"query": query[:200]},
                                response="[CrewAI agent initialized]",
                                metadata={"version": str(version), "stage": "agent_init"}
                            )
                        )
                    except RuntimeError:
                        # Event loop already running, skip async tracing
                        pass

            # Create crew for query processing
            crew = self.create_crew(query)

            # Execute the crew workflow
            result = crew.kickoff()

            # Extract the final response from the last task
            final_response = str(result)

            # Clean up the response - remove any exposed thought processes and bold formatting
            final_response = self._clean_response(final_response)
            
            # Get sources for information queries
            sources = []
            try:
                # Get sources using the same retrieval logic
                retriever = self.rag_service.index.as_retriever(
                    similarity_top_k=self.config.similarity_top_k
                )
                nodes = retriever.retrieve(query)
                
                for node in nodes:
                    source_info = {
                        "content": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        "score": float(getattr(node, 'score', 0.0)),
                        "metadata": node.metadata if hasattr(node, 'metadata') else {}
                    }
                    sources.append(source_info)
            except Exception as e:
                logger.warning("Could not retrieve sources", error=str(e))
            
            # Determine query type (removed greeting detection)
            query_type = "information"
            
            logger.info("CrewAI processing completed", 
                       response_length=len(final_response),
                       source_count=len(sources),
                       query_type=query_type)
            
            return {
                "response": final_response,
                "sources": sources,
                "metadata": {
                    "model": "crewai-agentic-rag-llama3.2:1b",
                    "agents_used": ["retrieval_specialist", "response_generator", "validator"],
                    "process_type": "sequential",
                    "query_type": query_type,
                    "source_count": len(sources)
                }
            }
            
        except Exception as e:
            logger.error("CrewAI processing failed", error=str(e))
            
            # Return a helpful error response
            error_response = "I apologize, but I encountered an error while processing your query. "
            if "connection" in str(e).lower():
                error_response += "It appears there may be a database connection issue. Please check your database connection and try again."
            elif "model" in str(e).lower() or "ollama" in str(e).lower():
                error_response += "There seems to be an issue with the language model. Please ensure Ollama is running and the llama3.2:1b model is available."
            else:
                error_response += f"Error details: {str(e)}. Please try rephrasing your question or contact support if the issue persists."
            
            return {
                "response": error_response,
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "model": "crewai-agentic-rag-llama3.2:1b",
                    "process_type": "error_handling",
                    "query_type": "error"
                }
            }


# Testing functions
def test_retrieval_tool_only(config: BackendConfig, query: str = "data types"):
    """Test only the document retrieval tool without CrewAI."""
    print(f"\n{'='*60}")
    print("TESTING DOCUMENT RETRIEVAL TOOL ONLY")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        import time
        
        # Parse database URL
        DATABASE_URL = config.database_url
        db_url_parts = urlparse(DATABASE_URL)
        
        print(f"Connecting to database: {db_url_parts.hostname}:{db_url_parts.port}")
        
        # Initialize vector store
        vector_store = PGVectorStore.from_params(
            host=db_url_parts.hostname,
            port=db_url_parts.port,
            database=db_url_parts.path.lstrip('/'),
            user=db_url_parts.username,
            password=db_url_parts.password,
            table_name="llamaindex_vectors_copy",
            embed_dim=768,
        )
        
        print(" Vector store connection successful")
        
        # Initialize embedding model
        embed_model = OllamaEmbedding(
            model_name=config.ollama_embedding_model,
            base_url=config.ollama_base_url,
        )
        
        print(" Embedding model initialized")
        
        # Create index
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        
        print(" Index created")
        
        # Test retrieval
        retriever = index.as_retriever(
            similarity_top_k=config.similarity_top_k,
            verbose=True
        )
        
        print(f" Retrieving documents for: '{query}'")
        start_time = time.time()
        
        nodes = retriever.retrieve(query)
        
        end_time = time.time()
        print(f" Retrieval took: {end_time - start_time:.2f} seconds")
        print(f" Found {len(nodes)} documents")
        
        if nodes:
            print("\n RETRIEVED DOCUMENTS:")
            for i, node in enumerate(nodes, 1):
                print(f"\n--- Document {i} ---")
                print(f"Score: {getattr(node, 'score', 0.0):.3f}")
                if hasattr(node, 'metadata') and node.metadata:
                    print(f"Source: {node.metadata.get('file_name', 'Unknown')}")
                    page = node.metadata.get('page_label', '')
                    if page:
                        print(f"Page: {page}")
                print(f"Content: {node.text[:300]}...")
                print("-" * 40)
        else:
            print(" No documents retrieved")
            
        return True
        
    except Exception as e:
        print(f" Error in retrieval test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_ollama_llm_only(config: BackendConfig):
    """Test only the Ollama LLM connection without retrieval."""
    print(f"\n{'='*60}")
    print("TESTING OLLAMA LLM CONNECTION ONLY")
    print(f"{'='*60}")
    
    try:
        import time
        
        # Initialize LLM
        llm = LLM(
            model="ollama/llama3.2:1b",
            api_base=config.ollama_base_url,
            temperature=config.temperature,
            max_tokens=512,
            timeout=30,
            max_retries=2
        )

        print(f" Testing LLM: llama3.2:1b")
        print(f"🔗 Ollama URL: {config.ollama_base_url}")
        
        # Test simple completion
        test_prompt = "What are data types? Answer in 2 sentences."
        
        print(f"💬 Test prompt: '{test_prompt}'")
        print("⏳ Calling LLM...")
        
        start_time = time.time()
        
        # Use CrewAI's LLM call method
        response = llm.call([{"role": "user", "content": test_prompt}])
        
        end_time = time.time()
        
        print(f" LLM call took: {end_time - start_time:.2f} seconds")
        print(f" LLM Response: {response}")
        
        return True
        
    except Exception as e:
        print(f" Error in LLM test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_rag_crew(config: BackendConfig, query: str = "What data types are mentioned?"):
    """Test the full RAG CrewAI system."""
    print(f"\n{'='*60}")
    print("TESTING FULL RAG CREWAI SYSTEM")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print("-" * 40)
    
    try:
        import time
        
        # Mock RAG service for testing
        class MockRAGService:
            def __init__(self, config):
                self.config = config
                # Mock index for sources retrieval
                try:
                    DATABASE_URL = config.database_url
                    db_url_parts = urlparse(DATABASE_URL)
                    
                    vector_store = PGVectorStore.from_params(
                        host=db_url_parts.hostname,
                        port=db_url_parts.port,
                        database=db_url_parts.path.lstrip('/'),
                        user=db_url_parts.username,
                        password=db_url_parts.password,
                        table_name="llamaindex_vectors_copy",
                        embed_dim=768,
                    )
                    
                    embed_model = OllamaEmbedding(
                        model_name=config.ollama_embedding_model,
                        base_url=config.ollama_base_url,
                    )
                    
                    self.index = VectorStoreIndex.from_vector_store(
                        vector_store=vector_store,
                        embed_model=embed_model
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize index for sources: {e}")
                    self.index = None
        
        rag_service = MockRAGService(config)
        rag_crew = RAGCrew(rag_service, config)
        
        print(" RAG Crew initialized")
        
        start_time = time.time()
        
        result = await rag_crew.process_query(query)
        
        end_time = time.time()
        
        print(f" Full processing took: {end_time - start_time:.2f} seconds")
        print(f"📝 Response: {result['response']}")
        print(f" Metadata: {result['metadata']}")
        print(f"🔗 Sources: {len(result['sources'])} found")
        
        return True
        
    except Exception as e:
        print(f" Error in full RAG test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests in sequence."""
    print("🧪 STARTING RAG CREWAI COMPREHENSIVE TESTS")
    print("=" * 80)
    
    # Initialize config - UPDATE THESE WITH YOUR ACTUAL VALUES
    config = BackendConfig()
    
    print(f"Configuration:")
    print(f"  Database: {config.database_url}")
    print(f"  Ollama URL: {config.ollama_base_url}")
    print(f"  LLM Model: llama3.2:1b")
    print(f"  Embedding Model: {config.ollama_embedding_model}")
    
    # Test 1: Document Retrieval Tool Only
    retrieval_success = test_retrieval_tool_only(config, "What is the primary purpose of the “Negotiation Plan” document?")
    
    # Test 2: LLM Only
    llm_success = test_ollama_llm_only(config)
    
    # Test 3: Full RAG Crew (only if previous tests pass)
    if retrieval_success and llm_success:
        print("\n Basic tests passed, testing full RAG Crew...")
        try:
            import asyncio
            asyncio.run(test_full_rag_crew(config, "What is the primary purpose of the “Negotiation Plan” document?"))
        except Exception as e:
            print(f" Full RAG Crew test failed: {e}")
    else:
        print("\n Skipping full RAG test due to basic test failures")
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f" Document Retrieval: {' PASS' if retrieval_success else ' FAIL'}")
    print(f" LLM Connection: {' PASS' if llm_success else ' FAIL'}")
    print(f" Integration Ready: {' YES' if retrieval_success and llm_success else ' NO'}")


if __name__ == "__main__":
    # Configure logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run comprehensive tests
    run_all_tests()