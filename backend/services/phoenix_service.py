"""Phoenix integration for prompt lifecycle management and observability."""

import os
from typing import Dict, Any, Optional, List
import structlog
import httpx
from pydantic import BaseModel

# OpenTelemetry imports for Phoenix tracing
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource

# Phoenix specific imports
try:
    import phoenix as px
    from phoenix.trace import using_project
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    px = None
    using_project = None

logger = structlog.get_logger()


class PromptTemplate(BaseModel):
    """Prompt template model."""
    id: str
    name: str
    template: str
    variables: List[str]
    description: Optional[str] = None
    tags: List[str] = []


class PhoenixService:
    """Service for observability via Phoenix and prompt management via PromptStore."""

    def __init__(self, config, prompt_store=None):
        self.config = config
        self.base_url = config.phoenix_base_url
        self.project_name = config.phoenix_project_name
        self.prompt_store = prompt_store
        self.client = None
        self.tracer = None
        self.tracing_enabled = False

    async def initialize(self):
        """Initialize Phoenix service and tracing."""
        try:
            # Create HTTP client
            self.client = httpx.AsyncClient(timeout=30.0)

            # Test Phoenix connection
            await self._test_connection()

            # Initialize Phoenix tracing
            await self._setup_phoenix_tracing()

            # Sync prompts to Phoenix if prompt_store is available
            if self.prompt_store:
                await self._sync_prompts_to_phoenix()

            logger.info("Phoenix service initialized successfully", tracing_enabled=self.tracing_enabled)

        except Exception as e:
            logger.warning("Phoenix initialization failed, tracing disabled", error=str(e))

    async def _setup_phoenix_tracing(self):
        """Setup OpenTelemetry tracing to Phoenix."""
        try:
            if not PHOENIX_AVAILABLE:
                logger.warning("Phoenix package not available, tracing disabled")
                return

            # Parse Phoenix URL
            from urllib.parse import urlparse
            parsed_url = urlparse(self.base_url)
            phoenix_host = parsed_url.hostname or "localhost"
            phoenix_port = parsed_url.port or 6006

            # Set environment variables
            os.environ["PHOENIX_HOST"] = phoenix_host
            os.environ["PHOENIX_PORT"] = str(phoenix_port)

            # Configure OpenTelemetry to send traces to Phoenix
            resource = Resource(attributes={
                SERVICE_NAME: self.project_name
            })

            # Create OTLP exporter pointing to Phoenix
            otlp_endpoint = f"http://{phoenix_host}:{phoenix_port}/v1/traces"

            exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            span_processor = BatchSpanProcessor(exporter)

            # Set up tracer provider
            tracer_provider = trace_sdk.TracerProvider(resource=resource)
            tracer_provider.add_span_processor(span_processor)

            # Set the global tracer provider
            trace.set_tracer_provider(tracer_provider)

            # Get tracer for this service
            self.tracer = trace.get_tracer(__name__)
            self.tracing_enabled = True

            logger.info("Phoenix tracing configured successfully",
                       endpoint=otlp_endpoint,
                       project=self.project_name)

        except Exception as e:
            logger.warning("Failed to setup Phoenix tracing", error=str(e))
            self.tracing_enabled = False

    async def _test_connection(self):
        """Test connection to Phoenix."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                logger.info("Phoenix connection successful")
            else:
                raise Exception(f"Phoenix health check failed: {response.status_code}")
        except Exception as e:
            logger.warning("Phoenix connection test failed", error=str(e))
            raise

    async def _sync_prompts_to_phoenix(self):
        """Sync prompts from PromptStore to Phoenix for visibility."""
        if not self.prompt_store:
            return

        prompts = self.prompt_store.list_prompts()
        for prompt_id, prompt_data in prompts.items():
            try:
                # Register prompt in Phoenix via API (if Phoenix supports it)
                if self.client and self.tracing_enabled:
                    # Log prompt registration as a span
                    if self.tracer:
                        with self.tracer.start_as_current_span("prompt_registered") as span:
                            span.set_attribute("prompt.id", prompt_id)
                            span.set_attribute("prompt.name", prompt_data["name"])
                            span.set_attribute("prompt.version", prompt_data.get("active_version", 1))
                            span.set_attribute("prompt.variables", ",".join(prompt_data.get("variables", [])))
                            span.add_event("prompt_synced_to_phoenix", {
                                "prompt_id": prompt_id,
                                "name": prompt_data["name"],
                            })

                logger.info("Prompt synced to Phoenix", prompt_id=prompt_id)
            except Exception as e:
                logger.warning(f"Failed to sync prompt {prompt_id} to Phoenix", error=str(e))

    # --- Prompt Access (delegates to PromptStore) ---

    async def get_prompt(self, prompt_id: str) -> Optional[PromptTemplate]:
        """Retrieve a prompt template from PromptStore."""
        if not self.prompt_store:
            logger.warning("No prompt store available", prompt_id=prompt_id)
            return None

        entry = self.prompt_store.get_prompt(prompt_id)
        if not entry:
            logger.warning("Prompt not found", prompt_id=prompt_id)
            return None

        template = self.prompt_store.get_active_template(prompt_id)
        variables = self.prompt_store.get_active_variables(prompt_id)

        return PromptTemplate(
            id=prompt_id,
            name=entry.name,
            template=template or "",
            variables=variables or [],
            description=entry.description,
            tags=entry.tags,
        )

    async def render_prompt(self, prompt_id: str, variables: Dict[str, str]) -> Optional[str]:
        """Render a prompt template with variables using PromptStore."""
        if not self.prompt_store:
            logger.warning("No prompt store available for rendering", prompt_id=prompt_id)
            return None

        rendered = self.prompt_store.render(prompt_id, variables)
        if rendered:
            logger.info("Prompt rendered", prompt_id=prompt_id, variables=list(variables.keys()))
        return rendered

    async def get_available_prompts(self) -> List[PromptTemplate]:
        """Get list of all available prompts."""
        if not self.prompt_store:
            return []

        prompts = self.prompt_store.list_prompts()
        result = []
        for prompt_id, data in prompts.items():
            result.append(PromptTemplate(
                id=prompt_id,
                name=data["name"],
                template=data.get("template", ""),
                variables=data.get("variables", []),
                description=data.get("description"),
                tags=data.get("tags", []),
            ))
        return result

    # --- Observability Logging ---

    async def log_prompt_execution(self, prompt_id: str, variables: Dict[str, str],
                                 response: str, metadata: Dict[str, Any] = None):
        """Log prompt execution for observability with version tracking."""
        try:
            # Get version info from prompt store
            version = None
            if self.prompt_store:
                entry = self.prompt_store.get_prompt(prompt_id)
                if entry:
                    version = entry.active_version

            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span(f"prompt_execution_{prompt_id}") as span:
                    span.set_attribute("prompt.id", prompt_id)
                    span.set_attribute("prompt.variables_count", len(variables))
                    span.set_attribute("response.length", len(response))

                    if version is not None:
                        span.set_attribute("prompt.version", version)

                    for key, value in variables.items():
                        span.set_attribute(f"prompt.variable.{key}", str(value)[:100])

                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"metadata.{key}", str(value))

                    span.add_event("prompt_executed", {
                        "prompt_id": prompt_id,
                        "version": str(version) if version else "unknown",
                        "variables": str(variables),
                        "response_preview": response[:200] if response else ""
                    })

                    logger.info("Prompt execution traced to Phoenix",
                               prompt_id=prompt_id,
                               version=version,
                               span_id=span.get_span_context().span_id)
            else:
                log_data = {
                    "prompt_id": prompt_id,
                    "version": version,
                    "variables": variables,
                    "response_length": len(response),
                    "metadata": metadata or {}
                }
                logger.info("Prompt execution logged (no tracing)", **log_data)

        except Exception as e:
            logger.error("Failed to log prompt execution", error=str(e))

    async def log_chat_interaction(self, conversation_id: str, user_message: str,
                                 assistant_response: str, sources: List[Dict] = None,
                                 metadata: Dict[str, Any] = None):
        """Log complete chat interaction for observability."""
        try:
            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span("chat_interaction") as span:
                    span.set_attribute("chat.conversation_id", conversation_id)
                    span.set_attribute("chat.user_message", user_message[:500])
                    span.set_attribute("chat.assistant_response", assistant_response[:500])
                    span.set_attribute("chat.user_message_length", len(user_message))
                    span.set_attribute("chat.assistant_response_length", len(assistant_response))

                    if sources:
                        span.set_attribute("chat.sources_count", len(sources))
                        for i, source in enumerate(sources[:3]):
                            if isinstance(source, dict):
                                span.set_attribute(f"chat.source_{i}.score", source.get("score", 0.0))
                                if source.get("metadata"):
                                    span.set_attribute(f"chat.source_{i}.file",
                                                     source["metadata"].get("file_name", "unknown"))

                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"chat.metadata.{key}", str(value))

                    span.add_event("chat_completed", {
                        "conversation_id": conversation_id,
                        "user_message": user_message,
                        "assistant_response": assistant_response,
                        "sources": str(sources) if sources else "[]",
                        "metadata": str(metadata) if metadata else "{}"
                    })

                    logger.info("Chat interaction traced to Phoenix",
                               conversation_id=conversation_id,
                               message_length=len(user_message),
                               response_length=len(assistant_response),
                               sources_count=len(sources) if sources else 0)
            else:
                log_data = {
                    "chat_interaction": True,
                    "conversation_id": conversation_id,
                    "user_message": user_message,
                    "assistant_response": assistant_response,
                    "user_message_length": len(user_message),
                    "assistant_response_length": len(assistant_response),
                    "sources": sources,
                    "sources_count": len(sources) if sources else 0,
                    "metadata": metadata or {}
                }
                logger.info("Chat interaction logged", **log_data)

        except Exception as e:
            logger.error("Failed to log chat interaction", error=str(e))

    async def log_agent_workflow(self, workflow_type: str, query: str, agents_used: List[str],
                               execution_time: float, result: Dict[str, Any],
                               metadata: Dict[str, Any] = None):
        """Log agent workflow execution for observability."""
        try:
            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span(f"agent_workflow_{workflow_type}") as span:
                    span.set_attribute("workflow.type", workflow_type)
                    span.set_attribute("workflow.query", query[:500])
                    span.set_attribute("workflow.agents_used", ",".join(agents_used))
                    span.set_attribute("workflow.execution_time", execution_time)
                    span.set_attribute("workflow.response_length", len(result.get("response", "")))

                    if result.get("sources"):
                        span.set_attribute("workflow.sources_count", len(result["sources"]))

                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"workflow.metadata.{key}", str(value))

                    span.add_event("agent_workflow_completed", {
                        "workflow_type": workflow_type,
                        "query": query,
                        "agents_used": agents_used,
                        "execution_time": execution_time,
                        "result": str(result)[:1000],
                        "metadata": str(metadata) if metadata else "{}"
                    })

                    logger.info("Agent workflow traced to Phoenix",
                               workflow_type=workflow_type,
                               execution_time=execution_time,
                               agents_count=len(agents_used))
            else:
                log_data = {
                    "agent_workflow": True,
                    "workflow_type": workflow_type,
                    "query": query,
                    "agents_used": agents_used,
                    "execution_time": execution_time,
                    "result": result,
                    "metadata": metadata or {}
                }
                logger.info("Agent workflow logged", **log_data)

        except Exception as e:
            logger.error("Failed to log agent workflow", error=str(e))

    async def log_document_retrieval(self, query: str, retrieved_docs: List[Dict],
                                   retrieval_time: float, metadata: Dict[str, Any] = None):
        """Log document retrieval for observability."""
        try:
            if self.tracing_enabled and self.tracer:
                with self.tracer.start_as_current_span("document_retrieval") as span:
                    span.set_attribute("retrieval.query", query[:500])
                    span.set_attribute("retrieval.docs_count", len(retrieved_docs))
                    span.set_attribute("retrieval.time", retrieval_time)

                    for i, doc in enumerate(retrieved_docs[:5]):
                        if isinstance(doc, dict):
                            span.set_attribute(f"retrieval.doc_{i}.score", doc.get("score", 0.0))
                            if doc.get("metadata"):
                                span.set_attribute(f"retrieval.doc_{i}.file",
                                                 doc["metadata"].get("file_name", "unknown"))

                    if metadata:
                        for key, value in metadata.items():
                            span.set_attribute(f"retrieval.metadata.{key}", str(value))

                    span.add_event("documents_retrieved", {
                        "query": query,
                        "docs_count": len(retrieved_docs),
                        "retrieval_time": retrieval_time,
                        "documents": str(retrieved_docs)[:1000],
                        "metadata": str(metadata) if metadata else "{}"
                    })

                    logger.info("Document retrieval traced to Phoenix",
                               query_length=len(query),
                               docs_count=len(retrieved_docs),
                               retrieval_time=retrieval_time)
            else:
                log_data = {
                    "document_retrieval": True,
                    "query": query,
                    "retrieved_docs": retrieved_docs,
                    "docs_count": len(retrieved_docs),
                    "retrieval_time": retrieval_time,
                    "metadata": metadata or {}
                }
                logger.info("Document retrieval logged", **log_data)

        except Exception as e:
            logger.error("Failed to log document retrieval", error=str(e))

    def create_trace_span(self, operation_name: str, attributes: Dict[str, Any] = None):
        """Create a new trace span for manual tracing."""
        if not self.tracing_enabled or not self.tracer:
            return None

        span = self.tracer.start_span(operation_name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))

        return span

    async def cleanup(self):
        """Cleanup Phoenix service."""
        try:
            if self.client:
                await self.client.aclose()
            logger.info("Phoenix service cleanup completed")
        except Exception as e:
            logger.error("Error during Phoenix cleanup", error=str(e))
