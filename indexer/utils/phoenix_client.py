"""Phoenix client for observability and tracing."""

from typing import Optional
import structlog
try:
    import phoenix as px
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    logger = structlog.get_logger()
    logger.warning("Phoenix not available, tracing will be disabled")

from config import IndexerConfig


logger = structlog.get_logger()


class PhoenixClient:
    """Phoenix observability client."""
    
    def __init__(self, config: IndexerConfig):
        self.config = config
        self.tracer = None
        self.instrumentor = None
        self.enabled = PHOENIX_AVAILABLE
    
    async def initialize(self):
        """Initialize Phoenix tracing."""
        if not self.enabled:
            logger.info("Phoenix not available, skipping tracing setup")
            return
            
        try:
            # Set up basic OpenTelemetry tracing without Phoenix session
            trace.set_tracer_provider(TracerProvider())
            self.tracer = trace.get_tracer(__name__)
            
            logger.info("Basic tracing initialized (Phoenix optional)")
            
        except Exception as e:
            logger.warning("Phoenix initialization failed, continuing without tracing", error=str(e))
            self.enabled = False
    
    def create_span(self, name: str, **attributes):
        """Create a new tracing span."""
        if self.tracer and self.enabled:
            return self.tracer.start_span(name, attributes=attributes)
        return None
    
    async def cleanup(self):
        """Cleanup Phoenix resources."""
        try:
            if self.instrumentor:
                self.instrumentor.uninstrument()
            logger.info("Phoenix cleanup completed")
        except Exception as e:
            logger.error("Error during Phoenix cleanup", error=str(e))