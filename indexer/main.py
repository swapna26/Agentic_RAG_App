"""Main document indexer application."""

import asyncio
import logging
import os
import ssl
import sys
import urllib3
from pathlib import Path
from typing import List, Optional

import click
import structlog
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler

# Disable SSL verification globally
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['SSL_VERIFY'] = 'false'
os.environ['PYTHONHTTPSVERIFY'] = '0'

from config import config
from docling_processor import DoclingDocumentProcessor
from utils.database_utils import DatabaseManager
from utils.phoenix_client import PhoenixClient


class StandaloneConfig:
    """Standalone configuration for indexer without external dependencies."""
    chunk_size = 768
    chunk_overlap = 150
    documents_path = str(Path(__file__).parent.parent / "documents")
    database_url = "postgresql://raguser:ragpassword@localhost:5432/agentic_rag"
    batch_size = 5
    log_level = "INFO"


# Configure structured logging
try:
    log_level = getattr(logging, config.log_level.upper())
except AttributeError:
    log_level = logging.INFO

logging.basicConfig(
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
    level=log_level,
)

logger = structlog.get_logger()
console = Console()


class DocumentIndexer:
    """Main document indexer class."""

    def __init__(self):
        self.db_manager = DatabaseManager(config.database_url)
        self.doc_processor = DoclingDocumentProcessor(config)
        self.phoenix_client = PhoenixClient(config)


class StandaloneDocumentIndexer:
    """Standalone document indexer class without external dependencies."""

    def __init__(self, standalone_config=None):
        self.config = standalone_config or StandaloneConfig()
        self.doc_processor = DoclingDocumentProcessor(self.config)

        # Set up markdown directory
        self.markdown_dir = Path(__file__).parent.parent / "markdown_output"
        self.markdown_dir.mkdir(exist_ok=True)
        self.doc_processor.markdown_dir = self.markdown_dir

    async def initialize(self) -> bool:
        """Initialize the indexer components (standalone mode - only DoclingProcessor)."""
        try:
            # Only initialize document processor (no database manager or Phoenix client)
            await self.doc_processor.initialize()
            logger.info("Standalone document indexer initialized successfully with Contextual RAG")
            return True

        except Exception as e:
            logger.error("Failed to initialize standalone indexer", error=str(e))
            return False

    async def index_documents(self, documents_path: Optional[str] = None) -> int:
        """Index documents from the specified path (standalone mode)."""
        path = Path(documents_path or self.config.documents_path)

        if not path.exists():
            logger.error("Documents path does not exist", path=str(path))
            return 0

        # Find all supported documents
        supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.html'}
        documents = [
            f for f in path.rglob('*')
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not documents:
            logger.warning("No supported documents found", path=str(path))
            return 0

        logger.info("Found documents to index", count=len(documents))
        for doc in documents:
            size_mb = doc.stat().st_size / (1024*1024)
            console.print(f"  - {doc.name} ({size_mb:.1f} MB)")

        processed_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents with Contextual RAG...", total=len(documents))

            # Process documents in batches
            for i in range(0, len(documents), self.config.batch_size):
                batch = documents[i:i + self.config.batch_size]

                try:
                    batch_results = await self.doc_processor.process_batch(batch)
                    successful = len([r for r in batch_results if r])
                    processed_count += successful

                    progress.update(task, advance=len(batch))
                    console.print(f"Batch {i//self.config.batch_size + 1}: {successful}/{len(batch)} documents processed successfully", style="cyan")

                except Exception as e:
                    logger.error("Failed to process batch", error=str(e))
                    progress.update(task, advance=len(batch))
                    continue

        logger.info("Standalone document indexing completed",
                   total=len(documents),
                   processed=processed_count)

        # Show generated markdown files
        md_files = list(self.markdown_dir.glob("*.md"))
        console.print(f"Markdown files generated: {len(md_files)}", style="green")
        for md_file in md_files:
            size_kb = md_file.stat().st_size / 1024
            console.print(f"  - {md_file.name} ({size_kb:.1f} KB)", style="dim")

        return processed_count

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.doc_processor.cleanup()
            logger.info("Standalone document indexer cleanup completed")
        except Exception as e:
            logger.error("Error during standalone cleanup", error=str(e))


class DocumentIndexer:
    """Main document indexer class."""

    def __init__(self):
        self.db_manager = DatabaseManager(config.database_url)
        self.doc_processor = DoclingDocumentProcessor(config)
        self.phoenix_client = PhoenixClient(config)

    async def initialize(self) -> bool:
        """Initialize the indexer components."""
        try:
            # Initialize Phoenix tracing
            await self.phoenix_client.initialize()

            # Initialize database
            await self.db_manager.initialize()

            # Initialize document processor
            await self.doc_processor.initialize()

            logger.info("Document indexer initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize indexer", error=str(e))
            return False

    async def index_documents(self, documents_path: Optional[str] = None) -> int:
        """Index documents from the specified path."""
        path = Path(documents_path or config.documents_path)

        if not path.exists():
            logger.error("Documents path does not exist", path=str(path))
            return 0

        # Find all supported documents
        supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.html'}
        documents = [
            f for f in path.rglob('*')
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]

        if not documents:
            logger.warning("No supported documents found", path=str(path))
            return 0

        logger.info("Found documents to index", count=len(documents))

        processed_count = 0
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing documents...", total=len(documents))

            # Process documents in batches
            for i in range(0, len(documents), config.batch_size):
                batch = documents[i:i + config.batch_size]

                try:
                    batch_results = await self.doc_processor.process_batch(batch)
                    processed_count += len([r for r in batch_results if r])

                    progress.update(task, advance=len(batch))

                except Exception as e:
                    logger.error("Failed to process batch", error=str(e))
                    continue

        logger.info("Document indexing completed",
                   total=len(documents),
                   processed=processed_count)

        return processed_count

    async def cleanup(self):
        """Cleanup resources."""
        try:
            await self.doc_processor.cleanup()
            await self.db_manager.cleanup()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))


@click.group()
def cli():
    """Document Indexer for Agentic RAG System."""
    pass


@cli.command()
@click.option('--path', '-p', help='Path to documents directory')
@click.option('--force', '-f', is_flag=True, help='Force re-indexing of existing documents')
def index(path: Optional[str], force: bool):
    """Index documents from the specified directory."""
    async def run_indexing():
        indexer = DocumentIndexer()
        
        try:
            # Initialize
            if not await indexer.initialize():
                console.print(" Failed to initialize indexer", style="red")
                sys.exit(1)
            
            # Set force re-indexing if specified
            if force:
                indexer.doc_processor.force_reindex = True
            
            # Index documents
            count = await indexer.index_documents(path)
            
            if count > 0:
                console.print(f"Successfully indexed {count} documents", style="green")
            else:
                console.print("No documents were indexed", style="yellow")
                
        except KeyboardInterrupt:
            console.print(" Indexing interrupted by user", style="red")
        except Exception as e:
            logger.error("Indexing failed", error=str(e))
            console.print(f" Indexing failed: {e}", style="red")
            sys.exit(1)
        finally:
            await indexer.cleanup()
    
    asyncio.run(run_indexing())


@cli.command()
def status():
    """Check the status of the indexing system."""
    async def check_status():
        indexer = DocumentIndexer()
        
        try:
            success = await indexer.initialize()
            if success:
                console.print(" Indexer system is healthy", style="green")
                
                # Get document count from database
                count = await indexer.db_manager.get_document_count()
                console.print(f" Documents in index: {count}")
                
            else:
                console.print(" Indexer system is not healthy", style="red")
                sys.exit(1)
                
        except Exception as e:
            console.print(f" Status check failed: {e}", style="red")
            sys.exit(1)
        finally:
            await indexer.cleanup()
    
    asyncio.run(check_status())


@cli.command()
@click.option('--path', '-p', help='Path to documents directory')
@click.option('--force', '-f', is_flag=True, help='Force re-indexing of existing documents')
def standalone(path: Optional[str], force: bool):
    """Index documents with Contextual RAG (standalone mode)."""
    async def run_standalone_indexing():
        indexer = StandaloneDocumentIndexer()

        try:
            console.print(" Starting document indexer with Contextual RAG...", style="bold blue")

            # Initialize
            if not await indexer.initialize():
                console.print(" Failed to initialize indexer", style="red")
                sys.exit(1)

            console.print(" DoclingProcessor initialized with Ollama and vector store", style="green")

            # Set force re-indexing if specified
            if force:
                indexer.doc_processor.force_reindex = True

            # Index documents
            count = await indexer.index_documents(path)

            if count > 0:
                console.print(f" Successfully indexed {count} documents with Contextual RAG!", style="bold green")
            else:
                console.print(" No documents were indexed", style="yellow")

        except KeyboardInterrupt:
            console.print(" Indexing interrupted by user", style="red")
        except Exception as e:
            logger.error("Indexing failed", error=str(e))
            console.print(f" Indexing failed: {e}", style="red")
            sys.exit(1)
        finally:
            await indexer.cleanup()

    asyncio.run(run_standalone_indexing())


if __name__ == '__main__':
    cli()