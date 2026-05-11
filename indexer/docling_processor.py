"""Document processing using DoclingReader and LlamaIndex with Ollama."""

import hashlib
import mimetypes
import asyncio
import os
import copy
from pathlib import Path
from typing import List, Optional, Dict, Any

import structlog
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.docling import DoclingReader

from config import IndexerConfig


logger = structlog.get_logger()


# Contextual RAG prompt template
CONTEXT_PROMPT_TEMPLATE = """You are analyzing a document. Your task is to provide context for a specific chunk.

<document>
{WHOLE_DOCUMENT}
</document>

<chunk>
{CHUNK_CONTENT}
</chunk>

Provide a brief context (1-2 sentences) explaining:
1. Which section/topic this chunk relates to
2. How it connects to the overall document structure
3. Its relationship to other sections or procedures

Respond with only the context, nothing else."""


class DoclingDocumentProcessor:
    """Handles document processing using DoclingReader and LlamaIndex with Ollama."""

    def __init__(self, config: IndexerConfig):
        self.config = config
        self.force_reindex = False

        # Create markdown output directory (handle both Docker and local environments)
        try:
            self.markdown_dir = Path("/app/markdown_output")
            self.markdown_dir.mkdir(parents=True, exist_ok=True)
        except (OSError, PermissionError):
            # Fallback for local testing - use relative path
            self.markdown_dir = Path("./markdown_output_local")
            self.markdown_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.docling_reader = None
        self.node_parser = None
        self.vector_store = None
        self.index = None
        self.embedding_model = None
        self.llm = None
        self.context_llm = None  # For contextual RAG

    async def initialize(self):
        """Initialize the document processor."""
        try:
            # Initialize DoclingReader (like the sample reference code)
            self.docling_reader = DoclingReader(keep_image=False)

            # Initialize node parser for chunking
            self.node_parser = SentenceSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=" "
            )

            # Initialize Ollama models (detect local vs Docker environment)
            ollama_base_url = self._get_ollama_base_url()
            self.embedding_model = OllamaEmbedding(
                model_name="nomic-embed-text:v1.5",
                base_url=ollama_base_url
            )

            self.llm = Ollama(
                model="llama3.2:1b",
                base_url=ollama_base_url
            )

            # Initialize context LLM for contextual RAG
            self.context_llm = Ollama(
                model="llama3.2:1b",  # Using same model for context generation
                base_url=ollama_base_url,
                request_timeout=120.0
            )

            # Initialize PostgreSQL vector store
            self.vector_store = PGVectorStore.from_params(
                database=self._extract_db_name(self.config.database_url),
                host=self._extract_host(self.config.database_url),
                port=self._extract_port(self.config.database_url),
                user=self._extract_user(self.config.database_url),
                password=self._extract_password(self.config.database_url),
                table_name="llamaindex_vectors_copy",
                embed_dim=768,  # nomic-embed-text:v1.5 dimension
            )

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=self.vector_store)

            # Initialize vector index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embedding_model,
                storage_context=storage_context
            )

            logger.info("Document processor initialized successfully with DoclingReader and Ollama")

        except Exception as e:
            logger.error("Failed to initialize document processor", error=str(e))
            raise

    def _extract_db_name(self, db_url: str) -> str:
        """Extract database name from database URL."""
        return db_url.split('/')[-1]

    def _extract_host(self, db_url: str) -> str:
        """Extract host from database URL."""
        # postgresql://user:pass@host:port/db
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

    def _get_ollama_base_url(self) -> str:
        """Get Ollama base URL based on environment detection."""
        # Check if we're running in Docker (presence of /.dockerenv or Docker-specific hostname)
        if os.path.exists('/.dockerenv'):
            return "http://ollama:11434"  # Docker service name
        else:
            return "http://localhost:11434"  # Local development

    def create_contextual_nodes(self, nodes: List, whole_document: str) -> List:
        """Create contextual nodes using Ollama LLM for enhanced retrieval."""
        logger.info(f"Creating contextual nodes for {len(nodes)} chunks...")

        enhanced_nodes = []

        for i, node in enumerate(nodes):
            try:
                # Create a deep copy of the node
                enhanced_node = copy.deepcopy(node)

                # Generate context using LLM
                context_prompt = CONTEXT_PROMPT_TEMPLATE.format(
                    WHOLE_DOCUMENT=whole_document[:8000],  # Limit document size for context
                    CHUNK_CONTENT=node.text
                )

                # Get context from LLM
                context_response = self.context_llm.complete(context_prompt)
                context = str(context_response).strip()

                # Add context to metadata
                enhanced_node.metadata["context"] = context
                enhanced_node.metadata["has_context"] = True

                enhanced_nodes.append(enhanced_node)

                # Log progress every 10 nodes
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated context for {i + 1}/{len(nodes)} nodes")

            except Exception as e:
                logger.warning(f"Failed to generate context for node {i}: {e}")
                # Fallback: use original node with basic context
                fallback_node = copy.deepcopy(node)
                fallback_node.metadata["context"] = f"Part of {node.metadata.get('source_file', 'document')}"
                fallback_node.metadata["has_context"] = False
                enhanced_nodes.append(fallback_node)

        logger.info(f" Created {len(enhanced_nodes)} contextual nodes")
        return enhanced_nodes

    async def process_batch(self, file_paths: List[Path]) -> List[bool]:
        """Process a batch of documents."""
        results = []

        for file_path in file_paths:
            try:
                result = await self.process_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error("Failed to process document",
                           file=str(file_path),
                           error=str(e))
                results.append(False)

        return results

    async def process_document(self, file_path: Path) -> bool:
        """Process a single document using DoclingReader."""
        try:
            # Check if document already processed (unless force reindex)
            if not self.force_reindex:
                doc_hash = self._calculate_file_hash(file_path)
                if await self._is_document_indexed(file_path, doc_hash):
                    logger.info("Document already indexed, skipping", file=str(file_path))
                    return True

            # Use DoclingReader to load the document (this does the heavy processing once)
            logger.info("Processing document with DoclingReader", file=str(file_path))
            docs = self.docling_reader.load_data(file_path=[str(file_path)])

            if not docs:
                logger.warning("No content extracted by DoclingReader", file=str(file_path))
                return False

            # Save markdown from the processed document result
            markdown_path = await self._save_markdown_from_docs(file_path, docs)
            if markdown_path:
                logger.info("Successfully saved markdown from DoclingReader result",
                           source=str(file_path),
                           markdown=str(markdown_path))

            # Add metadata to documents
            for doc in docs:
                doc.metadata.update({
                    'source_file': file_path.name,
                    'file_path': str(file_path),
                    'markdown_path': str(markdown_path),
                    'file_size': file_path.stat().st_size,
                    'file_type': mimetypes.guess_type(str(file_path))[0],
                    'doc_hash': self._calculate_file_hash(file_path),
                    'processor': 'docling',
                    'embedding_model': 'nomic-embed-text:v1.5'
                })

            # Parse into nodes (chunks)
            nodes = self.node_parser.get_nodes_from_documents(docs)

            # Add metadata to each node
            for i, node in enumerate(nodes):
                node.metadata.update({
                    'chunk_id': i,
                    'total_chunks': len(nodes),
                    'source_document': file_path.name
                })

            # Generate contextual nodes using the whole document text
            whole_document_text = "\n\n".join([doc.text for doc in docs])
            contextual_nodes = self.create_contextual_nodes(nodes, whole_document_text)

            # Add to vector index using contextual nodes
            self.index.insert_nodes(contextual_nodes)

            logger.info("Document processed successfully with DoclingReader and Contextual RAG",
                       file=str(file_path),
                       semantic_docs=len(docs),
                       chunks=len(contextual_nodes))

            return True

        except Exception as e:
            logger.error("Error processing document",
                        file=str(file_path),
                        error=str(e))
            return False

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for change detection."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    async def _is_document_indexed(self, file_path: Path, file_hash: str) -> bool:
        """Check if document is already indexed."""
        try:
            # Query the vector store metadata to check if document exists
            # This is a simplified check - in production you'd query the database directly
            return False
        except Exception as e:
            logger.warning("Could not check if document is indexed", error=str(e))
            return False

    async def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the current index."""
        try:
            # Get basic stats from the vector store
            # This would need to be implemented based on the actual vector store API
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "embedding_model": "nomic-embed-text:v1.5"
            }
        except Exception as e:
            logger.error("Failed to get index stats", error=str(e))
            return {}

    async def _save_markdown_from_docs(self, file_path: Path, docs: List[Document]) -> Optional[Path]:
        """Save markdown from already processed DoclingReader documents."""
        try:
            # Generate markdown filename
            markdown_filename = f"{file_path.stem}.md"
            markdown_path = self.markdown_dir / markdown_filename

            # Combine all document content as markdown
            markdown_content = ""
            for i, doc in enumerate(docs):
                if i > 0:
                    markdown_content += "\n\n---\n\n"  # Separator between documents
                markdown_content += doc.text

            # Save markdown file
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info("Successfully saved markdown from DoclingReader",
                       source=str(file_path),
                       markdown=str(markdown_path),
                       size=len(markdown_content),
                       doc_count=len(docs))

            return markdown_path

        except Exception as e:
            logger.error("Failed to save markdown from documents",
                        file=str(file_path),
                        error=str(e))
            return None

    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self.vector_store:
                # Close database connections if needed
                pass
            logger.info("Document processor cleanup completed")
        except Exception as e:
            logger.error("Error during cleanup", error=str(e))