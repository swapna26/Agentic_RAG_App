"""Database client for accessing vector store and retrieving documents."""

import asyncio
import asyncpg
import json
import structlog
from typing import List, Dict, Any, Optional
from config import config

logger = structlog.get_logger()


class DatabaseClient:
    """Client for interacting with PostgreSQL vector database."""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or config.database_url
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool."""
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10
            )
            logger.info("Database connection pool initialized")
        except Exception as e:
            logger.error("Failed to initialize database connection", error=str(e))
            raise

    async def get_document_count(self) -> int:
        """Get total number of documents in the vector store."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    f"SELECT COUNT(DISTINCT metadata_->>'source_file') FROM {config.vector_table}"
                )
                return result or 0
        except Exception as e:
            logger.error("Failed to get document count", error=str(e))
            return 0

    async def get_chunk_count(self) -> int:
        """Get total number of chunks in the vector store."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {config.vector_table}"
                )
                return result or 0
        except Exception as e:
            logger.error("Failed to get chunk count", error=str(e))
            return 0

    async def get_sample_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get a sample of documents for test question generation."""
        try:
            async with self.pool.acquire() as conn:
                query = f"""
                    SELECT
                        metadata_->>'source_file' as source_file,
                        metadata_->>'file_path' as file_path,
                        text
                    FROM {config.vector_table}
                    WHERE metadata_->>'source_file' IS NOT NULL
                    ORDER BY RANDOM()
                    LIMIT $1
                """
                rows = await conn.fetch(query, limit)

                return [
                    {
                        "source_file": row["source_file"],
                        "file_path": row["file_path"],
                        "content": row["text"]
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("Failed to get sample documents", error=str(e))
            return []

    async def search_similar_chunks(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar chunks using vector similarity."""
        try:
            async with self.pool.acquire() as conn:
                # Convert embedding to PostgreSQL array format
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                query = f"""
                    SELECT
                        text,
                        metadata_,
                        embedding <-> $1::vector as distance
                    FROM {config.vector_table}
                    ORDER BY embedding <-> $1::vector
                    LIMIT $2
                """

                rows = await conn.fetch(query, embedding_str, top_k)

                return [
                    {
                        "content": row["text"],
                        "metadata": json.loads(row["metadata_"]) if row["metadata_"] else {},
                        "distance": float(row["distance"])
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("Failed to search similar chunks", error=str(e))
            return []

    async def get_document_chunks(self, source_file: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            async with self.pool.acquire() as conn:
                query = f"""
                    SELECT
                        text,
                        metadata_,
                        embedding
                    FROM {config.vector_table}
                    WHERE metadata_->>'source_file' = $1
                    ORDER BY (metadata_->>'chunk_id')::int
                """

                rows = await conn.fetch(query, source_file)

                return [
                    {
                        "content": row["text"],
                        "metadata": json.loads(row["metadata_"]) if row["metadata_"] else {},
                        "embedding": list(row["embedding"]) if row["embedding"] else None
                    }
                    for row in rows
                ]
        except Exception as e:
            logger.error("Failed to get document chunks", error=str(e), source_file=source_file)
            return []

    async def get_all_source_files(self) -> List[str]:
        """Get list of all source files in the database."""
        try:
            async with self.pool.acquire() as conn:
                query = f"""
                    SELECT DISTINCT metadata_->>'source_file' as source_file
                    FROM {config.vector_table}
                    WHERE metadata_->>'source_file' IS NOT NULL
                    ORDER BY source_file
                """

                rows = await conn.fetch(query)
                return [row["source_file"] for row in rows]
        except Exception as e:
            logger.error("Failed to get source files", error=str(e))
            return []

    async def cleanup(self):
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")