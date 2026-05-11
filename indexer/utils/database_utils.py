"""Database utilities for the document indexer."""

import asyncio
from typing import Optional

import asyncpg
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker


logger = structlog.get_logger()


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.async_engine = None
        self.session_factory = None
    
    async def initialize(self):
        """Initialize database connections."""
        try:
            # Test connection
            await self._test_connection()
            
            # Create async engine for LlamaIndex
            async_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')
            self.async_engine = create_async_engine(async_url)
            
            # Create session factory
            self.session_factory = sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error("Database initialization failed", error=str(e))
            raise
    
    async def _test_connection(self):
        """Test database connection."""
        # Parse connection parameters
        url_parts = self.database_url.replace('postgresql://', '').split('@')
        user_pass = url_parts[0].split(':')
        host_db = url_parts[1].split('/')
        host_port = host_db[0].split(':')
        
        user = user_pass[0]
        password = user_pass[1]
        host = host_port[0]
        port = int(host_port[1]) if len(host_port) > 1 else 5432
        database = host_db[1]
        
        try:
            conn = await asyncpg.connect(
                user=user,
                password=password,
                database=database,
                host=host,
                port=port,
                timeout=10
            )
            
            # Test basic query
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            
            if result != 1:
                raise Exception("Database connection test failed")
                
            logger.info("Database connection test successful")
            
        except Exception as e:
            logger.error("Database connection failed", error=str(e))
            raise
    
    async def get_document_count(self) -> int:
        """Get the count of indexed documents with detailed debugging."""
        try:
            # Parse connection parameters from URL
            url_parts = self.database_url.replace('postgresql://', '').split('@')
            user_pass = url_parts[0].split(':')
            host_db = url_parts[1].split('/')
            host_port = host_db[0].split(':')
            
            user = user_pass[0]
            password = user_pass[1]
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else 5432
            database = host_db[1]
            
            # Connect to database
            conn = await asyncpg.connect(
                user=user,
                password=password,
                database=database,
                host=host,
                port=port,
                timeout=10
            )
            
            # First, let's see ALL tables in the database
            all_tables_query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
            
            all_tables = await conn.fetch(all_tables_query)
            table_names = [row['table_name'] for row in all_tables]
            logger.info("All tables in database", tables=table_names)
            
            # Check specifically for vector-related tables
            vector_tables = [t for t in table_names if any(keyword in t.lower() for keyword in ['vector', 'llama', 'embedding', 'data_'])]
            logger.info("Vector-related tables found", tables=vector_tables)
            
            # If we have vector tables, let's examine their structure and content
            total_documents = 0
            
            for table_name in vector_tables:
                try:
                    # Get column info
                    columns_query = f"""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                    """
                    columns = await conn.fetch(columns_query)
                    column_info = [(row['column_name'], row['data_type']) for row in columns]
                    logger.info("Table structure", table=table_name, columns=column_info)
                    
                    # Get row count
                    count_query = f"SELECT COUNT(*) FROM {table_name}"
                    row_count = await conn.fetchval(count_query)
                    logger.info("Table row count", table=table_name, rows=row_count)
                    
                    # If this table has rows, try to get document count
                    if row_count > 0:
                        # Check if table has metadata column
                        has_metadata = any('metadata' in col[0].lower() for col in column_info)
                        has_node_info = any('node_info' in col[0].lower() for col in column_info)
                        
                        if has_metadata:
                            # Try different metadata field names for document identification
                            metadata_queries = [
                                f"SELECT COUNT(DISTINCT metadata_->>'source_document') FROM {table_name} WHERE metadata_->>'source_document' IS NOT NULL",
                                f"SELECT COUNT(DISTINCT metadata_->>'file_name') FROM {table_name} WHERE metadata_->>'file_name' IS NOT NULL",
                                f"SELECT COUNT(DISTINCT metadata_->>'filename') FROM {table_name} WHERE metadata_->>'filename' IS NOT NULL",
                                f"SELECT COUNT(DISTINCT metadata_->>'file_path') FROM {table_name} WHERE metadata_->>'file_path' IS NOT NULL",
                            ]
                            
                            for query in metadata_queries:
                                try:
                                    result = await conn.fetchval(query)
                                    if result and result > 0:
                                        logger.info("Found documents via metadata", 
                                                   table=table_name,
                                                   field=query.split(">>")[1].split("'")[1],
                                                   count=result)
                                        total_documents = max(total_documents, int(result))
                                        break
                                except Exception as e:
                                    logger.debug("Metadata query failed", query=query, error=str(e))
                                    continue
                        
                        elif has_node_info:
                            # Try node_info field
                            try:
                                query = f"SELECT COUNT(DISTINCT node_info->>'file_name') FROM {table_name} WHERE node_info->>'file_name' IS NOT NULL"
                                result = await conn.fetchval(query)
                                if result and result > 0:
                                    logger.info("Found documents via node_info", 
                                               table=table_name,
                                               count=result)
                                    total_documents = max(total_documents, int(result))
                            except Exception as e:
                                logger.debug("Node info query failed", error=str(e))
                        
                        else:
                            # Try ref_doc_id or other identifier columns
                            id_columns = [col[0] for col in column_info if 'ref_doc' in col[0].lower() or 'doc_id' in col[0].lower()]
                            for col in id_columns:
                                try:
                                    query = f"SELECT COUNT(DISTINCT {col}) FROM {table_name} WHERE {col} IS NOT NULL"
                                    result = await conn.fetchval(query)
                                    if result and result > 0:
                                        logger.info("Found documents via ID column", 
                                                   table=table_name,
                                                   column=col,
                                                   count=result)
                                        total_documents = max(total_documents, int(result))
                                        break
                                except Exception as e:
                                    logger.debug("ID column query failed", column=col, error=str(e))
                                    continue
                
                except Exception as e:
                    logger.warning("Failed to examine table", table=table_name, error=str(e))
                    continue
            
            await conn.close()
            
            if total_documents > 0:
                logger.info("Final document count determined", count=total_documents)
            else:
                logger.info("No documents found in any vector tables")
            
            return total_documents
                    
        except Exception as e:
            logger.error("Failed to get document count", error=str(e))
            return 0
    
    async def cleanup(self):
        """Cleanup database connections."""
        try:
            if self.async_engine:
                await self.async_engine.dispose()
            logger.info("Database cleanup completed")
        except Exception as e:
            logger.error("Error during database cleanup", error=str(e))