"""Configuration for the Document Indexer with Gemini."""

import os
from pathlib import Path
from dotenv import load_dotenv


class IndexerConfig:
    """Simple configuration class using environment variables."""
    
    def __init__(self):
        # Load environment variables
        self._load_env()
        
        # Database
        self.database_url = os.getenv(
            'DATABASE_URL', 
            'postgresql://raguser:ragpassword@localhost:5432/agentic_rag'
        )
        
        # Gemini API
        # self.gemini_api_key = os.getenv('GEMINI_API_KEY', 'your_actual_gemini_api_key_here')
        # self.gemini_model = os.getenv('GEMINI_MODEL', 'gemini-1.5-flash')
        # self.gemini_embedding_model = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/text-embedding-004')
        
        # Phoenix
        self.phoenix_base_url = os.getenv('PHOENIX_BASE_URL', 'http://localhost:6006')
        self.phoenix_project_name = os.getenv('PHOENIX_PROJECT_NAME', 'agentic_rag_indexer')
        
        # Document processing
        self.documents_path = os.getenv('DOCUMENTS_PATH', './documents')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '768'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '150'))
        
        # Processing
        self.batch_size = int(os.getenv('BATCH_SIZE', '10'))
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # Validate required settings
        self._validate()
    
    def _load_env(self):
        """Load environment variables from .env file."""
        env_paths = [
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env",
            Path.cwd() / ".env"
        ]
        
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                print(f"Loaded environment from: {env_path}")
                return
        
        print("No .env file found, using defaults")
    
    def _validate(self):
        """Validate configuration."""
        if not self.database_url.startswith(('postgresql://', 'postgresql+psycopg2://')):
            raise ValueError('DATABASE_URL must be a PostgreSQL connection string')
        
        # if self.gemini_api_key == 'your_actual_gemini_api_key_here':
        #     print("Warning: GEMINI_API_KEY not set. Please set it in your .env file")


# Global config instance
config = IndexerConfig()