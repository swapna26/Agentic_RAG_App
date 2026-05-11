"""Configuration for the RAG Backend API."""

import os
from pathlib import Path
from dotenv import load_dotenv


class BackendConfig:
    """Configuration for the RAG backend."""
    
    def __init__(self):
        # Load environment variables
        self._load_env()
        
        # Database
        self.database_url = os.getenv(
            'DATABASE_URL', 
            'postgresql://raguser:ragpassword@localhost:5432/agentic_rag'
        )
        
        # Ollama API
        self.ollama_base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.ollama_model = os.getenv('OLLAMA_MODEL', 'llama3.2:1b')
        self.ollama_embedding_model = os.getenv('OLLAMA_EMBEDDING_MODEL', 'nomic-embed-text:v1.5')
        
        # Phoenix
        self.phoenix_base_url = os.getenv('PHOENIX_BASE_URL', 'http://localhost:6006')
        self.phoenix_project_name = os.getenv('PHOENIX_PROJECT_NAME', 'agentic_rag_backend')
        
        # RAG Settings
        #self.chunk_size = int(os.getenv('CHUNK_SIZE', '700'))
        #self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        self.similarity_top_k = int(os.getenv('SIMILARITY_TOP_K', '5'))
        self.max_tokens = int(os.getenv('MAX_TOKENS', '4000'))
        self.temperature = float(os.getenv('TEMPERATURE', '0.1'))
        
        # Server Settings
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))
        self.cors_origins = os.getenv('CORS_ORIGINS', '*').split(',')
        
        # OpenWebUI Integration
        self.model_name = os.getenv('MODEL_NAME', 'agentic-rag-ollama')
        self.model_description = os.getenv('MODEL_DESCRIPTION', 'Agentic RAG System with Ollama')
        
        # Crew AI
        self.crew_verbose = os.getenv('CREW_VERBOSE', 'true').lower() == 'true'
        self.crew_memory = os.getenv('CREW_MEMORY', 'true').lower() == 'true'
        
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
        
        if not self.ollama_base_url:
            print("Warning: OLLAMA_BASE_URL not set. Using default: http://localhost:11434")


# Global config instance
config = BackendConfig()