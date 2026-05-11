# Document Indexer

Document processing and indexing system with Contextual RAG implementation using DoclingReader and Ollama.

## Features

- **Multi-format Support**: PDF, DOCX, TXT, MD, HTML
- **Contextual RAG**: Enhanced chunking with LLM-generated context
- **Vector Storage**: PostgreSQL with PGVector extension
- **Ollama Integration**: Local models for embeddings and context generation

## Setup

### Requirements
```bash
# Install Ollama models
ollama pull llama3.2:1b
ollama pull nomic-embed-text:v1.5
```

### Environment Variables
```env
DATABASE_URL=postgresql://raguser:ragpassword@localhost:5432/agentic_rag
DOCUMENTS_PATH=./documents
CHUNK_SIZE=3000
CHUNK_OVERLAP=200
BATCH_SIZE=10
```

## Usage

### Docker (Recommended)
```bash
# Build and run
docker-compose build document_indexer
docker-compose --profile indexer up

# Index documents
docker-compose run --rm document_indexer python main.py index
```

### Local Development
```bash
pip install -r requirements.txt
python main.py index --path ./documents
```

## Commands

```bash
# Index documents with database storage
python main.py index [--path PATH] [--force]

# Standalone processing (no database)
python main.py standalone [--path PATH] [--force]

# Check system status
python main.py status
```

## Supported Formats

- PDF: Full parsing with layout preservation
- DOCX: Text and structure extraction
- TXT/MD: Plain text processing
- HTML: Web content processing

## Output

- **Database**: Vector embeddings in PostgreSQL
- **Markdown**: Converted documents in `markdown_output/`
- **Metadata**: Rich chunk metadata with context