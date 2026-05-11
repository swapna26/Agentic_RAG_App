"""Document management endpoints."""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import structlog

logger = structlog.get_logger()

router = APIRouter()

class DocumentStats(BaseModel):
    total_documents: int
    total_chunks: int
    embedding_model: str
    llm_model: str
    status: str

@router.get("/documents/stats")
async def get_document_stats(request: Request) -> DocumentStats:
    """Get document index statistics."""
    try:
        rag_service = getattr(request.app.state, 'rag_service', None)
        if not rag_service:
            import main
            rag_service = main.rag_service
            
        if not rag_service:
            raise HTTPException(status_code=503, detail="RAG service not available")
        
        stats = await rag_service.get_index_stats()
        
        return DocumentStats(
            total_documents=stats.get("total_documents", 0),
            total_chunks=stats.get("total_chunks", 0),
            embedding_model=stats.get("embedding_model", "unknown"),
            llm_model=stats.get("llm_model", "unknown"),
            status=stats.get("status", "unknown")
        )
        
    except Exception as e:
        logger.error("Failed to get document stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/health")
async def get_documents_health(request: Request) -> Dict[str, Any]:
    """Check document index health."""
    try:
        rag_service = getattr(request.app.state, 'rag_service', None)
        if not rag_service:
            import main
            rag_service = main.rag_service
            
        if not rag_service:
            return {"status": "unhealthy", "reason": "RAG service not available"}
        
        is_healthy = await rag_service.is_healthy()
        stats = await rag_service.get_index_stats()
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "rag_service": "available" if rag_service else "unavailable",
            "total_documents": stats.get("total_documents", 0),
            "total_chunks": stats.get("total_chunks", 0)
        }
        
    except Exception as e:
        logger.error("Document health check failed", error=str(e))
        return {"status": "error", "error": str(e)}