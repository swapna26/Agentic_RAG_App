"""
Fraud Detection API Endpoints

Provides an endpoint to view fraud prediction statistics
for drift monitoring.
"""

from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["fraud"])


def _get_fraud_service():
    """Get fraud detection service from RAG service."""
    from main import rag_service
    if not rag_service or not rag_service.fraud_detection_service:
        raise HTTPException(status_code=503, detail="Fraud detection service not available")
    return rag_service.fraud_detection_service


@router.get("/fraud/stats")
async def get_fraud_prediction_stats():
    """Get fraud prediction statistics for drift monitoring."""
    service = _get_fraud_service()
    return service.get_prediction_stats()
