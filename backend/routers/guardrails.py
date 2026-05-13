"""
Guardrails Management API Endpoints

Provides endpoints to view guardrail statistics, update configuration,
and test guardrails against sample queries.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["guardrails"])


class GuardrailConfigUpdate(BaseModel):
    """Request body for updating guardrail configuration."""
    enabled: Optional[bool] = None
    enable_injection_detection: Optional[bool] = None
    enable_topic_restriction: Optional[bool] = None
    enable_input_length_check: Optional[bool] = None
    enable_pii_redaction: Optional[bool] = None
    enable_hallucination_check: Optional[bool] = None
    enable_relevance_check: Optional[bool] = None
    min_query_length: Optional[int] = None
    max_query_length: Optional[int] = None
    hallucination_threshold: Optional[float] = None


class GuardrailTestRequest(BaseModel):
    """Request body for testing guardrails against a query."""
    query: str


def _get_guardrails_service():
    """Get guardrails service from RAG service."""
    from main import rag_service
    if not rag_service or not rag_service.guardrails_service:
        raise HTTPException(status_code=503, detail="Guardrails service not available")
    return rag_service.guardrails_service


@router.get("/guardrails/stats")
async def get_guardrails_stats():
    """Get guardrail activity statistics (blocks, detections, etc.)."""
    guardrails = _get_guardrails_service()
    return {
        "stats": guardrails.get_stats(),
        "config": guardrails.get_config(),
    }


@router.get("/guardrails/config")
async def get_guardrails_config():
    """Get current guardrail configuration."""
    guardrails = _get_guardrails_service()
    return guardrails.get_config()


@router.post("/guardrails/config")
async def update_guardrails_config(update: GuardrailConfigUpdate):
    """Update guardrail configuration at runtime."""
    guardrails = _get_guardrails_service()
    updates = {k: v for k, v in update.model_dump().items() if v is not None}
    guardrails.update_config(**updates)
    return {"message": "Guardrail config updated", "config": guardrails.get_config()}


@router.post("/guardrails/test")
async def test_guardrails(request: GuardrailTestRequest):
    """
    Test input guardrails against a query without processing it.
    Useful for debugging which guardrail would block a specific query.
    """
    guardrails = _get_guardrails_service()
    result = guardrails.check_input(request.query)
    return {
        "query": request.query,
        "passed": result.passed,
        "guardrail_name": result.guardrail_name,
        "message": result.message,
        "details": result.details,
    }
