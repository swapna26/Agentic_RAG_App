"""
Cache Management API Endpoints

Provides endpoints to view cache statistics, update configuration,
and clear the cache at runtime.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["cache"])


class CacheConfigUpdate(BaseModel):
    """Request body for updating cache configuration."""
    enabled: Optional[bool] = None
    ttl_seconds: Optional[int] = None
    max_size: Optional[int] = None
    semantic_threshold: Optional[float] = None


def _get_cache_service():
    """Get cache service from RAG service."""
    from main import rag_service
    if not rag_service or not rag_service.cache_service:
        raise HTTPException(status_code=503, detail="Cache service not available")
    return rag_service.cache_service


@router.get("/cache/stats")
async def get_cache_stats():
    """Get cache performance statistics (hit rate, size, etc.)."""
    cache = _get_cache_service()
    return {
        "stats": cache.get_stats(),
        "config": cache.get_config(),
    }


@router.get("/cache/config")
async def get_cache_config():
    """Get current cache configuration."""
    cache = _get_cache_service()
    return cache.get_config()


@router.post("/cache/config")
async def update_cache_config(update: CacheConfigUpdate):
    """Update cache configuration at runtime."""
    cache = _get_cache_service()
    new_config = cache.update_config(
        enabled=update.enabled,
        ttl_seconds=update.ttl_seconds,
        max_size=update.max_size,
        semantic_threshold=update.semantic_threshold,
    )
    return {"message": "Cache config updated", "config": new_config}


@router.post("/cache/clear")
async def clear_cache():
    """Clear all cached entries."""
    cache = _get_cache_service()
    count = cache.clear()
    return {"message": f"Cache cleared", "entries_removed": count}
