"""
Query Cache Service for Agentic RAG

Provides two layers of caching to avoid redundant LLM calls:
1. Exact cache  — SHA256 hash of query string for identical questions
2. Semantic cache — cosine similarity of query embeddings for similar questions

Cache entries have a configurable TTL and max size with LRU eviction.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class CacheEntry:
    """Single cached query-response pair."""
    query: str
    query_hash: str
    embedding: Optional[List[float]]
    response: Dict[str, Any]
    created_at: float
    hit_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        if self.last_accessed == 0.0:
            self.last_accessed = self.created_at


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    current_size: int = 0

    @property
    def total_hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_hits / self.total_requests

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "total_hits": self.total_hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "hit_rate_percent": f"{self.hit_rate * 100:.1f}%",
            "evictions": self.evictions,
            "current_size": self.current_size,
        }


class CacheService:
    """
    Two-layer query cache: exact match + semantic similarity.

    Layer 1 (exact): Hash the query string, O(1) lookup.
    Layer 2 (semantic): Compare query embedding against cached embeddings,
                        return match if cosine similarity > threshold.
    """

    def __init__(
        self,
        embedding_model=None,
        ttl_seconds: int = 3600,
        max_size: int = 500,
        semantic_threshold: float = 0.95,
        enabled: bool = True,
    ):
        self.embedding_model = embedding_model
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.semantic_threshold = semantic_threshold
        self.enabled = enabled

        # Exact cache: hash -> CacheEntry
        self._exact_cache: Dict[str, CacheEntry] = {}
        # Semantic cache: ordered list for linear scan
        self._semantic_entries: List[CacheEntry] = []

        self.stats = CacheStats()

        logger.info(
            "CacheService initialized",
            ttl=ttl_seconds,
            max_size=max_size,
            semantic_threshold=semantic_threshold,
            enabled=enabled,
        )

    # ── public API ──────────────────────────────────────────

    async def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Look up a cached response for the given query."""
        if not self.enabled:
            return None

        self.stats.total_requests += 1
        now = time.time()

        # Layer 1: exact match
        query_hash = self._hash(query)
        entry = self._exact_cache.get(query_hash)
        if entry and not self._is_expired(entry, now):
            entry.hit_count += 1
            entry.last_accessed = now
            self.stats.exact_hits += 1
            logger.info("Cache exact hit", query=query[:80], hits=entry.hit_count)
            return self._enrich_response(entry.response, "exact_cache")

        # Layer 2: semantic match
        if self.embedding_model and self._semantic_entries:
            match = await self._semantic_lookup(query, now)
            if match:
                match.hit_count += 1
                match.last_accessed = now
                self.stats.semantic_hits += 1
                logger.info(
                    "Cache semantic hit",
                    query=query[:80],
                    matched=match.query[:80],
                )
                return self._enrich_response(match.response, "semantic_cache")

        self.stats.misses += 1
        return None

    async def put(self, query: str, response: Dict[str, Any]) -> None:
        """Store a query-response pair in the cache."""
        if not self.enabled:
            return

        now = time.time()
        query_hash = self._hash(query)

        # Generate embedding for semantic cache
        embedding = None
        if self.embedding_model:
            try:
                embedding = self.embedding_model.get_text_embedding(query)
            except Exception as e:
                logger.warning("Failed to generate cache embedding", error=str(e))

        entry = CacheEntry(
            query=query,
            query_hash=query_hash,
            embedding=embedding,
            response=response,
            created_at=now,
        )

        # Store in exact cache
        self._exact_cache[query_hash] = entry

        # Store in semantic cache if we have an embedding
        if embedding is not None:
            self._semantic_entries.append(entry)

        self.stats.current_size = len(self._exact_cache)

        # Evict if over max size
        self._evict_if_needed()

        # Purge expired entries periodically (every 50 puts)
        if self.stats.total_requests % 50 == 0:
            self._purge_expired(now)

        logger.info("Cache stored", query=query[:80], size=self.stats.current_size)

    def clear(self) -> int:
        """Clear all cache entries. Returns number of entries cleared."""
        count = len(self._exact_cache)
        self._exact_cache.clear()
        self._semantic_entries.clear()
        self.stats.current_size = 0
        logger.info("Cache cleared", entries_removed=count)
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return self.stats.to_dict()

    def get_config(self) -> Dict[str, Any]:
        """Return current cache configuration."""
        return {
            "enabled": self.enabled,
            "ttl_seconds": self.ttl_seconds,
            "max_size": self.max_size,
            "semantic_threshold": self.semantic_threshold,
            "semantic_enabled": self.embedding_model is not None,
        }

    def update_config(
        self,
        enabled: Optional[bool] = None,
        ttl_seconds: Optional[int] = None,
        max_size: Optional[int] = None,
        semantic_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update cache configuration at runtime."""
        if enabled is not None:
            self.enabled = enabled
        if ttl_seconds is not None:
            self.ttl_seconds = ttl_seconds
        if max_size is not None:
            self.max_size = max_size
        if semantic_threshold is not None:
            self.semantic_threshold = semantic_threshold
        logger.info("Cache config updated", **self.get_config())
        return self.get_config()

    # ── internals ───────────────────────────────────────────

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

    def _is_expired(self, entry: CacheEntry, now: float) -> bool:
        return (now - entry.created_at) > self.ttl_seconds

    async def _semantic_lookup(self, query: str, now: float) -> Optional[CacheEntry]:
        """Find semantically similar cached query."""
        try:
            query_emb = self.embedding_model.get_text_embedding(query)
            query_vec = np.array(query_emb, dtype=np.float32)
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                return None

            best_entry = None
            best_score = 0.0

            for entry in self._semantic_entries:
                if self._is_expired(entry, now):
                    continue
                if entry.embedding is None:
                    continue

                cached_vec = np.array(entry.embedding, dtype=np.float32)
                cached_norm = np.linalg.norm(cached_vec)
                if cached_norm == 0:
                    continue

                similarity = float(np.dot(query_vec, cached_vec) / (query_norm * cached_norm))

                if similarity > best_score:
                    best_score = similarity
                    best_entry = entry

            if best_entry and best_score >= self.semantic_threshold:
                logger.debug(
                    "Semantic match found",
                    score=round(best_score, 4),
                    threshold=self.semantic_threshold,
                )
                return best_entry

        except Exception as e:
            logger.warning("Semantic lookup failed", error=str(e))

        return None

    def _evict_if_needed(self) -> None:
        """Evict least-recently-accessed entries if over max size."""
        while len(self._exact_cache) > self.max_size:
            # Find LRU entry
            lru_hash = min(self._exact_cache, key=lambda h: self._exact_cache[h].last_accessed)
            lru_entry = self._exact_cache.pop(lru_hash)

            # Remove from semantic list too
            self._semantic_entries = [
                e for e in self._semantic_entries if e.query_hash != lru_hash
            ]

            self.stats.evictions += 1
            self.stats.current_size = len(self._exact_cache)
            logger.debug("Cache evicted", query=lru_entry.query[:60])

    def _purge_expired(self, now: float) -> None:
        """Remove all expired entries."""
        expired_hashes = [
            h for h, e in self._exact_cache.items() if self._is_expired(e, now)
        ]
        for h in expired_hashes:
            del self._exact_cache[h]
            self.stats.evictions += 1

        self._semantic_entries = [
            e for e in self._semantic_entries if not self._is_expired(e, now)
        ]
        self.stats.current_size = len(self._exact_cache)

        if expired_hashes:
            logger.info("Purged expired cache entries", count=len(expired_hashes))

    @staticmethod
    def _enrich_response(response: Dict[str, Any], cache_type: str) -> Dict[str, Any]:
        """Add cache metadata to response without mutating the original."""
        enriched = dict(response)
        metadata = dict(enriched.get("metadata", {}))
        metadata["served_from_cache"] = True
        metadata["cache_type"] = cache_type
        enriched["metadata"] = metadata
        return enriched
