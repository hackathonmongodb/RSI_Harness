"""MongoDB Atlas Embedding & Reranking API (ai.mongodb.com / Voyage)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from rsi_harness.config import mongodb_ai_endpoint, voyage_api_key


class MongoAIError(RuntimeError):
    pass


def _headers() -> dict[str, str]:
    key = voyage_api_key()
    if not key:
        raise MongoAIError(
            "VOYAGE_API_KEY missing. Add your MongoDB AI (ai.mongodb.com) key to .env."
        )
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def embed(texts: list[str], *, input_type: str = "query") -> list[list[float]]:
    """POST /v1/embeddings"""
    if not texts:
        return []
    base = mongodb_ai_endpoint().rstrip("/")
    model = os.getenv("VOYAGE_EMBED_MODEL", "voyage-4-large")
    r = httpx.post(
        f"{base}/embeddings",
        headers=_headers(),
        json={"input": texts, "model": model, "input_type": input_type},
        timeout=45,
    )
    if r.status_code >= 400:
        raise MongoAIError(f"embeddings failed HTTP {r.status_code}: {r.text[:300]}")
    data = r.json().get("data") or []
    # API may return unordered; sort by index if present
    data = sorted(data, key=lambda d: int(d.get("index", 0)))
    return [list(d.get("embedding") or []) for d in data]


def rerank(query: str, documents: list[str], top_k: int | None = None) -> list[dict[str, Any]]:
    """POST /v1/rerank — returns [{index, relevance_score}, ...] best first."""
    if not documents:
        return []
    base = mongodb_ai_endpoint().rstrip("/")
    model = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2")
    body: dict[str, Any] = {"query": query, "documents": documents, "model": model}
    if top_k is not None:
        body["top_k"] = top_k
    r = httpx.post(
        f"{base}/rerank",
        headers=_headers(),
        json=body,
        timeout=45,
    )
    if r.status_code >= 400:
        raise MongoAIError(f"rerank failed HTTP {r.status_code}: {r.text[:300]}")
    return list(r.json().get("data") or [])
