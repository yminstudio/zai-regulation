"""Weaviate 검색 유틸."""
from __future__ import annotations

import json
from dataclasses import dataclass

import requests
from openai import OpenAI

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, PROJECT_WEAVIATE_CLASS, WEAVIATE_URL


@dataclass
class SearchHit:
    original_id: str
    title: str
    source_url: str
    summary_text: str
    summary_keywords: list[str]
    rule_names: list[str]
    reg_date: str
    reg_user: str
    score: float
    distance: float | None = None


@dataclass
class SearchResult:
    query: str
    hits: list[SearchHit]
    mode: str = "vector"

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0


def _embed_query(query: str) -> list[float]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=OPENAI_API_KEY.strip(), timeout=120.0)
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=query)
    return resp.data[0].embedding


def _build_graphql(class_name: str, vector: list[float], limit: int) -> str:
    vec = json.dumps(vector)
    return (
        "{ Get { "
        f"{class_name}(nearVector: {{vector: {vec}}}, limit: {limit}) "
        "{ original_id title source_url summary_text summary_keywords rule_names reg_date reg_user "
        "_additional { distance } } } }"
    )


def _build_hybrid_graphql(
    class_name: str,
    *,
    query_text: str,
    vector: list[float],
    limit: int,
    alpha: float,
    autocut: int | None = None,
) -> str:
    vec = json.dumps(vector)
    q = json.dumps(query_text)
    autocut_part = f", autocut: {int(autocut)}" if autocut is not None else ""
    return (
        "{ Get { "
        f"{class_name}(hybrid: {{query: {q}, alpha: {alpha}, vector: {vec}, fusionType: relativeScoreFusion}}{autocut_part}, limit: {limit}) "
        "{ original_id title source_url summary_text summary_keywords rule_names reg_date reg_user "
        "_additional { distance score } } } }"
    )


def _extract_items_or_raise(body: dict, class_name: str) -> list[dict]:
    errors = body.get("errors") or []
    if errors:
        msg = "; ".join(str(e.get("message", "")).strip() for e in errors if isinstance(e, dict))
        raise RuntimeError(f"Weaviate GraphQL error: {msg or errors}")
    return (((body.get("data") or {}).get("Get") or {}).get(class_name) or [])


def _parse_hits(items: list[dict]) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for item in items:
        add = item.get("_additional", {}) or {}
        distance = add.get("distance")
        raw_score = add.get("score")
        score = 0.0
        if raw_score is not None:
            try:
                score = float(raw_score)
            except Exception:
                score = 0.0
        elif distance is not None:
            score = 1.0 - float(distance)
        hits.append(
            SearchHit(
                original_id=item.get("original_id", ""),
                title=item.get("title", ""),
                source_url=item.get("source_url", ""),
                summary_text=item.get("summary_text", ""),
                summary_keywords=item.get("summary_keywords", []) or [],
                rule_names=item.get("rule_names", []) or [],
                reg_date=item.get("reg_date", ""),
                reg_user=item.get("reg_user", ""),
                score=score,
                distance=float(distance) if distance is not None else None,
            )
        )
    return hits


def vector_search(query: str, *, limit: int = 5, class_name: str = PROJECT_WEAVIATE_CLASS) -> SearchResult:
    query = (query or "").strip()
    if not query:
        return SearchResult(query=query, hits=[])

    vector = _embed_query(query)
    gql = _build_graphql(class_name, vector, limit)
    url = f"{WEAVIATE_URL.rstrip('/')}/v1/graphql"
    resp = requests.post(url, json={"query": gql}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    items = _extract_items_or_raise(body, class_name)

    hits = _parse_hits(items)
    return SearchResult(query=query, hits=hits, mode="vector")


def hybrid_search(
    query: str,
    *,
    limit: int = 30,
    class_name: str = PROJECT_WEAVIATE_CLASS,
    alpha: float = 0.35,
    autocut: int = 2,
) -> SearchResult:
    query = (query or "").strip()
    if not query:
        return SearchResult(query=query, hits=[], mode="hybrid")

    vector = _embed_query(query)
    gql = _build_hybrid_graphql(
        class_name,
        query_text=query,
        vector=vector,
        limit=limit,
        alpha=alpha,
        autocut=autocut,
    )
    url = f"{WEAVIATE_URL.rstrip('/')}/v1/graphql"
    resp = requests.post(url, json={"query": gql}, timeout=60)
    resp.raise_for_status()
    body = resp.json()
    items = _extract_items_or_raise(body, class_name)
    hits = _parse_hits(items)
    return SearchResult(query=query, hits=hits, mode="hybrid")


def search_with_fallback(
    query: str,
    *,
    limit: int = 30,
    class_name: str = PROJECT_WEAVIATE_CLASS,
    alpha: float = 0.35,
    autocut: int = 2,
) -> SearchResult:
    """기본은 hybrid, 실패 시 vector fallback."""
    try:
        return hybrid_search(query, limit=limit, class_name=class_name, alpha=alpha, autocut=autocut)
    except Exception:
        return vector_search(query, limit=limit, class_name=class_name)

