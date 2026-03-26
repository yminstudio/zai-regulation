"""Embedding backend wrapper for OpenAI and local sentence-transformers."""
from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from src.config import (
    EMBEDDING_BACKEND,
    LOCAL_EMBEDDING_DEVICE,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_EMBEDDING_NORMALIZE,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)

OPENAI_BACKEND = "openai"
LOCAL_BACKEND = "local_sentence_transformers"


def _resolve_backend(backend: str | None) -> str:
    value = (backend or EMBEDDING_BACKEND or OPENAI_BACKEND).strip().lower()
    aliases = {
        "openai": OPENAI_BACKEND,
        "local": LOCAL_BACKEND,
        "st": LOCAL_BACKEND,
        "sentence_transformers": LOCAL_BACKEND,
        "local_sentence_transformers": LOCAL_BACKEND,
    }
    resolved = aliases.get(value)
    if not resolved:
        raise RuntimeError(f"unsupported embedding backend: {value}")
    return resolved


def _resolve_model(backend: str, model: str | None) -> str:
    if model and model.strip():
        return model.strip()
    if backend == OPENAI_BACKEND:
        return OPENAI_EMBEDDING_MODEL
    return LOCAL_EMBEDDING_MODEL


@lru_cache(maxsize=8)
def _get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=120.0)


@lru_cache(maxsize=8)
def _get_sentence_transformer(model_name: str, device: str) -> object:
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers is required for local embedding backend"
        ) from e
    kwargs = {}
    if device:
        kwargs["device"] = device
    return SentenceTransformer(model_name, **kwargs)


def _embed_openai(texts: list[str], model_name: str) -> list[list[float]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing for openai embedding backend")
    client = _get_openai_client(OPENAI_API_KEY.strip())
    vectors: list[list[float]] = []
    chunk_size = 32
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i : i + chunk_size]
        resp = client.embeddings.create(model=model_name, input=chunk)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def _embed_local(texts: list[str], model_name: str) -> list[list[float]]:
    model = _get_sentence_transformer(model_name, LOCAL_EMBEDDING_DEVICE)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=LOCAL_EMBEDDING_NORMALIZE,
        show_progress_bar=False,
    )
    return [vec.tolist() for vec in vectors]


def embed_texts(
    texts: list[str],
    *,
    backend: str | None = None,
    model: str | None = None,
) -> tuple[list[list[float]], str, str]:
    resolved_backend = _resolve_backend(backend)
    resolved_model = _resolve_model(resolved_backend, model)
    if resolved_backend == OPENAI_BACKEND:
        vectors = _embed_openai(texts, resolved_model)
    else:
        vectors = _embed_local(texts, resolved_model)
    return vectors, resolved_backend, resolved_model


def embed_query(
    query: str,
    *,
    backend: str | None = None,
    model: str | None = None,
) -> tuple[list[float], str, str]:
    vectors, resolved_backend, resolved_model = embed_texts(
        [query],
        backend=backend,
        model=model,
    )
    return vectors[0], resolved_backend, resolved_model
