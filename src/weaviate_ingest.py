"""Weaviate 적재 유틸 (안전 가드 포함)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Iterable

import requests
from openai import OpenAI

from src.config import EMBEDDING_MODEL, OPENAI_API_KEY, WEAVIATE_URL

ALLOWED_CLASS_PREFIX = "ZaiRegulation"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_class_name(class_name: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", class_name or ""):
        raise ValueError(f"invalid weaviate class name: {class_name}")
    if not class_name.startswith(ALLOWED_CLASS_PREFIX):
        raise ValueError(
            f"class '{class_name}' is not allowed. must start with '{ALLOWED_CLASS_PREFIX}'"
        )
    return class_name


def _wv_request(method: str, path: str, *, payload: dict | None = None) -> requests.Response:
    base = WEAVIATE_URL.rstrip("/")
    url = f"{base}{path}"
    resp = requests.request(method, url, json=payload, timeout=60)
    return resp


def _get_schema_classes() -> list[str]:
    resp = _wv_request("GET", "/v1/schema")
    resp.raise_for_status()
    data = resp.json()
    classes = data.get("classes", []) or []
    return [c.get("class", "") for c in classes if c.get("class")]


def _get_class_properties(class_name: str) -> set[str]:
    resp = _wv_request("GET", f"/v1/schema/{class_name}")
    if resp.status_code != 200:
        return set()
    data = resp.json() or {}
    props = data.get("properties", []) or []
    return {str(p.get("name", "")).strip() for p in props if str(p.get("name", "")).strip()}


def _ensure_class_properties(class_name: str, properties: list[dict]) -> None:
    existing = _get_class_properties(class_name)
    for prop in properties:
        name = str(prop.get("name", "")).strip()
        if not name or name in existing:
            continue
        resp = _wv_request("POST", f"/v1/schema/{class_name}/properties", payload=prop)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"add schema property failed ({name}): {resp.status_code} {resp.text}")


def _create_schema(class_name: str) -> None:
    schema = {
        "class": class_name,
        "description": "Zai regulation documents",
        "vectorizer": "none",
        "properties": [
            {"name": "original_id", "dataType": ["text"]},
            {"name": "title", "dataType": ["text"]},
            {"name": "reg_num", "dataType": ["int"]},
            {"name": "reg_user", "dataType": ["text"]},
            {"name": "reg_date", "dataType": ["date"]},
            {"name": "source_url", "dataType": ["text"]},
            {"name": "source_text", "dataType": ["text"]},
            {"name": "file_info_json", "dataType": ["text"]},
            {"name": "summary_text", "dataType": ["text"]},
            {"name": "summary_keywords", "dataType": ["text[]"]},
            {"name": "rule_names", "dataType": ["text[]"]},
            {"name": "embedding_text", "dataType": ["text"]},
            {"name": "run_id", "dataType": ["text"]},
            {"name": "ingested_at", "dataType": ["date"]},
        ],
    }
    resp = _wv_request("POST", "/v1/schema", payload=schema)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"create schema failed: {resp.status_code} {resp.text}")


def ensure_collection(
    class_name: str,
    *,
    replace_own_collection: bool = False,
    allowed_replace_classes: list[str] | None = None,
) -> None:
    class_name = _safe_class_name(class_name)
    classes = _get_schema_classes()
    exists = class_name in classes
    if replace_own_collection and exists:
        allowed = set(allowed_replace_classes or [])
        if class_name not in allowed:
            raise RuntimeError(
                f"replace not allowed for class '{class_name}'. allowed: {sorted(allowed)}"
            )
        resp = _wv_request("DELETE", f"/v1/schema/{class_name}")
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"delete own schema failed: {resp.status_code} {resp.text}")
        exists = False
    if not exists:
        _create_schema(class_name)
    _ensure_class_properties(
        class_name,
        properties=[
            {"name": "rule_names", "dataType": ["text[]"]},
        ],
    )


def _extract_rule_names(title: str) -> list[str]:
    raw = re.findall(r"[0-9A-Za-z가-힣]+(?:규정|규칙|부칙|기준|방침|준칙|정책|지침|제정)", title or "")
    out: list[str] = []
    seen: set[str] = set()
    for r in raw:
        item = r.strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _build_embedding_text(doc: dict) -> str:
    title = (doc.get("title") or "").strip()
    summary_text = (doc.get("summary_text") or "").strip()
    kws = [k.strip() for k in (doc.get("summary_keywords") or []) if str(k).strip()]
    rule_names = _extract_rule_names(title)
    kw_text = ", ".join(kws)
    parts = [title, summary_text]
    if rule_names:
        parts.append(f"규정명: {', '.join(rule_names)}")
    if kw_text:
        parts.append(f"키워드: {kw_text}")
    return "\n".join([p for p in parts if p]).strip()


def _to_rfc3339_date(value: str) -> str:
    s = (value or "").strip()
    if not s:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return f"{s}T00:00:00Z"
    return s


def _chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _embed_texts(texts: list[str]) -> list[list[float]]:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=OPENAI_API_KEY.strip(), timeout=120.0)
    vectors: list[list[float]] = []
    for chunk in _chunks(texts, 32):
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=chunk)
        vectors.extend([d.embedding for d in resp.data])
    return vectors


def upsert_documents(
    *,
    class_name: str,
    run_id: str,
    docs: list[dict],
) -> dict:
    class_name = _safe_class_name(class_name)
    embedding_texts = [_build_embedding_text(doc) for doc in docs]
    vectors = _embed_texts(embedding_texts)

    objects = []
    for doc, vec, emb_text in zip(docs, vectors, embedding_texts):
        props = {
            "original_id": doc.get("original_id", ""),
            "title": doc.get("title", ""),
            "reg_num": int(doc.get("reg_num") or 0),
            "reg_user": doc.get("reg_user", ""),
            "reg_date": _to_rfc3339_date(str(doc.get("reg_date") or "")),
            "source_url": doc.get("source_url", ""),
            "source_text": doc.get("source_text", ""),
            "file_info_json": json.dumps(doc.get("file_info", []), ensure_ascii=False),
            "summary_text": doc.get("summary_text", ""),
            "summary_keywords": doc.get("summary_keywords", []),
            "rule_names": _extract_rule_names(str(doc.get("title", ""))),
            "embedding_text": emb_text,
            "run_id": run_id,
            "ingested_at": _utc_now(),
        }
        objects.append(
            {
                "class": class_name,
                "id": doc.get("original_id"),
                "properties": props,
                "vector": vec,
            }
        )

    resp = _wv_request("POST", "/v1/batch/objects", payload={"objects": objects})
    if resp.status_code not in (200, 202):
        raise RuntimeError(f"batch ingest failed: {resp.status_code} {resp.text}")
    data = resp.json()
    errors = 0
    error_details: list[dict] = []
    for item in data:
        result = item.get("result", {})
        err = result.get("errors")
        if err:
            errors += 1
            error_details.append(
                {
                    "id": item.get("id", ""),
                    "errors": err,
                }
            )
    return {
        "input_count": len(docs),
        "error_count": errors,
        "errors": error_details,
    }

