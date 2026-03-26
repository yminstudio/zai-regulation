"""Ollama API client with OpenAI-like response objects."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import requests

from src.config import OLLAMA_API_KEY, OLLAMA_BASE_URL


@dataclass
class _ChatCompletions:
    client: "LLMClient"

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> SimpleNamespace:
        payload: dict[str, Any] = {
            "model": model,
            "messages": self.client.normalize_messages(messages),
            "stream": False,
        }
        if temperature is not None:
            payload["options"] = {"temperature": float(temperature)}
        if (response_format or {}).get("type") == "json_object":
            payload["format"] = "json"

        body = self.client._post("/api/chat", payload)
        content = str(((body.get("message") or {}).get("content")) or "").strip()
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                )
            ]
        )


@dataclass
class _Embeddings:
    client: "LLMClient"

    def create(self, *, model: str, input: str | list[str]) -> SimpleNamespace:  # noqa: A002
        if isinstance(input, str):
            inputs = [input]
        else:
            inputs = [str(x or "") for x in input]

        try:
            body = self.client._post("/api/embed", {"model": model, "input": inputs})
            vectors = body.get("embeddings") or []
            if len(vectors) != len(inputs):
                raise RuntimeError("invalid embedding response length")
        except Exception:
            vectors = []
            for text in inputs:
                legacy = self.client._post("/api/embeddings", {"model": model, "prompt": text})
                vec = legacy.get("embedding") or []
                vectors.append(vec)

        return SimpleNamespace(data=[SimpleNamespace(embedding=v) for v in vectors])


class _Chat:
    def __init__(self, client: "LLMClient") -> None:
        self.completions = _ChatCompletions(client)


class LLMClient:
    """Minimal OpenAI-compatible facade backed by Ollama."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")
        self.api_key = (api_key or OLLAMA_API_KEY).strip()
        self.timeout = float(timeout)
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)

    @staticmethod
    def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        normalized: list[dict[str, str]] = []
        for m in messages or []:
            role = str(m.get("role", "user") or "user")
            content = m.get("content", "")
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(str(part.get("text", "")))
                text = "\n".join(parts).strip()
            else:
                text = str(content or "")
            normalized.append({"role": role, "content": text})
        return normalized

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}{path}",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("invalid ollama response")
        if data.get("error"):
            raise RuntimeError(str(data.get("error")))
        return data
