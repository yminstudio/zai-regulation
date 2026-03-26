"""프로젝트 전역 설정 모듈.

설정 우선순위(높음 -> 낮음):
1) OS 환경변수
2) .env / .venv/.env
3) config/<env>.yaml
4) config/base.yaml
5) 코드 기본값
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_ENV = "dev"


def _load_dotenv_files() -> None:
    # 기존 하위호환(.venv/.env) + 표준(.env) 모두 지원한다.
    for env_path in (PROJECT_ROOT / ".venv" / ".env", PROJECT_ROOT / ".env"):
        if env_path.exists():
            load_dotenv(env_path)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _resolve_config() -> tuple[str, Path, dict[str, Any]]:
    config_env = (os.getenv("APP_ENV") or DEFAULT_ENV).strip() or DEFAULT_ENV
    config_file = os.getenv("CONFIG_FILE", "").strip()
    if config_file:
        file_path = Path(config_file)
        if not file_path.is_absolute():
            file_path = PROJECT_ROOT / file_path
        return config_env, file_path, _read_yaml(file_path)

    base_path = CONFIG_DIR / "base.yaml"
    env_path = CONFIG_DIR / f"{config_env}.yaml"
    merged = {}
    merged.update(_read_yaml(base_path))
    merged.update(_read_yaml(env_path))
    return config_env, env_path, merged


def _get(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if raw is not None and raw != "":
        return raw
    return _YAML_CONFIG.get(name, default)


def _get_path(name: str, default_relative: str) -> Path:
    raw = str(_get(name, default_relative)).strip()
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _get_bool(name: str, default: bool) -> bool:
    raw = str(_get(name, "true" if default else "false")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


_load_dotenv_files()
APP_ENV, CONFIG_PATH, _YAML_CONFIG = _resolve_config()

# --- LLM (Ollama) ---
# 모델/엔드포인트는 코드가 아니라 환경변수(.env)로만 변경한다.
OLLAMA_BASE_URL: str = str(_get("OLLAMA_BASE_URL", "http://localhost:11434")).strip()
OLLAMA_API_KEY: str = str(_get("OLLAMA_API_KEY", "")).strip()
SUMMARIZE_MODEL: str = str(_get("SUMMARIZE_MODEL", "gpt-oss-20b-128k:latest")).strip()
ANSWER_MODEL: str = str(_get("ANSWER_MODEL", "gpt-oss-20b-128k:latest")).strip()

# --- Embedding ---
# 백엔드는 openai 또는 local_sentence_transformers 중 선택한다.
EMBEDDING_BACKEND: str = str(_get("EMBEDDING_BACKEND", "local_sentence_transformers")).strip().lower()
OPENAI_API_KEY: str = str(_get("OPENAI_API_KEY", "")).strip()
OPENAI_EMBEDDING_MODEL: str = str(
    _get("OPENAI_EMBEDDING_MODEL", _get("EMBEDDING_MODEL", "text-embedding-3-small"))
).strip()
LOCAL_EMBEDDING_MODEL: str = str(
    _get("LOCAL_EMBEDDING_MODEL", "BAAI/bge-m3")
).strip()
LOCAL_EMBEDDING_DEVICE: str = str(_get("LOCAL_EMBEDDING_DEVICE", "")).strip()
LOCAL_EMBEDDING_NORMALIZE: bool = _get_bool("LOCAL_EMBEDDING_NORMALIZE", True)

# --- Groupware ---
GW_USER_ID: str = str(_get("GW_USER_ID", "")).strip()
GW_PASSWORD: str = str(_get("GW_PASSWORD", "")).strip()
GW_BASE_URL: str = str(_get("GW_BASE_URL", "https://gw.kggroup.co.kr")).strip()
GW_BOARD_FOLDER_ID: str = str(_get("GW_BOARD_FOLDER_ID", "8233")).strip()

# --- Weaviate ---
WEAVIATE_URL: str = str(_get("WEAVIATE_URL", "http://localhost:8080")).strip()
PROJECT_WEAVIATE_CLASS: str = str(_get("PROJECT_WEAVIATE_CLASS", "ZaiRegulation")).strip()
PROJECT_WEAVIATE_TEST_CLASS: str = str(_get("PROJECT_WEAVIATE_TEST_CLASS", "ZaiRegulation_test")).strip()
PROJECT_WEAVIATE_DB_CLASS: str = str(_get("PROJECT_WEAVIATE_DB_CLASS", "ZaiRegulation_db")).strip()

# --- Paths ---
DATA_DIR: Path = _get_path("DATA_DIR", "data")
LOG_DIR: Path = _get_path("LOG_DIR", "logs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
