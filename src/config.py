"""프로젝트 전역 설정 모듈.

.venv/.env 파일에서 기밀 값(API 키, 로그인 정보 등)을 로딩하고
프로젝트 전체에서 참조할 수 있도록 제공합니다.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VENV_ENV_PATH = PROJECT_ROOT / ".venv" / ".env"

load_dotenv(VENV_ENV_PATH)

# --- OpenAI ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
SUMMARIZE_MODEL: str = os.getenv("SUMMARIZE_MODEL", "gpt-5.4-nano")
ANSWER_MODEL: str = os.getenv("ANSWER_MODEL", "gpt-5.4-nano")

# --- Groupware ---
GW_USER_ID: str = os.getenv("GW_USER_ID", "")
GW_PASSWORD: str = os.getenv("GW_PASSWORD", "")
GW_BASE_URL: str = os.getenv("GW_BASE_URL", "https://gw.kggroup.co.kr")
GW_BOARD_FOLDER_ID: str = os.getenv("GW_BOARD_FOLDER_ID", "8233")

# --- Weaviate ---
WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")
PROJECT_WEAVIATE_CLASS: str = os.getenv("PROJECT_WEAVIATE_CLASS", "ZaiRegulation")
PROJECT_WEAVIATE_TEST_CLASS: str = os.getenv("PROJECT_WEAVIATE_TEST_CLASS", "ZaiRegulation_test")

# --- Paths ---
DATA_DIR: Path = PROJECT_ROOT / "data"
LOG_DIR: Path = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
