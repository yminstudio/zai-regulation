"""개발 중 비교 분석용 JSONL 로그 유틸."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOG_DIR

DEV_LOG_PATH = LOG_DIR / "ingest_dev.jsonl"


def write_dev_log(event: str, payload: dict) -> Path:
    """개발 이벤트를 JSONL 한 줄로 기록."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    with open(DEV_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return DEV_LOG_PATH

