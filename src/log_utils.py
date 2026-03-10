"""파이프라인 JSONL 로그 유틸."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import LOG_DIR

LOG_PATH = LOG_DIR / "ingest_pipeline.jsonl"


def write_log(event: str, payload: dict, *, log_path: Path | None = None) -> Path:
    """파이프라인 이벤트를 JSONL 한 줄로 기록."""
    target = log_path or LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload,
    }
    with open(target, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return target

