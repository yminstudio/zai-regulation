"""같은 요약 결과 JSON을 임베딩만 바꿔 재적재하는 유틸."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import (
    EMBEDDING_BACKEND,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    PROJECT_WEAVIATE_DB_CLASS,
    PROJECT_WEAVIATE_TEST_CLASS,
    PROJECT_WEAVIATE_CLASS,
)
from src.weaviate_ingest import ensure_collection, upsert_documents


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="요약 JSON 재임베딩 적재")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="07_final_for_ingest.json 경로",
    )
    parser.add_argument(
        "--weaviate-class",
        type=str,
        default=PROJECT_WEAVIATE_DB_CLASS,
        help="적재 대상 Weaviate class",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="지정하지 않으면 현재 시각 기반 run_id 사용",
    )
    parser.add_argument(
        "--replace-own-collection",
        action="store_true",
        help="대상 class 삭제 후 재생성",
    )
    parser.add_argument(
        "--embedding-backend",
        type=str,
        default=EMBEDDING_BACKEND,
        help="임베딩 백엔드 (openai | local_sentence_transformers)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="",
        help=(
            "임베딩 모델명 override. 비우면 backend 기본값 사용 "
            f"(openai={OPENAI_EMBEDDING_MODEL}, local={LOCAL_EMBEDDING_MODEL})"
        ),
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    docs = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(docs, list):
        raise RuntimeError("input json must be a list of documents")

    ensure_collection(
        args.weaviate_class,
        replace_own_collection=bool(args.replace_own_collection),
        allowed_replace_classes=[
            PROJECT_WEAVIATE_CLASS,
            PROJECT_WEAVIATE_TEST_CLASS,
            PROJECT_WEAVIATE_DB_CLASS,
            args.weaviate_class,
        ],
    )
    report = upsert_documents(
        class_name=args.weaviate_class,
        run_id=args.run_id.strip() or _new_run_id(),
        docs=docs,
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model or None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
