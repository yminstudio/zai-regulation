"""게시글/첨부 수집 후 스키마 시드 JSON 생성.

수행 단계:
1) 게시글 목록 수집
2) 각 게시글 본문/첨부 수집
3) 첨부 텍스트 추출(OCR 포함)
4) 요약 전 단계 스키마 JSON 저장
"""
from __future__ import annotations

import argparse
import json
from uuid import NAMESPACE_URL, uuid5

from src.config import DATA_DIR
from src.extractors import extract_text
from src.gw_downloader import GWDownloader
from src.gw_list_fetcher import fetch_board_list
from src.log_utils import LOG_PATH, write_log

OUTPUT_PATH = DATA_DIR / "regulation_schema_seed.json"
EXCLUDED_ATTACHMENT_KEYWORDS = ("신규조문", "신구조문")


def _is_excluded_attachment(file_name: str) -> bool:
    name = (file_name or "").strip().lower()
    if not name:
        return False
    return any(keyword.lower() in name for keyword in EXCLUDED_ATTACHMENT_KEYWORDS)


def main() -> None:
    parser = argparse.ArgumentParser(description="게시글/첨부 수집")
    parser.add_argument("--limit", type=int, default=3, help="처리할 게시글 수 (기본 3)")
    args = parser.parse_args()

    gw = GWDownloader()
    board = [x.to_dict() for x in fetch_board_list(gw)]
    selected = board[: max(1, args.limit)]

    write_log(
        "posts_selected",
        {
            "limit": args.limit,
            "count": len(selected),
            "items": [
                {
                    "title": x.get("title"),
                    "reg_date": x.get("reg_date"),
                    "source_url": x.get("source_url"),
                    "reg_num": x.get("reg_num"),
                    "reg_user": x.get("reg_user"),
                }
                for x in selected
            ],
        },
    )

    schema_docs: list[dict] = []

    for idx, post in enumerate(selected, 1):
        source_url = post.get("source_url", "")
        title = post.get("title", "")
        source_text = gw.fetch_source_text(source_url)
        write_log(
            "post_start",
            {
                "index": idx,
                "title": title,
                "source_url": source_url,
                "source_text_length": len(source_text or ""),
            },
        )

        attachments = gw.get_attachments(source_url)
        kept_attachments = [a for a in attachments if not _is_excluded_attachment(a.file_name)]
        skipped_file_names = [a.file_name for a in attachments if _is_excluded_attachment(a.file_name)]
        write_log(
            "attachments_listed",
            {
                "source_url": source_url,
                "count": len(kept_attachments),
                "file_names": [a.file_name for a in kept_attachments],
                "skipped_count": len(skipped_file_names),
                "skipped_file_names": skipped_file_names,
            },
        )

        file_info: list[dict] = []
        for att in kept_attachments:
            try:
                file_path = gw.download_file(att)
                write_log(
                    "attachment_downloaded",
                    {"source_url": source_url, "file_name": att.file_name, "path": str(file_path)},
                )
            except Exception as e:
                write_log(
                    "attachment_download_failed",
                    {"source_url": source_url, "file_name": att.file_name, "error": str(e)},
                )
                continue

            try:
                text, method = extract_text(file_path)
                file_info.append({"file_name": att.file_name, "file_summary": ""})
                write_log(
                    "attachment_text_extracted",
                    {
                        "source_url": source_url,
                        "file_name": att.file_name,
                        "extract_method": method,
                        "text_length": len(text),
                        "text_preview": text[:200],
                    },
                )
            except Exception as e:
                write_log(
                    "attachment_extract_failed",
                    {"source_url": source_url, "file_name": att.file_name, "error": str(e)},
                )

        write_log("post_done", {"index": idx, "source_url": source_url})
        schema_docs.append(
            {
                "original_id": str(uuid5(NAMESPACE_URL, source_url)),
                "title": title,
                "reg_num": post.get("reg_num"),
                "reg_user": post.get("reg_user"),
                "reg_date": post.get("reg_date"),
                "source_url": source_url,
                "source_text": source_text,
                "file_info": file_info,
                "summary_text": "",
                "summary_keywords": [],
            }
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema_docs, f, ensure_ascii=False, indent=2)

    write_log(
        "collection_done",
        {
            "limit": args.limit,
            "posts_processed": len(selected),
            "files_processed": sum(len(x.get("file_info", [])) for x in schema_docs),
            "output_json": str(OUTPUT_PATH),
        },
    )
    print(f"saved: {OUTPUT_PATH}")
    print(f"log:   {LOG_PATH}")


if __name__ == "__main__":
    main()

