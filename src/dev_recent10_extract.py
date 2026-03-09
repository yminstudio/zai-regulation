"""최근 10개 게시글 첨부파일 추출 미리보기.

수행 단계:
1) 최근 게시글 10개 URL 수집
2) 각 게시글 첨부파일 다운로드
3) 파일별 텍스트 추출(OCR 포함)
4) 단계별 JSONL 로그 기록
"""
from __future__ import annotations

import json
import argparse

from src.config import DATA_DIR
from src.dev_log import write_dev_log, DEV_LOG_PATH
from src.extractors import extract_text
from src.gw_downloader import GWDownloader
from src.gw_list_fetcher import fetch_board_list


def main() -> None:
    parser = argparse.ArgumentParser(description="최근 N개 게시글 첨부 추출 테스트")
    parser.add_argument("--limit", type=int, default=5, help="처리할 게시글 수 (기본 5)")
    args = parser.parse_args()

    gw = GWDownloader()
    board = [x.to_dict() for x in fetch_board_list(gw)]
    recent10 = board[: max(1, args.limit)]

    write_dev_log(
        "recent10_selected",
        {
            "limit": args.limit,
            "count": len(recent10),
            "items": [
                {
                    "title": x.get("title"),
                    "reg_date": x.get("reg_date"),
                    "source_url": x.get("source_url"),
                    "reg_num": x.get("reg_num"),
                    "reg_user": x.get("reg_user"),
                }
                for x in recent10
            ],
        },
    )

    extracted_rows: list[dict] = []

    for idx, post in enumerate(recent10, 1):
        source_url = post.get("source_url", "")
        title = post.get("title", "")
        write_dev_log(
            "post_start",
            {"index": idx, "title": title, "source_url": source_url},
        )

        attachments = gw.get_attachments(source_url)
        write_dev_log(
            "attachments_listed",
            {
                "source_url": source_url,
                "count": len(attachments),
                "file_names": [a.file_name for a in attachments],
            },
        )

        for att in attachments:
            try:
                file_path = gw.download_file(att)
                write_dev_log(
                    "attachment_downloaded",
                    {"source_url": source_url, "file_name": att.file_name, "path": str(file_path)},
                )
            except Exception as e:
                write_dev_log(
                    "attachment_download_failed",
                    {"source_url": source_url, "file_name": att.file_name, "error": str(e)},
                )
                continue

            try:
                text, method = extract_text(file_path)
                row = {
                    "source_url": source_url,
                    "title": title,
                    "reg_num": post.get("reg_num"),
                    "reg_user": post.get("reg_user"),
                    "reg_date": post.get("reg_date"),
                    "file_name": att.file_name,
                    "file_path": str(file_path),
                    "extract_method": method,
                    "text_length": len(text),
                    "text_preview": text[:500],
                }
                extracted_rows.append(row)
                write_dev_log(
                    "attachment_text_extracted",
                    {
                        "source_url": source_url,
                        "file_name": att.file_name,
                        "extract_method": method,
                        "text_length": len(text),
                    },
                )
            except Exception as e:
                write_dev_log(
                    "attachment_extract_failed",
                    {"source_url": source_url, "file_name": att.file_name, "error": str(e)},
                )

        write_dev_log("post_done", {"index": idx, "source_url": source_url})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "recent10_attachment_texts.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extracted_rows, f, ensure_ascii=False, indent=2)

    write_dev_log(
        "recent10_done",
        {
            "limit": args.limit,
            "posts_processed": len(recent10),
            "files_processed": len(extracted_rows),
            "output_json": str(out_path),
        },
    )
    print(f"saved: {out_path}")
    print(f"log:   {DEV_LOG_PATH}")


if __name__ == "__main__":
    main()

