"""개발용 미리보기 파이프라인.

목록 수집(5필드 후보) -> 최신본 필터(규칙/LLM) 결과를
JSONL 로그와 JSON 파일로 저장해 비교 분석할 수 있게 한다.
"""
from __future__ import annotations

import json
from uuid import uuid5, NAMESPACE_URL

from src.config import DATA_DIR
from src.dev_log import write_dev_log, DEV_LOG_PATH
from src.filter_latest import llm_refine_filter, rule_based_filter
from src.gw_downloader import GWDownloader
from src.gw_list_fetcher import fetch_board_list


def to_full_schema_docs(candidates: list[dict], keep_source_urls: list[str]) -> list[dict]:
    """필터 이후 결과를 최종 스키마 형태로 매핑."""
    by_url = {item.get("source_url", ""): item for item in candidates}
    docs: list[dict] = []

    for url in keep_source_urls:
        base = by_url.get(url)
        if not base:
            continue

        docs.append(
            {
                "original_id": str(uuid5(NAMESPACE_URL, url)),
                "title": base.get("title", ""),
                "reg_num": base.get("reg_num", 0),
                "reg_user": base.get("reg_user", ""),
                "reg_date": base.get("reg_date", ""),
                "source_url": base.get("source_url", ""),
                "source_text": "",
                "file_info": [],
                "summary_text": "",
                "summary_keywords": [],
            }
        )

    return docs


def main() -> None:
    gw = GWDownloader()
    candidates = [x.to_dict() for x in fetch_board_list(gw)]
    write_dev_log(
        "board_candidates_fetched",
        {
            "count": len(candidates),
            "sample": candidates[:3],
        },
    )

    rule_result = rule_based_filter(candidates)
    rule_payload = {
        "input_count": rule_result.input_count,
        "keep_count": rule_result.keep_count,
        "regulation_list": rule_result.regulation_list,
        "documents": to_full_schema_docs(candidates, rule_result.keep_source_urls),
    }
    write_dev_log("rule_filter_done", rule_payload)

    llm_result = llm_refine_filter(candidates, use_llm=True)
    llm_payload = {
        "input_count": llm_result.input_count,
        "keep_count": llm_result.keep_count,
        "regulation_list": llm_result.regulation_list,
        "documents": to_full_schema_docs(candidates, llm_result.keep_source_urls),
    }
    write_dev_log("llm_filter_done", llm_payload)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates_path = DATA_DIR / "board_candidates.json"
    rule_path = DATA_DIR / "board_filtered_rule.json"
    llm_path = DATA_DIR / "board_filtered_llm.json"
    rule_docs_path = DATA_DIR / "board_filtered_rule_documents.json"
    llm_docs_path = DATA_DIR / "board_filtered_llm_documents.json"

    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    with open(rule_path, "w", encoding="utf-8") as f:
        json.dump(rule_payload, f, ensure_ascii=False, indent=2)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_payload, f, ensure_ascii=False, indent=2)
    with open(rule_docs_path, "w", encoding="utf-8") as f:
        json.dump(rule_payload["documents"], f, ensure_ascii=False, indent=2)
    with open(llm_docs_path, "w", encoding="utf-8") as f:
        json.dump(llm_payload["documents"], f, ensure_ascii=False, indent=2)

    print(f"saved: {candidates_path}")
    print(f"saved: {rule_path}")
    print(f"saved: {llm_path}")
    print(f"saved: {rule_docs_path}")
    print(f"saved: {llm_docs_path}")
    print(f"log:   {DEV_LOG_PATH}")


if __name__ == "__main__":
    main()

