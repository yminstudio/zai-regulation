"""개발용 미리보기 파이프라인.

목록 수집(5필드 후보) -> 최신본 필터(규칙/LLM) 결과를
JSONL 로그와 JSON 파일로 저장해 비교 분석할 수 있게 한다.
"""
from __future__ import annotations

import json

from src.config import DATA_DIR
from src.dev_log import write_dev_log, DEV_LOG_PATH
from src.filter_latest import llm_refine_filter, rule_based_filter
from src.gw_downloader import GWDownloader
from src.gw_list_fetcher import fetch_board_list


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
    }
    write_dev_log("rule_filter_done", rule_payload)

    llm_result = llm_refine_filter(candidates, use_llm=True)
    llm_payload = {
        "input_count": llm_result.input_count,
        "keep_count": llm_result.keep_count,
        "regulation_list": llm_result.regulation_list,
    }
    write_dev_log("llm_filter_done", llm_payload)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    candidates_path = DATA_DIR / "board_candidates.json"
    rule_path = DATA_DIR / "board_filtered_rule.json"
    llm_path = DATA_DIR / "board_filtered_llm.json"

    with open(candidates_path, "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    with open(rule_path, "w", encoding="utf-8") as f:
        json.dump(rule_payload, f, ensure_ascii=False, indent=2)
    with open(llm_path, "w", encoding="utf-8") as f:
        json.dump(llm_payload, f, ensure_ascii=False, indent=2)

    print(f"saved: {candidates_path}")
    print(f"saved: {rule_path}")
    print(f"saved: {llm_path}")
    print(f"log:   {DEV_LOG_PATH}")


if __name__ == "__main__":
    main()

