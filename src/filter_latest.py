"""최신 규정 선별기 (Hybrid: 규칙 기반 + LLM 보조 판정)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from openai import OpenAI

from src.config import OPENAI_API_KEY

DATE_PATTERNS = [
    r"\d{4}[.\-]\d{1,2}[.\-]\d{1,2}",
    r"\d{4}\s*년\s*\d{1,2}\s*월\s*\d{1,2}\s*일",
]
NOISE_PATTERNS = [
    r"\(\s*시행일?[^\)]*\)",
    r"\(\s*개정일?[^\)]*\)",
    r"\(\s*개정[^\)]*\)",
    r"\(\s*시행[^\)]*\)",
    r"\(\s*전문\s*\)",
    r"\[\s*개정\s*\]",
    r"\[\s*신설\s*\]",
]

SYSTEM_PROMPT = """
너는 사내 규정 게시판 데이터에서 "최신 규정만 선별"하는 데이터 정리 엔진이다.

입력은 게시글 목록(JSON 배열)이며 각 항목은 다음 필드를 가진다.
- original_id
- title
- reg_num
- reg_user
- reg_date
- source_url
- source_text
- file_info
- summary_text
- summary_keywords

목표:
같은 규정으로 판단되는 게시글이 여러 개 있으면
reg_date 기준으로 가장 최신 게시글 1개만 유지한다.

규칙:
1. 같은 규정 여부는 title에서 날짜, 시행일, 개정일 등의 정보를 제거하여 판단한다.
2. 규정 이름이 동일하면 동일 규정으로 본다.
3. 동일 규정 그룹에서는 reg_date이 가장 최신인 것만 유지한다.
4. 판단이 애매하면 삭제하지 말고 유지한다. (안전 우선)
5. 반드시 JSON만 출력한다. 설명 문장은 출력하지 않는다.
""".strip()

USER_PROMPT_TEMPLATE = """
아래 게시글 목록을 규정 중복 제거 규칙에 따라 정리해줘.

요구사항:
동일 규정(regulation_key 동일) 그룹이 있으면 reg_date가 가장 최신인 1건만 keep.
나머지는 제거한다.
결과 JSON 형태:
{
  "input_count": 65,
  "keep_count": 58,
  "regulation_list": [
    {
      "regulation_key": "취업규칙",
      "keep": {
        "title": "취업규칙(2026년 1월 1일 시행)",
        "reg_date": "2025-03-14",
        "source_url": "https://gw.kggroup.co.kr/...527827"
      }
    }
  ]
}

입력(JSON):
{{BOARD_LIST_JSON}}
""".strip()


@dataclass
class FilterResult:
    input_count: int
    keep_count: int
    regulation_list: list[dict]
    keep_source_urls: list[str]
    keep_items: list[dict]


def _extract_rule_names(title: str) -> list[str]:
    raw = re.findall(r"[0-9A-Za-z가-힣]+(?:규정|규칙|부칙|기준|방침|준칙|정책|지침|제정)", title or "")
    out: list[str] = []
    seen: set[str] = set()
    for r in raw:
        key = r.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def normalize_title_to_key(title: str) -> str:
    """제목에서 날짜/개정 표기를 제거해 regulation_key 생성(단일 fallback)."""
    v = (title or "").strip()
    for p in NOISE_PATTERNS:
        v = re.sub(p, " ", v)
    for p in DATE_PATTERNS:
        v = re.sub(p, " ", v)
    v = re.sub(r"\(\s*[^)]*\s*\)", " ", v)
    v = re.sub(r"\s+", " ", v).strip(" -_")
    return (v or title.strip()).lower()


def normalize_title_to_keys(title: str) -> list[str]:
    """번들 제목이면 규정명별 N개 key를 반환한다."""
    rule_names = _extract_rule_names(title or "")
    if rule_names:
        return rule_names
    return [normalize_title_to_key(title)]


def _parse_date_safe(v: str) -> datetime:
    s = (v or "").strip()
    if not s:
        return datetime.min
    for fmt in ("%Y-%m-%d", "%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return datetime.min


def _to_keep_item(item: dict) -> dict:
    """후속 파이프라인 연계를 위한 keep 메타(스키마 seed 호환) 생성."""
    return {
        "original_id": item.get("original_id", ""),
        "title": item.get("title", ""),
        "reg_num": item.get("reg_num", 0),
        "reg_user": item.get("reg_user", ""),
        "reg_date": item.get("reg_date", ""),
        "source_url": item.get("source_url", ""),
        "source_text": item.get("source_text", ""),
        "file_info": item.get("file_info", []),
        "summary_text": item.get("summary_text", ""),
        "summary_keywords": item.get("summary_keywords", []),
    }


def rule_based_filter(board_list: list[dict]) -> FilterResult:
    """규칙 기반 1차 최신본 선별.

    번들 문서는 규정명별 N개 그룹에 동시에 매핑한다.
    """
    groups: dict[str, list[dict]] = {}
    for item in board_list:
        keys = normalize_title_to_keys(item.get("title", ""))
        for key in keys:
            groups.setdefault(key, []).append(item)

    regulation_list: list[dict] = []
    keep_urls: list[str] = []
    keep_items: list[dict] = []
    seen_keep_urls: set[str] = set()
    for key, items in groups.items():
        if len(items) == 1:
            keep = items[0]
        else:
            keep = max(items, key=lambda x: _parse_date_safe(x.get("reg_date", "")))

        regulation_list.append(
            {
                "regulation_key": key,
                "keep": {
                    "title": keep.get("title", ""),
                    "reg_date": keep.get("reg_date", ""),
                    "source_url": keep.get("source_url", ""),
                },
            }
        )
        keep_url = keep.get("source_url", "")
        if keep_url and keep_url not in seen_keep_urls:
            seen_keep_urls.add(keep_url)
            keep_urls.append(keep_url)
            keep_items.append(_to_keep_item(keep))

    return FilterResult(
        input_count=len(board_list),
        keep_count=len(keep_urls),
        regulation_list=regulation_list,
        keep_source_urls=keep_urls,
        keep_items=keep_items,
    )


def llm_refine_filter(board_list: list[dict], *, use_llm: bool = True) -> FilterResult:
    """Hybrid 2차 선별: 규칙 기반 결과를 기본으로 하고 필요 시 LLM 사용."""
    baseline = rule_based_filter(board_list)
    if not use_llm:
        return baseline

    if not OPENAI_API_KEY:
        return baseline

    client = OpenAI(api_key=OPENAI_API_KEY, timeout=120.0)
    user_prompt = USER_PROMPT_TEMPLATE.replace(
        "{{BOARD_LIST_JSON}}",
        json.dumps(board_list, ensure_ascii=False, indent=2),
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5.4",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        parsed = json.loads(raw)
    except Exception:
        return baseline

    reg_list = parsed.get("regulation_list", [])
    by_url = {b.get("source_url", ""): b for b in board_list}
    keep_urls = [
        (r.get("keep") or {}).get("source_url", "")
        for r in reg_list
        if (r.get("keep") or {}).get("source_url")
    ]
    keep_items = []
    for url in keep_urls:
        base = by_url.get(url, {})
        if base:
            keep_items.append(_to_keep_item(base))

    return FilterResult(
        input_count=parsed.get("input_count", len(board_list)),
        keep_count=parsed.get("keep_count", len(keep_urls)),
        regulation_list=reg_list,
        keep_source_urls=keep_urls,
        keep_items=keep_items,
    )
