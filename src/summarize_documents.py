"""스키마 시드 JSON에 1차/2차 요약 반영.

1차 요약:
- 각 첨부파일 원문을 요약해 file_info[].summary 채움

2차 요약:
- title + source_text + file_info 요약들을 통합해 summary_text 채움
"""
from __future__ import annotations

import argparse
import json
import re
import time

from src.config import DATA_DIR, SUMMARIZE_MODEL
from src.extractors import extract_text
from src.llm_client import LLMClient

INPUT_PATH = DATA_DIR / "regulation_schema_seed.json"
OUTPUT_PATH = DATA_DIR / "regulation_texts_and_summaries.json"

FIRST_STAGE_SYSTEM_PROMPT = """
당신은 기업 내부 규정 문서를 분석하여 첨부파일별 핵심 내용을 정리하는 AI입니다.

입력은 사내 게시글에 첨부된 파일 1개의 원문 텍스트입니다.
이 파일은 규정 본문, 별표, 별첨, 결재선, 참고자료, 양식, 안내문 등 다양한 형태일 수 있습니다.

목표:
- 이 파일만 읽고, 후속 검색/임베딩에 쓸 수 있는 요약을 만든다.

중요 규칙:

1. 이 단계에서는 다른 첨부파일과의 관계를 추측하지 마세요.
2. 이 파일 자체가 어떤 문서인지 중심으로 요약하세요.
3. 규정에서 등장하는 **숫자, 금액, 기간, 일자, 비율(%), 한도, 기한, 조건, 기준**은 가능한 한 그대로 유지하세요.
4. 규정에서 정의된 **절차, 승인 단계, 제출 기한, 지급 기준**이 있다면 반드시 포함하세요.
5. 별첨, 별표, 양식, 기준표가 포함된 경우 어떤 기준을 담고 있는지 설명하세요.
6. 불필요한 서론 없이 바로 규정 내용을 설명하세요.
7. 문단 수는 제한하지 않지만 **핵심 규정 기준이 빠지지 않도록 작성**하세요.

목표:
이 요약만 읽어도 해당 첨부파일이 어떤 규정이며
주요 기준과 수치가 무엇인지 이해할 수 있어야 합니다.
""".strip()

FIRST_STAGE_USER_TEMPLATE = """
다음은 게시글에 첨부된 파일 1개의 정보입니다.

게시글 제목:
{title}

첨부파일명:
{file_name}

첨부파일 원문:
{file_text}

이 첨부파일의 핵심 내용을 file_info.summary 필드에 저장할 수 있도록 한국어 요약문을 작성하세요.

작성 기준:

- 이 파일이 어떤 규정 또는 문서인지 명확히 드러나야 합니다.
- 규정명 또는 주제가 있으면 반드시 포함하세요.
- 다음 정보가 있으면 가능한 한 유지하세요
  - 금액
  - 기간
  - 일자
  - 비율(%)
  - 지급 기준
  - 승인 절차
  - 제출 기한
  - 한도 금액
- 규정에서 정의된 주요 항목(예: 출장비, 식대, 교통비 등)은 구분하여 설명하세요.
- 별첨이나 기준표가 있으면 어떤 기준을 담고 있는지 요약하세요.
- 다른 첨부파일과의 관계를 단정하지 마세요.
- 원문에 없는 사실은 추가하지 마세요.

출력 형식(반드시 아래 형식으로만 출력하세요):
SUMMARY:
(여기에 요약문 한 덩어리)
KEYWORDS:
(쉼표로 구분된 키워드. 개수 제한 없이, 검색에 유리한 단어를 충분히 포함하세요. 예: 규정명, 지급기준, 절차, 금액, 기간, 대상, 예외, 승인조건 등)
""".strip()

SECOND_STAGE_SYSTEM_PROMPT = """
당신은 사내 규정 게시글 통합 요약기입니다.
입력은 title, source_text, 첨부파일별 summary/keywords 입니다.

목표:
summary_text 필드에 들어갈 최종 통합 요약문을 작성한다.
이 요약문은 단순 축약이 아니라, 사용자가 사내 규정을 검색할 때 잘 검색되도록 핵심 규정명과 구체 주제를 풍부하게 담은 설명문이어야 합니다.

중요 규칙:
1. 게시글에 여러 첨부파일이 있더라도, 같은 규정의 본문/별표/별첨/결재선/참고자료는 하나의 맥락으로 자연스럽게 묶어 설명하세요.
2. 결재선, 참고자료, 부속 문서는 핵심 규정보다 덜 중요하므로 필요 범위에서만 짧게 반영하세요.
3. 서로 다른 규정이나 서로 다른 핵심 주제가 함께 포함되어 있다면, 각각의 규정명과 핵심 내용을 빠뜨리지 말고 모두 반영하세요.
4. 규정명, 적용 대상, 절차, 기준, 금액, 기간, 예외, 승인 조건 등 검색에 중요한 정보는 가능한 한 유지하세요.
5. 문서에 없는 내용을 추측하지 마세요.
6. 출력은 plain text 하나의 완성된 설명문으로만 작성하세요.
7. 사람이 읽어도 자연스럽고, 벡터 검색에도 잘 걸릴 수 있도록 정보량 있게 작성하세요.
8. 공지성 진행정보(예: 특정 신청일/승인완료 보고)는 규정 조항처럼 쓰지 않는다.

""".strip()

SECOND_STAGE_USER_TEMPLATE = """
다음은 사내 게시글 JSON의 일부입니다.

title:
{title}

source_text:
{source_text}

file_info:
{file_info_json}

위 정보를 바탕으로 summary_text 필드에 저장할 최종 통합 요약문을 작성하세요.

작성 기준:
- 제목, 게시글 본문, 모든 첨부파일 요약을 함께 반영하세요.
- 하나의 규정에 대한 여러 첨부파일이면 하나의 규정 맥락으로 자연스럽게 묶으세요.
- 서로 다른 규정이 함께 있으면 각 규정명과 핵심 내용을 모두 드러내세요.
- 검색에 걸릴 핵심 표현(규정명/대상/절차/기준/조건/예외/수치)을 유지하세요.
- 결과는 summary_text에 넣을 텍스트만 출력하세요.
""".strip()


def _chat(client: LLMClient, system: str, user: str) -> str:
    if not user.strip():
        return ""
    last_error: Exception | None = None
    # 대용량 첨부/혼잡 시간대에 Ollama 응답이 간헐적으로 timeout 나므로 재시도한다.
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=SUMMARIZE_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                # temperature=0.1,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            last_error = e
            if attempt >= 3:
                break
            time.sleep(2 * attempt)
    raise RuntimeError(f"summarize chat failed after retries: {last_error}") from last_error


def _parse_file_summary_response(raw: str) -> tuple[str, list[str]]:
    """1차 요약 응답에서 SUMMARY / KEYWORDS 구간 파싱. (요약문, 키워드 리스트) 반환."""
    summary = ""
    keywords: list[str] = []
    raw = (raw or "").strip()
    if not raw:
        return summary, keywords
    upper = raw.upper()
    if "KEYWORDS:" in upper:
        idx = upper.index("KEYWORDS:")
        summary = raw[:idx].strip()
        # "SUMMARY:" 접두어 제거
        if "SUMMARY:" in summary.upper():
            summary = summary.split("SUMMARY:", 1)[-1].strip()
        kw_line = raw[idx + len("KEYWORDS:"):].strip()
        keywords = [k.strip() for k in kw_line.split(",") if k.strip()]
    else:
        summary = raw
        if "SUMMARY:" in summary.upper():
            summary = summary.split("SUMMARY:", 1)[-1].strip()
    return summary, keywords


def _file_summary(
    client: LLMClient,
    *,
    title: str,
    file_name: str,
    file_text: str,
) -> tuple[str, list[str]]:
    """첨부 1개에 대해 (요약문, 키워드 리스트) 반환."""
    if not file_text.strip():
        return "", []
    user = FIRST_STAGE_USER_TEMPLATE.format(
        title=(title or "")[:2000],
        file_name=(file_name or "")[:1000],
        file_text=(file_text or "")[:12000],
    )
    raw = _chat(client, FIRST_STAGE_SYSTEM_PROMPT, user)
    return _parse_file_summary_response(raw)


def _sanitize_source_text_for_summary(source_text: str) -> str:
    """게시글 본문의 공지성/처리현황 문구를 줄여 요약 오염을 완화한다."""
    src = (source_text or "").strip()
    if not src:
        return ""
    blocked_patterns = (
        r"고용노동부\s*승인여부",
        r"\b승인여부\b",
        r"\b신청\s*:\s*",
        r"\b승인\s*:\s*",
        r"인장날인석",
        r"동의서\s*내용\s*확인",
        r"근로자\s*과반수\s*이상\s*동의\s*안내",
    )
    kept_lines: list[str] = []
    for raw_line in src.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if any(re.search(pat, line, flags=re.IGNORECASE) for pat in blocked_patterns):
            continue
        kept_lines.append(line)
    merged = "\n".join(kept_lines).strip()
    if not merged:
        return ""
    return merged[:3000]


def _doc_summary(
    client: LLMClient,
    *,
    title: str,
    source_text: str,
    file_info: list[dict],
) -> str:
    file_info_json = json.dumps(file_info, ensure_ascii=False, indent=2)
    safe_source_text = _sanitize_source_text_for_summary(source_text)
    user = SECOND_STAGE_USER_TEMPLATE.format(
        title=(title or "")[:2000],
        source_text=safe_source_text,
        file_info_json=file_info_json[:10000],
    )
    return _chat(client, SECOND_STAGE_SYSTEM_PROMPT, user)


def main() -> None:
    parser = argparse.ArgumentParser(description="첨부 1차 요약 + 게시글 2차 통합 요약")
    parser.add_argument("--limit", type=int, default=0, help="처리 문서 수 제한(0이면 전체)")
    args = parser.parse_args()

    docs = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    if args.limit and args.limit > 0:
        docs = docs[: args.limit]

    client = LLMClient(timeout=120.0)

    out_docs: list[dict] = []
    for doc in docs:
        source_text = doc.get("source_text", "") or ""
        files = doc.get("file_info", []) or []

        file_rows: list[dict] = []
        all_keywords: list[str] = []
        for f in files:
            # 입력 호환: file_name/file_summary 또는 name/summary
            name = f.get("file_name") or f.get("name") or ""
            p = DATA_DIR / name
            text = ""
            if p.exists():
                text, _method = extract_text(p)
            file_summary, file_keywords = _file_summary(
                client,
                title=doc.get("title", "") or "",
                file_name=name,
                file_text=text,
            )
            # 중복 제거 유지하며 수집 (첫 등장 순서)
            for kw in file_keywords:
                if kw and kw not in all_keywords:
                    all_keywords.append(kw)
            file_rows.append(
                {
                    "name": name,
                    "summary": file_summary,
                    "keywords": file_keywords,
                }
            )

        doc_summary = _doc_summary(
            client,
            title=doc.get("title", "") or "",
            source_text=source_text,
            file_info=file_rows,
        )

        out_docs.append(
            {
                "original_id": doc.get("original_id"),
                "title": doc.get("title"),
                "reg_num": doc.get("reg_num"),
                "reg_user": doc.get("reg_user"),
                "reg_date": doc.get("reg_date"),
                "source_url": doc.get("source_url"),
                "source_text": source_text,
                "file_info": file_rows,
                "summary_text": doc_summary,
                "summary_keywords": all_keywords,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(out_docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

