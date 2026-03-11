"""사규 검색/답변 서비스."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime

from openai import OpenAI

from src.config import ANSWER_MODEL, OPENAI_API_KEY
from src.weaviate_search import SearchResult, search_with_fallback

SEARCH_FETCH_LIMIT = 30
LOW_SCORE_THRESHOLD = 0.45

SYSTEM_PROMPT = """
당신은 KG그룹의 사내규정(사규) 검색 챗봇입니다.
아래에 검색된 사규 정보(제목, 요약, 일부 본문)가 제공됩니다.
반드시 제공된 정보 범위 내에서만 판단하고 답변하세요.

[답변 원칙]

1. 검색 결과에 질문에 직접적으로 명시된 규정이 존재하면 해당 내용만으로 답변하세요.
2. 검색 결과에 질문과 완전히 동일한 문구는 없더라도, 제목/요약/본문 맥락상 질문과 높은 관련성이 있다면
   "직접적인 명시 규정은 없으나, 다음 규정이 관련 가능성이 있습니다."라고 안내하고 링크를 제공하세요.
3. 추측성 세부 내용은 작성하지 마세요. 반드시 검색 결과에 포함된 정보만 근거로 판단하세요.
4. 숫자(금액), 퍼센트(%), 기한(일수), 조건은 그대로 유지하세요.
5. 일반 원칙과 예외 규정이 함께 존재하면 반드시 함께 안내하세요.
6. 반드시 한국어로 작성하세요.
7. 질문과 무관한 규정은 포함하지 마세요.
8. 링크는 반드시 마크다운 하이퍼링크 형식으로 작성하세요.
9. 동일 문서에서 여러 항목을 답변할 경우 링크는 한 번만 정리하세요.
10. 확정되지 않은 경우에는 "관련 가능성이 있는 규정"이라는 표현을 사용하세요.
11. "관련 사규를 찾지 못했습니다."라는 문장은 검색 결과 전체가 질문과 명확히 무관할 때에만 사용하세요.

[응답 형식]
① 질문에 직접적인 규정이 존재하는 경우:
- (핵심 내용 요약)
- (필요 시 조건/예외 안내)
- [관련링크 : 규정명](링크URL)

② 직접 규정은 없으나, 제목/요약상 높은 관련성이 있는 경우:
- 질문에 대해 직접적으로 명시된 조항은 확인되지 않았습니다.
- 그러나 다음 규정이 관련 가능성이 있는 것으로 보입니다.
- (요약 기반 관련 내용 정리)
- [관련링크 : 규정명](링크URL)

③ 검색 결과가 질문과 명확히 무관한 경우:
- 관련 사규를 찾지 못했습니다.

추가 제약:
출력은 반드시 JSON object 한 개:
{
  "standalone_query": "다음 턴 검색에 사용할 독립 질의",
  "answer": "사용자에게 보여줄 답변"
}
""".strip()


@dataclass
class QueryDecision:
    chosen_query: str
    score_a: float
    score_b: float
    tie_break_reason: str
    result: SearchResult


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
        return "\n".join(out).strip()
    return ""


def extract_current_user_question(messages: list[dict]) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            return _content_to_text(m.get("content", "")).strip()
    return ""


def _parse_reg_date(value: str) -> datetime | None:
    text = (value or "").strip()
    if not text:
        return None
    candidates = [text, text.replace(".", "-"), text.replace("/", "-")]
    formats = ("%Y-%m-%d", "%Y-%m", "%Y%m%d", "%Y.%m.%d", "%Y/%m/%d", "%Y")
    for item in candidates:
        for fmt in formats:
            try:
                return datetime.strptime(item, fmt)
            except ValueError:
                continue
    digits = re.sub(r"\D", "", text)
    if len(digits) >= 8:
        try:
            return datetime.strptime(digits[:8], "%Y%m%d")
        except ValueError:
            return None
    return None


def _query_terms(query: str) -> set[str]:
    return {t.lower() for t in re.findall(r"[0-9A-Za-z가-힣]+", (query or "").strip()) if t}


def _keyword_match_boost(query_terms: set[str], title: str, keywords: list[str]) -> float:
    if not query_terms:
        return 0.0
    title_text = (title or "").lower()
    keywords_text = " ".join(keywords or []).lower()
    title_hits = sum(1 for term in query_terms if term and term in title_text)
    keyword_hits = sum(1 for term in query_terms if term and term in keywords_text)
    return min(title_hits * 0.12, 0.36) + min(keyword_hits * 0.08, 0.24)


def _rerank_hits_by_last_query(result: SearchResult, query: str) -> SearchResult:
    hits = result.hits or []
    if not hits:
        return result

    query_terms = _query_terms(query)
    parsed_dates = [_parse_reg_date(h.reg_date) for h in hits]
    ordinals = [d.toordinal() for d in parsed_dates if d is not None]
    min_ord = min(ordinals) if ordinals else None
    max_ord = max(ordinals) if ordinals else None

    def recency_boost(index: int) -> float:
        d = parsed_dates[index]
        if d is None or min_ord is None or max_ord is None:
            return 0.0
        if max_ord == min_ord:
            return 0.06
        normalized = (d.toordinal() - min_ord) / (max_ord - min_ord)
        return normalized * 0.12

    scored: list[tuple[float, int]] = []
    for idx, hit in enumerate(hits):
        boost = _keyword_match_boost(query_terms, hit.title, hit.summary_keywords) + recency_boost(idx)
        final_score = float(hit.score) + boost
        scored.append((final_score, idx))

    scored.sort(key=lambda x: x[0], reverse=True)
    reranked_hits = [replace(hits[idx], score=score) for score, idx in scored]
    return SearchResult(query=result.query, hits=reranked_hits, mode=result.mode)


def _sort_hits_by_reg_date_desc(hits: list) -> list:
    # Stable sort keeps prior rerank order when dates are equal.
    return sorted(hits, key=lambda h: _parse_reg_date(h.reg_date) or datetime.min, reverse=True)


def _build_context(result: SearchResult) -> str:
    if not result.hits:
        return "검색 결과 없음"
    rows = []
    for i, h in enumerate(result.hits, 1):
        rows.append(
            f"[{i}] title={h.title}\n"
            f"source_url={h.source_url}\n"
            f"reg_date={h.reg_date}, reg_user={h.reg_user}\n"
            f"summary={h.summary_text}\n"
            f"keywords={', '.join(h.summary_keywords)}"
        )
    return "\n\n".join(rows)


def _search_with_low_score_fallback(query: str) -> SearchResult:
    result = search_with_fallback(query, limit=SEARCH_FETCH_LIMIT)
    if result.top_score >= LOW_SCORE_THRESHOLD:
        return result

    expanded = f"{query} 사규 규정 기준 금액 조건".strip()
    retried = search_with_fallback(expanded, limit=SEARCH_FETCH_LIMIT)
    if retried.top_score > result.top_score:
        return retried
    return result


def choose_search_query(messages: list[dict], current_question: str) -> QueryDecision:
    # 검색 질의는 항상 마지막 유저 질문 1개를 사용한다.
    raw_result = _search_with_low_score_fallback(current_question)
    reranked_result = _rerank_hits_by_last_query(raw_result, current_question)
    date_sorted_hits = _sort_hits_by_reg_date_desc(reranked_result.hits)
    final_result = SearchResult(query=reranked_result.query, hits=date_sorted_hits, mode=reranked_result.mode)
    return QueryDecision(
        chosen_query=current_question,
        score_a=final_result.top_score,
        score_b=0.0,
        tie_break_reason="last_user_query_rerank_then_reg_date_desc",
        result=final_result,
    )


def _messages_to_history(messages: list[dict], *, max_user_turns: int = 4) -> str:
    user_only = [m for m in (messages or []) if m.get("role") == "user"]
    trimmed = user_only[-max_user_turns:]
    out: list[str] = []
    for m in trimmed:
        content = _content_to_text(m.get("content", ""))
        if content:
            out.append(f"user: {content}")
    return "\n".join(out)


def generate_answer_json(*, messages: list[dict], current_question: str, decision: QueryDecision) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    client = OpenAI(api_key=OPENAI_API_KEY.strip(), timeout=120.0)

    context_text = _build_context(decision.result)
    history_text = _messages_to_history(messages)
    user_prompt = (
        f"[검색에 사용된 질의]\n{decision.chosen_query}\n\n"
        f"[검색 점수]\nscore_A={decision.score_a:.4f}, score_B={decision.score_b:.4f}\n"
        f"decision={decision.tie_break_reason}\n\n"
        f"[검색 컨텍스트]\n{context_text}\n\n"
        f"[최근 대화(최대 4턴)]\n{history_text}\n\n"
        f"[현재 사용자 질문]\n{current_question}"
    )
    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = json.loads(raw)
    except Exception:
        parsed = {"standalone_query": decision.chosen_query, "answer": raw}
    return {
        "standalone_query": str(parsed.get("standalone_query", decision.chosen_query)),
        "answer": _dedupe_markdown_links(str(parsed.get("answer", "")).strip()),
    }


def _dedupe_markdown_links(answer: str) -> str:
    lines = (answer or "").splitlines()
    seen: set[tuple[str, str]] = set()
    deduped: list[str] = []
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    for line in lines:
        kept_line = line
        matches = list(link_re.finditer(line))
        if matches:
            filtered_parts: list[str] = []
            last_end = 0
            for m in matches:
                label = m.group(1).strip()
                url = m.group(2).strip()
                key = (label, url)
                filtered_parts.append(line[last_end : m.start()])
                if key not in seen:
                    filtered_parts.append(m.group(0))
                    seen.add(key)
                last_end = m.end()
            filtered_parts.append(line[last_end:])
            kept_line = "".join(filtered_parts).rstrip()
        cleaned = kept_line.strip()
        if cleaned and cleaned not in {"-", "*", "- ", "* "}:
            deduped.append(kept_line)
    return "\n".join(deduped).strip()


def append_sq_comment(answer: str, standalone_query: str) -> str:
    sq = (standalone_query or "").strip().replace("-->", "-- >")
    return f"{(answer or '').strip()}\n<!-- sq:{sq} -->"

