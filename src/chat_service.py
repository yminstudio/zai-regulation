"""사규 검색/답변 서비스."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime

from openai import OpenAI

from src.config import ANSWER_MODEL, OPENAI_API_KEY
from src.weaviate_search import SearchHit, SearchResult, search_with_fallback

SEARCH_FETCH_LIMIT = 30
LOW_SCORE_THRESHOLD = 0.45
RELATED_LINK_MAX_CANDIDATES = 5

SYSTEM_PROMPT = """
당신은 KG그룹의 사내규정(사규) 검색 챗봇입니다.
아래에 검색된 사규 정보(제목, 요약, 일부 본문)가 제공됩니다.
반드시 제공된 정보 범위 내에서만 판단하고 답변하세요.

[답변 원칙]

1. 검색 결과에 질문과 직접적으로 관련된 규정이 2개 이상이면 핵심 내용을 비교 요약하고 관련 링크를 모두 제시하세요.
2. 검색 결과에 질문과 완전히 동일한 문구는 없더라도, 제목/요약/본문 맥락상 질문과 높은 관련성이 있다면
   "직접적인 명시 규정은 없으나, 다음 규정이 관련 가능성이 있습니다."라고 안내하고 관련 링크를 2개 이상 제시하세요.
3. 추측성 세부 내용은 작성하지 마세요. 반드시 검색 결과에 포함된 정보만 근거로 판단하세요.
4. 숫자(금액), 퍼센트(%), 기한(일수), 조건은 그대로 유지하세요.
5. 일반 원칙과 예외 규정이 함께 존재하면 반드시 함께 안내하세요.
6. 반드시 한국어로 작성하세요.
7. 질문과 무관한 규정은 포함하지 마세요.
8. 링크는 반드시 마크다운 하이퍼링크 형식으로 작성하세요.
9. 동일 문서에서 여러 항목을 답변할 경우 링크는 한 번만 정리하세요.
10. 확정되지 않은 경우에는 "관련 가능성이 있는 규정"이라는 표현을 사용하세요.
11. "관련 사규를 찾지 못했습니다."라는 문장은 검색 결과 전체가 질문과 명확히 무관할 때에만 사용하세요.
12. 관련 문서가 여러 개면 최신 `reg_date` 문서를 먼저 안내하되, 다른 관련 문서도 생략하지 마세요.

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

LATEST_REWRITE_SYSTEM_PROMPT = """
당신은 KG그룹 사규 챗봇입니다.
기존 답변에 과거 버전 규정 링크가 섞인 경우가 있어, 아래에 제공되는 최신 규정 근거로 답변을 다시 작성해야 합니다.

[작성 규칙]
1. 반드시 제공된 최신 규정 근거만 사용하세요.
2. 숫자(금액/퍼센트/기한/조건)는 근거에 있는 값만 사용하세요.
3. 기존 답변의 구조를 최대한 유지하되, 과거 기준 내용은 최신 기준으로 교체하세요.
4. 링크는 반드시 최신 규정 링크를 사용하세요.
5. 근거가 부족하면 단정하지 말고 "관련 가능성"으로 표현하세요.
6. 출력은 반드시 JSON object 한 개:
{
  "answer": "사용자에게 보여줄 최종 답변"
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


def _build_expanded_query_by_intent(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q

    # LLM 없이 키워드 기반 의도 분기
    amount_terms = ("얼마", "금액", "한도", "%", "퍼센트", "수당", "지급", "비용", "경비")
    procedure_terms = ("어떻게", "절차", "신청", "결재", "승인", "제출", "처리", "방법")
    criteria_terms = ("기준", "요건", "대상", "조건", "자격", "해당", "가능", "언제")

    if any(t in q for t in amount_terms):
        suffix = "규정 기준 금액 지급 한도"
    elif any(t in q for t in procedure_terms):
        suffix = "규정 절차 신청 결재 제출 방법"
    elif any(t in q for t in criteria_terms):
        suffix = "규정 기준 요건 대상 조건"
    else:
        suffix = "사규 규정 기준"
    return f"{q} {suffix}".strip()


def _build_context_from_hits(hits: list[SearchHit]) -> str:
    if not hits:
        return "검색 결과 없음"
    rows = []
    for i, h in enumerate(hits, 1):
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

    expanded = _build_expanded_query_by_intent(query)
    retried = search_with_fallback(expanded, limit=SEARCH_FETCH_LIMIT)
    if retried.top_score > result.top_score:
        return retried
    return result


def choose_search_query(messages: list[dict], current_question: str) -> QueryDecision:
    # 검색 질의는 항상 마지막 유저 질문 1개를 사용한다.
    raw_result = _search_with_low_score_fallback(current_question)
    reranked_result = _rerank_hits_by_last_query(raw_result, current_question)
    # 최종 rank는 리랭킹 score 기준을 우선한다.
    final_result = SearchResult(
        query=reranked_result.query,
        hits=reranked_result.hits,
        mode=reranked_result.mode,
    )
    return QueryDecision(
        chosen_query=current_question,
        score_a=final_result.top_score,
        score_b=0.0,
        tie_break_reason="last_user_query_rerank_score_first",
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
    parsed = _safe_json_loads(raw)
    standalone_query = str(parsed.get("standalone_query", decision.chosen_query))
    base_answer = str(parsed.get("answer", "")).strip() or raw
    latest_aligned_answer = _rewrite_answer_with_latest_sources_if_needed(
        client=client,
        answer=base_answer,
        current_question=current_question,
        decision=decision,
    )
    answer = _dedupe_markdown_links(latest_aligned_answer.strip())
    answer = _append_related_links_if_needed(
        answer,
        decision=decision,
        current_question=current_question,
    )
    return {
        "standalone_query": standalone_query,
        "answer": answer,
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


def _extract_existing_link_urls(answer: str) -> set[str]:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    return {m.group(1).strip() for m in link_re.finditer(answer or "") if m.group(1).strip()}


def _extract_existing_links(answer: str) -> list[tuple[str, str]]:
    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    out: list[tuple[str, str]] = []
    for m in link_re.finditer(answer or ""):
        label = m.group(1).strip()
        url = m.group(2).strip()
        if label and url:
            out.append((label, url))
    return out


def _extract_rule_names(text: str) -> set[str]:
    # 예: "경비지급규정", "복리후생규정", "위임전결규정", "취업규칙" 등
    raw = re.findall(r"[0-9A-Za-z가-힣]+(?:규정|규칙|기준|준칙)", text or "")
    return {r.lower() for r in raw if r}


def _safe_json_loads(text: str) -> dict:
    try:
        parsed = json.loads((text or "").strip())
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def _select_latest_hit_for_rule_names(hits: list[SearchHit], rule_names: set[str]) -> SearchHit | None:
    if not rule_names:
        return None
    matched = []
    for hit in hits:
        if _extract_rule_names(hit.title).intersection(rule_names):
            matched.append(hit)
    if not matched:
        return None
    matched.sort(key=lambda h: _parse_reg_date(h.reg_date) or datetime.min, reverse=True)
    return matched[0]


def _build_latest_rewrite_targets(answer: str, result: SearchResult) -> list[tuple[SearchHit, SearchHit]]:
    by_url = {h.source_url: h for h in result.hits if h.source_url}
    replacements: list[tuple[SearchHit, SearchHit]] = []
    seen: set[tuple[str, str]] = set()
    for label, url in _extract_existing_links(answer):
        old_hit = by_url.get(url)
        if not old_hit:
            continue
        rule_names = _extract_rule_names(label) or _extract_rule_names(old_hit.title)
        latest_hit = _select_latest_hit_for_rule_names(result.hits, rule_names)
        if not latest_hit:
            continue
        if latest_hit.source_url == old_hit.source_url:
            continue
        key = (old_hit.source_url, latest_hit.source_url)
        if key in seen:
            continue
        seen.add(key)
        replacements.append((old_hit, latest_hit))
    return replacements


def _collect_latest_hits_for_answer_rules(answer: str, result: SearchResult) -> list[SearchHit]:
    """답변에 등장한 규정군별 최신 hit를 수집한다."""
    rule_names: set[str] = set()
    for label, _ in _extract_existing_links(answer):
        rule_names.update(_extract_rule_names(label))

    latest_hits: list[SearchHit] = []
    seen_urls: set[str] = set()
    for rule_name in sorted(rule_names):
        latest = _select_latest_hit_for_rule_names(result.hits, {rule_name})
        if latest and latest.source_url and latest.source_url not in seen_urls:
            latest_hits.append(latest)
            seen_urls.add(latest.source_url)
    return latest_hits


def _force_link_urls_to_latest(answer: str, latest_hits: list[SearchHit]) -> str:
    """재생성 결과에 과거 URL이 남아도 규정명 기준으로 최신 URL로 강제 동기화한다."""
    if not answer.strip() or not latest_hits:
        return answer
    latest_by_rule: dict[str, tuple[str, datetime]] = {}
    for hit in latest_hits:
        dt = _parse_reg_date(hit.reg_date) or datetime.min
        for rn in _extract_rule_names(hit.title):
            prev = latest_by_rule.get(rn)
            if prev is None or dt > prev[1]:
                latest_by_rule[rn] = (hit.source_url, dt)

    link_re = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    out = answer
    for m in reversed(list(link_re.finditer(answer))):
        label = m.group(1).strip()
        current_url = m.group(2).strip()
        target_url = ""
        for rn in _extract_rule_names(label):
            if rn in latest_by_rule:
                target_url = latest_by_rule[rn][0]
                break
        if target_url and target_url != current_url:
            replacement = f"[{label}]({target_url})"
            out = out[: m.start()] + replacement + out[m.end() :]
    return out


def _rewrite_answer_with_latest_sources_if_needed(
    *,
    client: OpenAI,
    answer: str,
    current_question: str,
    decision: QueryDecision,
) -> str:
    replacements = _build_latest_rewrite_targets(answer, decision.result)
    latest_hits = _collect_latest_hits_for_answer_rules(answer, decision.result)
    # "규정군 최신본 치환 대상"이 있거나, 답변에 규정군이 식별되면 2차 생성 수행
    if not replacements and not latest_hits:
        return answer
    if not latest_hits:
        return answer

    rewrite_prompt = (
        f"[현재 사용자 질문]\n{current_question}\n\n"
        f"[기존 답변]\n{answer}\n\n"
        f"[과거→최신 교체 대상]\n"
        + "\n".join(
            f"- old: {old.title} ({old.reg_date}) -> new: {new.title} ({new.reg_date})"
            for old, new in replacements
        )
        + "\n\n"
        f"[최신 근거 문서]\n{_build_context_from_hits(latest_hits)}"
    )
    resp = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=[
            {"role": "system", "content": LATEST_REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": rewrite_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    raw = (resp.choices[0].message.content or "").strip()
    parsed = _safe_json_loads(raw)
    rewritten = str(parsed.get("answer", "")).strip() or answer
    return _force_link_urls_to_latest(rewritten, latest_hits)


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    src = re.sub(r"\s+", "", (text or "").lower())
    if len(src) < n:
        return {src} if src else set()
    return {src[i : i + n] for i in range(len(src) - n + 1)}


def _title_group_key(title: str) -> str:
    base = re.sub(r"\([^)]*\)", "", title or "")
    base = re.sub(r"[^0-9A-Za-z가-힣]+", "", base).lower()
    return base


def _select_related_candidates(result: SearchResult, query: str, *, max_candidates: int = RELATED_LINK_MAX_CANDIDATES) -> list:
    query_terms = _query_terms(query)
    if not result.hits:
        return []
    if not query_terms:
        return result.hits[:max_candidates]

    query_ngrams = _char_ngrams(" ".join(sorted(query_terms)), n=2)

    scored_candidates: list[tuple[float, datetime, object]] = []
    group_count: dict[str, int] = {}
    for hit in result.hits:
        title_kw_text = " ".join(
            [
                hit.title or "",
                " ".join(str(k) for k in (hit.summary_keywords or [])),
            ]
        ).lower()
        summary_text = (hit.summary_text or "").lower()
        doc_text = f"{title_kw_text} {summary_text}".strip()

        # 1) 정확 토큰 포함 비율
        exact_hits = sum(1 for term in query_terms if term and term in doc_text)
        exact_ratio = exact_hits / max(1, len(query_terms))
        title_kw_hits = sum(1 for term in query_terms if term and term in title_kw_text)
        summary_hits = sum(1 for term in query_terms if term and term in summary_text)
        # 2) 문자 bi-gram 유사도(복합어/띄어쓰기 변형 대응)
        doc_ngrams = _char_ngrams(doc_text, n=2)
        ngram_overlap = len(query_ngrams.intersection(doc_ngrams))
        ngram_ratio = ngram_overlap / max(1, len(query_ngrams))

        # n-gram 단독 통과는 무관 문서 유입 가능성이 커서 배제한다.
        passes_text_gate = (
            title_kw_hits > 0
            or (summary_hits > 0 and exact_ratio >= 0.5)
            or (exact_ratio >= 0.7 and ngram_ratio >= 0.4)
        )
        if not passes_text_gate:
            continue

        group_key = _title_group_key(hit.title) or hit.title
        if group_count.get(group_key, 0) >= 2:
            continue
        group_count[group_key] = group_count.get(group_key, 0) + 1
        relevance = (0.5 if title_kw_hits > 0 else 0.0) + (0.3 * exact_ratio) + (0.2 * ngram_ratio)
        dt = _parse_reg_date(hit.reg_date) or datetime.min
        scored_candidates.append((relevance, dt, hit))

    scored_candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [hit for _, _, hit in scored_candidates[:max_candidates]]


def _append_related_links_if_needed(answer: str, *, decision: QueryDecision, current_question: str) -> str:
    existing_links = _extract_existing_links(answer)
    existing_urls = {url for _, url in existing_links}

    # 1순위: 이미 답변에 등장한 규정명과 같은 규정군(최신/과거 버전) 확장
    anchor_rule_names: set[str] = set()
    for label, _ in existing_links:
        anchor_rule_names.update(_extract_rule_names(label))

    family_candidates: list = []
    if anchor_rule_names:
        for hit in decision.result.hits:
            if not hit.source_url:
                continue
            hit_rule_names = _extract_rule_names(hit.title)
            if hit_rule_names.intersection(anchor_rule_names):
                family_candidates.append(hit)
            if len(family_candidates) >= RELATED_LINK_MAX_CANDIDATES:
                break

    # 2순위: 일반 관련 후보
    if len(family_candidates) >= 2:
        candidates = family_candidates
    else:
        candidates = _select_related_candidates(
            decision.result,
            current_question,
            max_candidates=RELATED_LINK_MAX_CANDIDATES,
        )
    if len(candidates) < 2:
        return answer

    missing = [h for h in candidates if h.source_url and h.source_url not in existing_urls]
    if not missing:
        return answer

    latest_dt = None
    for h in candidates:
        dt = _parse_reg_date(h.reg_date)
        if dt is not None and (latest_dt is None or dt > latest_dt):
            latest_dt = dt

    lines = [answer.strip(), "", "추가로 확인할 관련 규정:"]
    for h in missing:
        label = h.title
        dt = _parse_reg_date(h.reg_date)
        if latest_dt is not None and dt is not None and dt == latest_dt:
            label = f"{label} [최신]"
        lines.append(f"- [관련링크 : {label}]({h.source_url})")
    return "\n".join(lines).strip()


def append_sq_comment(answer: str, standalone_query: str) -> str:
    sq = (standalone_query or "").strip().replace("-->", "-- >")
    return f"{(answer or '').strip()}\n<!-- sq:{sq} -->"

