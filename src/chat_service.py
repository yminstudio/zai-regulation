"""사규 검색/답변 서비스."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from datetime import datetime

from src.config import ANSWER_MODEL
from src.llm_client import LLMClient
from src.weaviate_search import SearchHit, SearchResult, search_with_fallback

SEARCH_FETCH_LIMIT = 30
ANSWER_CONTEXT_LIMIT = 10
LOW_SCORE_THRESHOLD = 0.45
RELATED_LINK_MAX_CANDIDATES = 5
RELATED_LINK_APPEND_LIMIT = 3
INTENT_NON_REGULATION_THRESHOLD = 0.8
NON_REGULATION_GUIDE_LINE = "사내 규정/복리후생/휴가/경비 관련 질문을 주시면 도와드릴게요."
TOPIC_LOCK_MAX_USER_TURNS = 3

SYSTEM_PROMPT = """
당신은 KG제로인의 사내규정(사규) 링크 추천 챗봇입니다.
목표는 장문 답변이 아니라 "정확한 관련 링크 선택"입니다.
아래 candidate_docs는 검색 시스템이 뽑은 후보이며, id/title/reg_date/summary/keywords/url이 포함됩니다.

[절대 규칙]
1. candidate_docs에 없는 문서를 만들지 마세요.
2. URL은 절대 생성/수정/추측하지 마세요. (id 선택만 수행)
3. 출력은 반드시 JSON object 한 개만 반환하세요.
4. selected_ids에는 반드시 candidate_docs의 id만 넣으세요.
5. 관련 문서가 있으면 2~3개를 우선 선택하세요. (최대 3개)
6. 동일 규정군(개정본/유사 제목)이 겹치면 최신 reg_date 문서를 우선 선택하세요.
7. 확정적 근거가 부족하면 brief에 "관련 가능성"을 명시하세요.
8. 질문과 명확히 무관하면 no_match=true로 반환하세요.
9. brief는 "의도 추출 안내" 1문장만 작성하세요. (정답형 설명/해설 금지)
10. 사용자의 질문에 대한 직접 답변(정책 해석, 결론 제시)을 작성하지 마세요.

[출력 스키마]
{
  "standalone_query": "다음 턴 검색에 사용할 독립 질의",
  "no_match": false,
  "brief": "질문에서 추출한 의도(예: 출산, 지원, 대상)와 관련된 규정을 안내드립니다.",
  "selected_ids": ["doc_1", "doc_3"],
  "selection_reasons": {
    "doc_1": "짧은 근거",
    "doc_3": "짧은 근거"
  }
}
""".strip()


# 얘는 이제 쓰지 않음 이유는 이미 위에 있는 규칙에 의해 처리되기 때문임. (INTENT_KEYWORD_SYSTEM_PROMPT)
INTENT_SYSTEM_PROMPT = """
당신은 KG제로인 사규 챗봇의 선행 라우터입니다.
입력 질문을 아래 2개 라벨 중 하나로 분류하세요.

- regulation: 사내 규정/규칙/기준/복리후생/휴가/경비/지급/절차/대상/조건 등 사규 문의
- non_regulation: 일상 대화, 잡담, 사규와 무관한 일반 질문

중요 규칙:
1) non_regulation은 "사규와 무관함이 명확한 경우"에만 선택하세요.
2) 질문에 규정/복지/휴가/경조/지급/대상/조건/절차 등 업무 맥락이 조금이라도 있으면 regulation으로 분류하세요.
3) 출력은 JSON object 한 개만 반환하세요.
4) confidence는 0.0~1.0 사이 실수로 반환하세요.

출력 스키마:
{
  "intent": "regulation | non_regulation",
  "confidence": 0.0,
  "reason": "짧은 근거"
}
""".strip()

KEYWORD_EXTRACT_SYSTEM_PROMPT = """
당신은 KG제로인 사규 검색용 키워드 추출기입니다.
사용자 질문에서 "규정 검색 정확도"에 가장 중요한 핵심어만 고르세요.

규칙:
1) 질문의 주변 맥락어(예: 우리, 회사, 나는, 이거, 좀)는 제외합니다.
2) 규정 탐색에 직접 쓰일 실질 키워드만 2~5개 반환합니다.
3) 한 단어 위주로 간결하게 반환합니다. (필요시 2어절까지 허용)
4) 가족/친족 표현은 아래 기준으로 표준화 키워드를 우선 포함합니다.
   - 외삼촌/이모/고모/고모부/외숙모/외숙부 -> 인척, 경조, 사망, 경조휴가, 경조금
   - 조부모/외조부모 -> 조부모, 경조, 사망
   - 배우자 부모/장인/장모/시부/시모 -> 배우자 가족, 인척, 경조
   - 형제자매 배우자 -> 인척, 경조
5) 이벤트 표현은 아래 기준으로 정규화합니다.
   - 돌아가셨다/상/장례 -> 사망, 경조휴가, 경조금, 신청기한, 증빙
   - 출산/애기 낳다 -> 출산, 복리후생, 지원금, 휴가, 신청기한
   - 결혼/혼인 -> 결혼, 경조, 휴가, 지원금
6) 애매하면 복리후생/휴가/경비/지급/대상/조건/절차 관련 핵심어를 우선합니다.
7) 정책 해석/결론은 하지 말고 키워드 추출만 수행합니다.
8) 출력은 JSON object 한 개만 반환합니다.

출력 스키마:
{
  "keywords": ["키워드1", "키워드2"],
  "reason": "짧은 근거"
}
""".strip()

INTENT_KEYWORD_SYSTEM_PROMPT = """
당신은 KG제로인 사규 챗봇의 선행 라우터이자 검색 키워드 추출기입니다.
입력 질문을 보고 아래 두 작업을 한 번에 수행하세요.

작업1) intent 분류
- regulation: 사내 규정/규칙/기준/복리후생/휴가/경비/지급/절차/대상/조건 등 사규 문의
- non_regulation: 일상 대화, 잡담, 사규와 무관한 일반 질문
- 질문에 업무/복지/경조/휴가/지급/대상/조건/절차 맥락이 조금이라도 있으면 regulation으로 분류하세요.

작업2) 검색 키워드 추출
- 규정 탐색에 직접 쓰일 실질 키워드만 2~5개
- 주변 맥락어(예: 우리, 회사, 나는, 이거, 좀)는 제외
- 한 단어 위주, 필요시 2어절 허용
- 정책 해석/결론은 하지 말고 키워드만 추출

출력은 반드시 JSON object 한 개만 반환하세요.
{
  "intent": "regulation | non_regulation",
  "confidence": 0.0,
  "intent_reason": "짧은 근거",
  "keywords": ["키워드1", "키워드2"],
  "keyword_reason": "짧은 근거"
}
""".strip()

FINAL_ANSWER_WRITER_SYSTEM_PROMPT = """
당신은 KG제로인 사규 안내 챗봇입니다.
역할은 "선택된 문서 링크를 바탕으로, 사용자 질문에 맞는 자연스러운 안내문"을 작성하는 것입니다.

중요:
1) 정책 해석을 단정하지 말고, 문서에 무엇이 담겨 있는지 안내 중심으로 작성하세요.
2) 어조는 부드럽고 자연스럽게 작성하세요. (기계적/딱딱한 문구 금지)
3) "질문에서 추출한 의도", "관련 가능성" 같은 메타 문구는 사용하지 마세요.
4) 각 문서 설명은 2~3문장으로, 해당 문서에 포함된 기준/절차/대상/조건을 짧게 요약하세요.
5) 가족관계(외숙모/인척/직계/혼인계 등) 판단은 summary에 명시된 경우에만 작성하세요.
6) 질문에 대한 결론(“가능/불가”, “지급됨/안됨”)을 단정하지 마세요.
7) 숫자(기간/금액/비율/기한)는 summary에 있는 값만 그대로 사용하세요.   
8) URL, 문서 id는 절대 생성하거나 수정하지 마세요. 설명문 텍스트만 작성하세요.
9) "doc_1", "doc_2" 같은 내부 id를 사용자 노출 문장에 절대 쓰지 마세요.
10) "인사팀에 문의", "담당부서 확인"처럼 다음 행동을 지시하는 문구는 작성하지 마세요.
11) 출력은 반드시 JSON object 한 개만 반환하세요.
12) 규정에 없는 절차/조직/사례를 절대 생성하지 마라
13) 명시된 내용만 사용하라 없으면 "명시되지 않음"이라고 답하라.

출력 스키마:
{
  "intro": "전체 안내 1~2문장",
  "item_descriptions": {
    "doc_1": "문서 설명",
    "doc_2": "문서 설명"
  },
  "closing": "마무리 1문장"
}
""".strip()

NON_REGULATION_SYSTEM_PROMPT = f"""
당신은 KG제로인 사규 챗봇입니다.
사규와 무관한 질문에도 친절하게 답하되, 챗봇의 역할 경계를 유지하세요.
규칙:
1) 답변은 한국어로, 자연스럽고 이해하기 쉽게 2~5문장으로 작성합니다.
2) 일반 상식/일상 질문은 간단히 도움되는 답변을 제공합니다.
3) 불확실한 내용은 단정하지 말고, 가능한 범위에서만 안내합니다.
4) 법률/노무/세무/의학 등 전문 판단이 필요한 질문은 일반 안내만 제공합니다.
5) 사규 관련 질문으로 이어질 수 있으면 한 문장으로 부드럽게 유도합니다.
6) 답변 마지막에 반드시 아래 문장을 그대로 추가합니다.
   "{NON_REGULATION_GUIDE_LINE}"
""".strip()

LATEST_REWRITE_SYSTEM_PROMPT = """
당신은 KG제로인 사규 챗봇입니다.
기존 답변에 과거 버전 규정 링크가 섞인 경우가 있어, 아래에 제공되는 최신 규정 근거로 답변을 다시 작성해야 합니다.

[작성 규칙]
1. 반드시 제공된 최신 규정 근거만 사용하세요.
2. 숫자(금액/퍼센트/기한/조건)는 근거에 있는 값만 사용하세요.
3. 기존 답변의 구조를 최대한 유지하되, 과거 기준 내용은 최신 기준으로 교체하세요.
4. 링크는 반드시 최신 규정 링크를 사용하세요.
5. 근거가 부족하면 단정하지 말고 "관련 가능성"으로 표현하세요.
6. 링크를 출력할 때는 반드시 아래 마크다운 형식을 사용하세요.
   - [관련링크 : 규정명](링크URL)
7. 출력은 반드시 JSON object 한 개:
{
  "answer": [사용자에게 보여줄 최종 답변]
}
""".strip()


@dataclass
class QueryDecision:
    chosen_query: str
    normalized_query: str
    score_a: float
    score_b: float
    tie_break_reason: str
    result: SearchResult
    extracted_keywords: list[str]
    search_queries: list[str]
    raw_extracted_keywords: list[str]
    keyword_source: str
    keyword_reason: str
    rerank_query: str
    debug: dict = field(default_factory=dict)


@dataclass
class IntentDecision:
    intent: str
    confidence: float
    reason: str


@dataclass
class KeywordExtractionDecision:
    raw_keywords: list[str]
    cleaned_keywords: list[str]
    source: str
    reason: str


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


def _clamp_confidence(value: object) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    if num < 0.0:
        return 0.0
    if num > 1.0:
        return 1.0
    return num


def classify_intent(question: str) -> IntentDecision:
    intent_decision, _keyword_decision = classify_intent_and_keywords(question)
    return intent_decision


def classify_intent_and_keywords(
    question: str,
    *,
    max_keywords: int = 5,
) -> tuple[IntentDecision, KeywordExtractionDecision]:
    q = (question or "").strip()
    if not q:
        return (
            IntentDecision(intent="regulation", confidence=0.0, reason="empty_question"),
            KeywordExtractionDecision(
                raw_keywords=[],
                cleaned_keywords=[],
                source="empty_question",
                reason="empty_question",
            ),
        )
    if _is_meta_task_prompt(q):
        # OpenWebUI 보조 태스크 프롬프트는 사규 검색 파이프라인에서 제외한다.
        return (
            IntentDecision(intent="non_regulation", confidence=1.0, reason="meta_task_prompt_guard"),
            KeywordExtractionDecision(
                raw_keywords=[],
                cleaned_keywords=[],
                source="meta_task_prompt_guard",
                reason="meta_task_prompt_guard",
            ),
        )
    client = LLMClient(timeout=60.0)
    llm_reason = "no_reason"
    llm_raw_keywords: list[str] = []
    try:
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": INTENT_KEYWORD_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _safe_json_loads(raw)
        raw_intent = str(parsed.get("intent", "regulation")).strip().lower()
        confidence = _clamp_confidence(parsed.get("confidence", 0.0))
        intent_reason = str(parsed.get("intent_reason", parsed.get("reason", ""))).strip() or "no_reason"
        llm_reason = str(parsed.get("keyword_reason", "")).strip() or "no_reason"
        keywords_raw = parsed.get("keywords", [])
        if isinstance(keywords_raw, list):
            llm_raw_keywords = [str(x or "").strip() for x in keywords_raw if str(x or "").strip()]
        cleaned = _sanitize_keywords(llm_raw_keywords, max_keywords=max_keywords)

        intent = "non_regulation" if raw_intent == "non_regulation" else "regulation"
        if cleaned:
            return (
                IntentDecision(intent=intent, confidence=confidence, reason=intent_reason),
                KeywordExtractionDecision(
                    raw_keywords=llm_raw_keywords,
                    cleaned_keywords=cleaned,
                    source="llm_combined",
                    reason=llm_reason,
                ),
            )

        heuristic = _heuristic_regulation_keywords(q, max_keywords=max_keywords)
        return (
            IntentDecision(intent=intent, confidence=confidence, reason=intent_reason),
            KeywordExtractionDecision(
                raw_keywords=llm_raw_keywords or heuristic,
                cleaned_keywords=heuristic,
                source="heuristic_llm_fallback",
                reason=llm_reason,
            ),
        )
    except Exception as e:
        llm_reason = f"llm_error:{type(e).__name__}"
        heuristic = _heuristic_regulation_keywords(q, max_keywords=max_keywords)
        return (
            IntentDecision(intent="regulation", confidence=0.0, reason=llm_reason),
            KeywordExtractionDecision(
                raw_keywords=heuristic,
                cleaned_keywords=heuristic,
                source="heuristic_llm_fallback",
                reason=llm_reason,
            ),
        )


def generate_non_regulation_answer(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return NON_REGULATION_GUIDE_LINE
    client = LLMClient(timeout=60.0)
    try:
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": NON_REGULATION_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            temperature=0.3,
        )
        answer = (resp.choices[0].message.content or "").strip()
    except Exception:
        answer = ""

    if not answer:
        return NON_REGULATION_GUIDE_LINE
    if NON_REGULATION_GUIDE_LINE not in answer:
        answer = f"{answer}\n{NON_REGULATION_GUIDE_LINE}".strip()
    return answer


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


def _strip_embedded_chat_history(text: str) -> str:
    src = (text or "").strip()
    if not src:
        return ""
    # 메타 프롬프트에 내장된 chat_history 블록은 현재 질의로 간주하지 않는다.
    src = re.sub(r"<chat_history>.*?</chat_history>", " ", src, flags=re.IGNORECASE | re.DOTALL)
    src = re.sub(r"\s+", " ", src).strip()
    return src


def _is_meta_task_prompt(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False
    markers = (
        "### task:",
        "<chat_history>",
        "json format:",
        "\"follow_ups\"",
        "\"title\"",
        "\"tags\"",
        "summarizing the chat history",
        "based on the chat history",
    )
    return any(marker in q for marker in markers)


def is_meta_task_prompt(question: str) -> bool:
    """서버 라우팅에서 사용할 공개 헬퍼."""
    return _is_meta_task_prompt(question)


def _sanitize_keywords(values: list[str], *, max_keywords: int = 5) -> list[str]:
    stopwords = {
        "나",
        "내",
        "저",
        "우리",
        "회사",
        "사내",
        "질문",
        "문의",
        "관련",
        "내용",
        "이거",
        "이것",
        "저거",
        "그거",
        "가능",
        "여부",
        "지금",
        "오늘",
        "이번",
        "그냥",
        "둘",
        "다",
    }
    out: list[str] = []
    seen: set[str] = set()
    for raw in values or []:
        text = str(raw or "").strip()
        if not text:
            continue
        norm = " ".join(re.findall(r"[0-9A-Za-z가-힣]+", text))
        if not norm:
            continue
        key = norm.lower()
        if len(key) <= 1:
            continue
        if key in stopwords:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
        if len(out) >= max(1, max_keywords):
            break
    return out


def _heuristic_regulation_keywords(question: str, *, max_keywords: int = 5) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []
    terms = re.findall(r"[0-9A-Za-z가-힣]{2,}", q)
    if not terms:
        return []

    priority_seeds = (
        "결혼",
        "경조",
        "가족",
        "부부",
        "출산",
        "육아",
        "휴가",
        "복리",
        "복리후생",
        "경비",
        "비용",
        "지급",
        "수당",
        "지원",
        "한도",
        "금액",
        "대상",
        "조건",
        "요건",
        "신청",
        "절차",
        "승인",
    )
    first_pos: dict[str, int] = {}
    for idx, t in enumerate(terms):
        low = t.lower()
        if low not in first_pos:
            first_pos[low] = idx

    scored: list[tuple[int, int, str]] = []
    for token in {t.lower() for t in terms}:
        score = 1
        if any(seed in token for seed in priority_seeds):
            score += 3
        if token.endswith(("비", "금", "료", "당")):
            score += 1
        scored.append((score, -first_pos.get(token, 0), token))
    scored.sort(reverse=True)

    ordered = [token for _, _, token in scored]
    return _sanitize_keywords(ordered, max_keywords=max_keywords)


def _augment_event_keywords(question: str, extracted_keywords: list[str], *, max_keywords: int = 7) -> list[str]:
    """질문 표면어 기반으로 경조/친족 이벤트 키워드를 보강한다."""
    q = (question or "").strip().lower()
    if not q:
        return _sanitize_keywords(extracted_keywords, max_keywords=max_keywords)

    out = list(_sanitize_keywords(extracted_keywords, max_keywords=max_keywords))

    def _append(items: list[str]) -> None:
        for item in items:
            if item not in out:
                out.append(item)

    bereavement_terms = ("돌아가셨", "상", "사망", "장례")
    kinship_terms = (
        "외숙모",
        "외숙부",
        "외삼촌",
        "이모",
        "고모",
        "고모부",
        "조부모",
        "외조부모",
        "장인",
        "장모",
        "시부",
        "시모",
        "형제자매",
        "인척",
        "친척",
    )

    if any(t in q for t in kinship_terms):
        _append(["인척", "경조", "복리후생"])
    if any(t in q for t in bereavement_terms):
        _append(["사망", "경조휴가", "경조금", "신청기한", "증빙"])
    if "경조" in q:
        _append(["경조휴가", "경조금"])

    return _sanitize_keywords(out, max_keywords=max_keywords)


def extract_regulation_keywords(question: str, *, max_keywords: int = 5) -> KeywordExtractionDecision:
    _intent_decision, keyword_decision = classify_intent_and_keywords(question, max_keywords=max_keywords)
    return keyword_decision


def _build_search_queries(current_question: str, extracted_keywords: list[str]) -> list[str]:
    q = _strip_embedded_chat_history(current_question)
    if not q:
        return []
    out = [q]
    keywords = _sanitize_keywords(extracted_keywords, max_keywords=5)
    if not keywords:
        return out

    joined = " ".join(keywords)
    out.append(joined)
    for kw in keywords[:3]:
        out.append(f"{kw} 규정")
    unique: list[str] = []
    seen: set[str] = set()
    for item in out:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(item.strip())
    return unique


def _merge_multi_query_results(results: list[SearchResult]) -> SearchResult:
    if not results:
        return SearchResult(query="", hits=[], mode="hybrid_multi")

    merged: dict[str, SearchHit] = {}
    seen_count: dict[str, int] = {}
    best_rank: dict[str, int] = {}
    query_weights = [1.0, 0.97, 0.95, 0.93, 0.91]

    for ridx, result in enumerate(results):
        q_weight = query_weights[ridx] if ridx < len(query_weights) else 0.9
        for rank, hit in enumerate(result.hits):
            key = hit.source_url or hit.original_id or f"{hit.title}|{hit.reg_date}"
            seen_count[key] = seen_count.get(key, 0) + 1
            best_rank[key] = min(best_rank.get(key, rank), rank)
            weighted_score = float(hit.score) * q_weight
            existing = merged.get(key)
            if existing is None or weighted_score > float(existing.score):
                merged[key] = replace(hit, score=weighted_score)

    final_hits: list[SearchHit] = []
    for key, hit in merged.items():
        occurrences = seen_count.get(key, 1)
        rank_bonus = max(0.0, 0.08 - (best_rank.get(key, 0) * 0.004))
        final_score = float(hit.score) + min((occurrences - 1) * 0.06, 0.18) + rank_bonus
        final_hits.append(replace(hit, score=final_score))

    final_hits.sort(key=lambda h: float(h.score), reverse=True)
    merged_query = " || ".join(r.query for r in results if r.query)
    return SearchResult(query=merged_query, hits=final_hits, mode="hybrid_multi")


def _keyword_match_boost(query_terms: set[str], title: str, keywords: list[str], summary_text: str) -> float:
    if not query_terms:
        return 0.0
    title_text = (title or "").lower()
    keywords_text = " ".join(keywords or []).lower()
    summary = (summary_text or "").lower()
    title_hits = sum(1 for term in query_terms if term and term in title_text)
    keyword_hits = sum(1 for term in query_terms if term and term in keywords_text)
    summary_hits = sum(1 for term in query_terms if term and term in summary)
    return min(title_hits * 0.14, 0.42) + min(keyword_hits * 0.1, 0.3) + min(summary_hits * 0.06, 0.18)


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
        boost = _keyword_match_boost(
            query_terms,
            hit.title,
            hit.summary_keywords,
            hit.summary_text,
        ) + recency_boost(idx)
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


def _build_candidate_docs_for_llm(hits: list[SearchHit], *, limit: int = 10) -> list[dict]:
    candidates: list[dict] = []
    for idx, hit in enumerate(hits[: max(1, limit)], 1):
        candidates.append(
            {
                "id": f"doc_{idx}",
                "title": hit.title,
                "reg_date": hit.reg_date,
                "summary": hit.summary_text,
                "keywords": hit.summary_keywords,
                "url": hit.source_url,
            }
        )
    return candidates


def _build_candidate_map(candidates: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for item in candidates:
        key = str(item.get("id", "")).strip()
        if not key:
            continue
        out[key] = item
    return out


def _candidate_rule_names(doc: dict) -> set[str]:
    title = str(doc.get("title", "")).strip()
    return _extract_rule_names(title)


def _promote_selected_ids_to_latest(
    selected_ids: list[str],
    candidates: list[dict],
    candidate_map: dict[str, dict],
) -> list[str]:
    """선택된 링크를 동일 규정군 최신본으로 승격한다."""
    if not selected_ids or not candidates:
        return selected_ids

    latest_by_rule: dict[str, tuple[str, datetime]] = {}
    for doc in candidates:
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            continue
        dt = _parse_reg_date(str(doc.get("reg_date", ""))) or datetime.min
        for rule_name in _candidate_rule_names(doc):
            prev = latest_by_rule.get(rule_name)
            if prev is None or dt > prev[1]:
                latest_by_rule[rule_name] = (doc_id, dt)

    promoted: list[str] = []
    seen: set[str] = set()
    for sid in selected_ids:
        current_id = str(sid or "").strip()
        if not current_id or current_id not in candidate_map:
            continue
        current_doc = candidate_map[current_id]
        current_dt = _parse_reg_date(str(current_doc.get("reg_date", ""))) or datetime.min
        best_id = current_id
        best_dt = current_dt
        for rule_name in _candidate_rule_names(current_doc):
            latest = latest_by_rule.get(rule_name)
            if latest is None:
                continue
            latest_id, latest_dt = latest
            if latest_dt > best_dt:
                best_id = latest_id
                best_dt = latest_dt
        if best_id in seen:
            continue
        seen.add(best_id)
        promoted.append(best_id)
    return promoted


def _pick_fallback_candidate_ids(candidates: list[dict], *, limit: int = 2) -> list[str]:
    if not candidates:
        return []
    seen_titles: set[str] = set()
    picked: list[str] = []
    # fallback은 최신일자 우선이 아니라 "이미 랭크된 후보 순서"를 존중한다.
    for doc in candidates:
        title_key = _title_group_key(str(doc.get("title", "")))
        if title_key and title_key in seen_titles:
            continue
        doc_id = str(doc.get("id", "")).strip()
        if not doc_id:
            continue
        if title_key:
            seen_titles.add(title_key)
        picked.append(doc_id)
        if len(picked) >= max(1, limit):
            break
    return picked


def _build_intent_notice(
    *,
    current_question: str,
    intent_terms: list[str],
    no_match: bool,
    link_count: int,
    llm_brief: str,
) -> str:
    def _is_placeholder_brief(text: str) -> bool:
        s = (text or "").strip()
        if not s:
            return True
        blocked_phrases = (
            "사용자에게 보일",
            "1~2문장 요약",
            "예시",
            "placeholder",
            "selected_ids",
            "no_match",
        )
        return any(p in s.lower() for p in blocked_phrases)

    terms = [str(t or "").strip() for t in intent_terms if str(t or "").strip()]
    if not terms:
        terms = _sanitize_keywords(
            re.findall(r"[0-9A-Za-z가-힣]{2,}", current_question or ""),
            max_keywords=3,
        )
    # 톤 일관성을 위해 서버에서 자연스러운 안내문을 고정해 사용한다.
    brief = (llm_brief or "").strip()
    if brief and _is_placeholder_brief(brief):
        brief = ""
    if no_match and link_count == 0:
        return "현재 질문만으로는 적용 대상을 단정하기 어려워요. 확인하려는 가족관계와 상황(예: 사망/결혼/출산)을 알려주시면 관련 규정을 정확히 안내해드릴게요."
    return "확인해볼 만한 사규를 찾아봤어요. 아래 문서부터 보시면 가장 빠르게 판단하실 수 있어요."


def _shorten_summary(text: str, *, max_len: int = 170) -> str:
    # 문자열 길이를 강제로 자르지 않고, 첫 완결 문장만 사용한다.
    # (max_len은 기존 호출부 호환을 위한 인자이며, 절단에는 사용하지 않는다.)
    src = re.sub(r"\s+", " ", text or "")
    if not src:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", src)
    first_sentence = sentences[0] if sentences else src
    return first_sentence.rstrip(" ,;:/-")


def _build_doc_brief_description(doc: dict, intent_terms: list[str]) -> str:
    points = _pick_link_points(doc.get("keywords", []) or [], intent_terms, max_points=3)
    if points:
        return f"이 문서에는 {', '.join(points)} 관련 기준과 절차가 정리돼 있어요."

    summary_first_sentence = _shorten_summary(str(doc.get("summary", "")))
    if summary_first_sentence:
        return summary_first_sentence

    return "이 문서에는 적용 대상, 기준, 절차 같은 핵심 내용이 정리돼 있어요."


def _build_llm_writer_output(
    *,
    client: LLMClient,
    current_question: str,
    intent_terms: list[str],
    selected_docs: list[dict],
) -> tuple[str, dict[str, str], str]:
    if not selected_docs:
        return "", {}, ""

    writer_payload = {
        "question": current_question,
        "intent_terms": intent_terms,
        "selected_docs": [
            {
                "id": str(d.get("id", "")),
                "title": str(d.get("title", "")),
                "summary": str(d.get("summary", "")),
                "keywords": d.get("keywords", []) or [],
            }
            for d in selected_docs
        ],
    }
    try:
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": FINAL_ANSWER_WRITER_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(writer_payload, ensure_ascii=False)},
            ],
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _safe_json_loads(raw)
    except Exception:
        return "", {}, ""

    intro = str(parsed.get("intro", "")).strip()
    closing = str(parsed.get("closing", "")).strip()
    raw_items = parsed.get("item_descriptions", {})
    item_descriptions: dict[str, str] = {}
    if isinstance(raw_items, dict):
        for key, value in raw_items.items():
            sid = str(key or "").strip()
            desc = str(value or "").strip()
            if sid and desc:
                item_descriptions[sid] = desc
    return intro, item_descriptions, closing


def _replace_doc_id_mentions(text: str, selected_docs: list[dict]) -> str:
    src = str(text or "")
    if not src.strip():
        return ""
    out = src
    for doc in selected_docs:
        sid = str(doc.get("id", "")).strip()
        title = str(doc.get("title", "")).strip()
        if not sid:
            continue
        replacement = title or "해당 문서"
        out = re.sub(rf"\b{re.escape(sid)}\b", replacement, out, flags=re.IGNORECASE)
    return out.strip()


def _sanitize_writer_text(text: str) -> str:
    src = str(text or "").strip()
    if not src:
        return ""
    banned_markers = (
        "인사팀에 문의",
        "인사팀 문의",
        "담당부서에 문의",
        "담당 부서에 문의",
        "문의해 주세요",
        "문의하시기 바랍니다",
    )
    if any(marker in src for marker in banned_markers):
        return ""
    return src


def _pick_link_points(doc_keywords: list[str], intent_terms: list[str], *, max_points: int = 3) -> list[str]:
    kws = [str(k).strip() for k in (doc_keywords or []) if str(k).strip()]
    if not kws:
        return []
    terms = [str(t).strip().lower() for t in (intent_terms or []) if str(t).strip()]
    if not terms:
        return kws[: max(1, max_points)]
    overlap = [kw for kw in kws if any(t in kw.lower() for t in terms)]
    if overlap:
        return overlap[: max(1, max_points)]
    return kws[: max(1, max_points)]


def _render_link_focused_answer(
    *,
    current_question: str,
    intent_terms: list[str],
    brief: str,
    selected_ids: list[str],
    no_match: bool,
    candidate_map: dict[str, dict],
    writer_intro: str = "",
    writer_item_descriptions: dict[str, str] | None = None,
    writer_closing: str = "",
) -> str:
    valid_ids: list[str] = []
    seen: set[str] = set()
    for sid in selected_ids:
        key = str(sid or "").strip()
        if not key or key in seen:
            continue
        if key not in candidate_map:
            continue
        seen.add(key)
        valid_ids.append(key)

    item_desc_map = writer_item_descriptions or {}
    first_line = (writer_intro or "").strip() or _build_intent_notice(
        current_question=current_question,
        intent_terms=intent_terms,
        no_match=no_match,
        link_count=len(valid_ids),
        llm_brief=brief,
    )
    lines: list[str] = [first_line]
    if valid_ids:
        lines.append("")
        for idx, sid in enumerate(valid_ids, 1):
            doc = candidate_map[sid]
            title = str(doc.get("title", "")).strip() or sid
            url = str(doc.get("url", "")).strip()
            if not url:
                continue
            brief_desc = (
                item_desc_map.get(sid, "").strip()
                or _build_doc_brief_description(doc, intent_terms)
            )
            lines.append(f"{idx}) [{title}]({url})")
            lines.append(brief_desc)
            lines.append("")
        lines.append(
            (writer_closing or "").strip()
            or "필요하시면 위 문서 중 어떤 항목을 먼저 확인하면 되는지 함께 정리해드릴게요."
        )

    if not lines:
        return "질문에서 추출한 의도와 직접 일치하는 사규 링크를 찾지 못했습니다."
    return "\n".join(lines).strip()


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


def _select_answer_context_hits(hits: list[SearchHit], query: str, *, limit: int = ANSWER_CONTEXT_LIMIT) -> list[SearchHit]:
    """답변용 컨텍스트를 축소하되, 질의 매칭 문서에 가중치를 준다."""
    if not hits:
        return []

    cap_size = max(max(1, limit), min(len(hits), max(1, limit) * 3))
    candidate_hits = hits[:cap_size]
    query_terms = _query_terms(query)
    if not query_terms:
        return candidate_hits[: max(1, limit)]

    weighted: list[tuple[float, int, SearchHit]] = []
    for idx, hit in enumerate(candidate_hits):
        title_kw_text = " ".join(
            [
                hit.title or "",
                " ".join(str(k) for k in (hit.summary_keywords or [])),
            ]
        ).lower()
        summary_text = (hit.summary_text or "").lower()
        title_kw_hits = sum(1 for term in query_terms if term and term in title_kw_text)
        summary_hits = sum(1 for term in query_terms if term and term in summary_text)
        # 완전 필터링 대신 매칭 문서 가중치만 크게 부여해 연관 문서를 함께 유지한다.
        match_boost = min(title_kw_hits * 0.45, 1.2) + min(summary_hits * 0.2, 0.6)
        weighted.append((float(hit.score) + match_boost, idx, hit))

    weighted.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return [hit for _, _, hit in weighted[: max(1, limit)]]


def _search_with_low_score_fallback(query: str) -> SearchResult:
    result = search_with_fallback(query, limit=SEARCH_FETCH_LIMIT)
    if result.top_score >= LOW_SCORE_THRESHOLD:
        return result

    expanded = _build_expanded_query_by_intent(query)
    retried = search_with_fallback(expanded, limit=SEARCH_FETCH_LIMIT)
    if retried.top_score > result.top_score:
        return retried
    return result


def _recent_user_questions(messages: list[dict], *, limit: int = TOPIC_LOCK_MAX_USER_TURNS, exclude_question: str = "") -> list[str]:
    out: list[str] = []
    ex = (exclude_question or "").strip()
    for m in reversed(messages or []):
        if m.get("role") != "user":
            continue
        q = _strip_embedded_chat_history(_content_to_text(m.get("content", "")))
        if not q:
            continue
        if ex and q == ex:
            continue
        out.append(q)
        if len(out) >= max(1, limit):
            break
    out.reverse()
    return out


def _has_regulation_hints(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    hint_terms = (
        "사규",
        "규정",
        "규칙",
        "복리",
        "복리후생",
        "경조",
        "경조금",
        "결혼",
        "사내부부",
        "출산",
        "육아",
        "휴가",
        "지원",
        "수당",
        "지급",
        "대상",
        "조건",
        "절차",
        "퇴사",
        "입사",
    )
    return any(term in t for term in hint_terms)


def has_regulation_hints(text: str) -> bool:
    """서버 라우팅에서 사용할 공개 헬퍼."""
    return _has_regulation_hints(text)


def _is_ambiguous_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    followup_markers = (
        "아니지",
        "그거",
        "이거",
        "그럼",
        "맞아",
        "맞지",
        "그건",
        "이건",
        "받을 수",
        "받을수",
        "될까",
        "되나",
        "가능",
    )
    if any(marker in t for marker in followup_markers):
        return True
    return False


def _is_clear_non_regulation_query(text: str) -> bool:
    """일상/잡담 질문은 topic lock 대상에서 제외한다."""
    t = (text or "").strip().lower()
    if not t:
        return False
    if _has_regulation_hints(t):
        return False
    non_reg_terms = (
        "안녕",
        "하이",
        "날씨",
        "기온",
        "비 와",
        "눈 와",
        "미세먼지",
        "교통",
        "지하철",
        "버스",
        "몇 시",
        "맛집",
        "밥",
        "점심",
        "저녁",
        "심심",
        "잡담",
    )
    return any(term in t for term in non_reg_terms)


def is_clear_non_regulation_query(text: str) -> bool:
    """서버 라우팅에서 사용할 공개 헬퍼."""
    return _is_clear_non_regulation_query(text)


def should_topic_lock(messages: list[dict], current_question: str, *, max_user_turns: int = TOPIC_LOCK_MAX_USER_TURNS) -> bool:
    current = _strip_embedded_chat_history(current_question)
    if not current or _is_meta_task_prompt(current):
        return False
    if _is_clear_non_regulation_query(current):
        return False
    recent = _recent_user_questions(messages, limit=max_user_turns, exclude_question=current)
    if not recent:
        return False
    regulation_hits = sum(1 for q in recent if _has_regulation_hints(q))
    if regulation_hits <= 0:
        return False
    # 직전 대화가 규정 주제면, 애매 후속질문뿐 아니라 규정 힌트 재지정 문장도 lock 유지
    return _is_ambiguous_followup(current) or _has_regulation_hints(current)


def _build_context_terms_from_recent_questions(
    messages: list[dict],
    current_question: str,
    *,
    max_user_turns: int = TOPIC_LOCK_MAX_USER_TURNS,
) -> list[str]:
    recent = _recent_user_questions(messages, limit=max_user_turns, exclude_question=current_question)
    merged = " ".join(recent)
    if not merged.strip():
        return []
    # 문맥 키워드는 heuristic으로 추출해 API 호출 추가를 막는다.
    return _heuristic_regulation_keywords(merged, max_keywords=3)


def _all_generic_keywords(keywords: list[str]) -> bool:
    if not keywords:
        return False
    generic_set = {"대상", "조건", "지원", "절차", "가능", "여부", "기준", "내용", "규정"}
    normalized = [k.strip().lower() for k in keywords if k.strip()]
    return bool(normalized) and all(k in generic_set for k in normalized)


def choose_search_query(
    messages: list[dict],
    current_question: str,
    *,
    keyword_decision: KeywordExtractionDecision | None = None,
) -> QueryDecision:
    normalized_question = _strip_embedded_chat_history(current_question)
    resolved_keyword_decision = keyword_decision or extract_regulation_keywords(normalized_question, max_keywords=5)
    extracted_keywords = _augment_event_keywords(
        normalized_question,
        resolved_keyword_decision.cleaned_keywords,
        max_keywords=7,
    )
    topic_locked = should_topic_lock(messages, normalized_question)
    prev_questions = _recent_user_questions(messages, limit=1, exclude_question=normalized_question)
    prev_question = prev_questions[-1] if prev_questions else ""
    base_query = normalized_question
    if topic_locked and _is_ambiguous_followup(normalized_question) and prev_question:
        # 애매한 후속질문은 직전 질문을 결합해 검색축을 고정한다.
        base_query = f"{prev_question} {normalized_question}".strip()

    # 단일 질의 정책:
    # - 원문 질문 + LLM 추출 핵심키워드를 하나의 검색 질의로 결합
    context_terms = _build_context_terms_from_recent_questions(
        messages,
        normalized_question,
        max_user_turns=TOPIC_LOCK_MAX_USER_TURNS,
    )
    context_terms = _sanitize_keywords(context_terms, max_keywords=3)
    use_context_terms = bool(context_terms and (topic_locked or _all_generic_keywords(extracted_keywords)))

    keyword_parts = list(extracted_keywords)
    if use_context_terms:
        for kw in context_terms:
            if kw not in keyword_parts:
                keyword_parts.append(kw)
    keyword_query = " ".join(keyword_parts).strip()
    single_query = base_query
    if keyword_query and keyword_query.lower() not in base_query.lower():
        single_query = f"{base_query} {keyword_query}".strip()
    search_queries = [single_query] if single_query else [normalized_question]
    raw_results = [_search_with_low_score_fallback(search_queries[0])] if search_queries[0] else []
    merged_result = raw_results[0] if raw_results else SearchResult(query=normalized_question, hits=[], mode="hybrid_single")
    tie_reason = "single_query_keyword_rerank_score_first"
    query_mode = "single_query"

    rerank_query = " ".join(extracted_keywords) if extracted_keywords else normalized_question
    reranked_result = _rerank_hits_by_last_query(merged_result, rerank_query)
    # 최종 rank는 리랭킹 score 기준을 우선한다.
    final_result = SearchResult(
        query=reranked_result.query,
        hits=reranked_result.hits,
        mode=reranked_result.mode,
    )
    return QueryDecision(
        chosen_query=current_question,
        normalized_query=normalized_question,
        score_a=final_result.top_score,
        score_b=0.0,
        tie_break_reason=tie_reason,
        result=final_result,
        extracted_keywords=extracted_keywords,
        search_queries=search_queries or [normalized_question],
        raw_extracted_keywords=resolved_keyword_decision.raw_keywords,
        keyword_source=resolved_keyword_decision.source,
        keyword_reason=resolved_keyword_decision.reason,
        rerank_query=rerank_query,
        debug={
            "query_mode": query_mode,
            "topic_locked": topic_locked,
            "prev_question": prev_question,
            "base_query": base_query,
            "augmented_keywords": extracted_keywords,
            "context_terms": context_terms,
            "use_context_terms": use_context_terms,
            "raw_result_count": len(raw_results),
            "raw_query_scores": [
                {
                    "query": r.query,
                    "mode": r.mode,
                    "top_score": r.top_score,
                    "hit_count": len(r.hits),
                    "top_title": (r.hits[0].title if r.hits else ""),
                }
                for r in raw_results
            ],
        },
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


def generate_answer_json(
    *,
    messages: list[dict],
    current_question: str,
    decision: QueryDecision,
    use_llm_selector: bool = True,
    use_llm_writer: bool = True,
) -> dict:
    client = LLMClient(timeout=120.0)

    context_query = " ".join(decision.extracted_keywords) if decision.extracted_keywords else decision.normalized_query
    answer_context_hits = _select_answer_context_hits(
        decision.result.hits,
        context_query,
        limit=ANSWER_CONTEXT_LIMIT,
    )
    answer_context_result = SearchResult(
        query=decision.result.query,
        hits=answer_context_hits,
        mode=decision.result.mode,
    )
    context_text = _build_context(answer_context_result)
    history_text = _messages_to_history(messages)
    candidate_docs = _build_candidate_docs_for_llm(answer_context_hits, limit=ANSWER_CONTEXT_LIMIT)
    candidate_map = _build_candidate_map(candidate_docs)
    user_prompt = (
        f"[검색에 사용된 질의]\n{decision.chosen_query}\n\n"
        f"[추출 키워드]\n{', '.join(decision.extracted_keywords) if decision.extracted_keywords else '없음'}\n\n"
        f"[멀티 검색 질의]\n" + "\n".join(f"- {q}" for q in decision.search_queries) + "\n\n"
        f"[검색 점수]\nscore_A={decision.score_a:.4f}, score_B={decision.score_b:.4f}\n"
        f"decision={decision.tie_break_reason}\n\n"
        f"[검색 컨텍스트]\n{context_text}\n\n"
        f"[candidate_docs]\n{json.dumps(candidate_docs, ensure_ascii=False, indent=2)}\n\n"
        f"[최근 대화(최대 4턴)]\n{history_text}\n\n"
        f"[현재 사용자 질문]\n{current_question}"
    )
    parsed: dict = {}
    raw = ""
    if use_llm_selector:
        resp = client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            # temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        parsed = _safe_json_loads(raw)

    standalone_query = str(parsed.get("standalone_query", decision.chosen_query))
    brief = str(parsed.get("brief", "")).strip()
    no_match = bool(parsed.get("no_match", False))
    raw_selected = parsed.get("selected_ids", [])
    selected_ids = [str(x).strip() for x in raw_selected] if isinstance(raw_selected, list) else []
    selected_ids = [sid for sid in selected_ids if sid in candidate_map]
    if not use_llm_selector:
        selected_ids = _pick_fallback_candidate_ids(candidate_docs, limit=2)
        no_match = len(selected_ids) == 0
    elif not selected_ids:
        # selector가 no_match로 닫았더라도 검색 점수가 충분히 높으면 링크 1~2개는 보수적으로 제공한다.
        if no_match and decision.score_a >= 1.0:
            selected_ids = _pick_fallback_candidate_ids(candidate_docs, limit=2)
            no_match = len(selected_ids) == 0
        elif not no_match:
            selected_ids = _pick_fallback_candidate_ids(candidate_docs, limit=2)
    selected_ids = _promote_selected_ids_to_latest(selected_ids, candidate_docs, candidate_map)
    selected_docs = [candidate_map[sid] for sid in selected_ids if sid in candidate_map]

    writer_intro = ""
    writer_item_descriptions: dict[str, str] = {}
    writer_closing = ""
    if use_llm_writer and selected_docs:
        writer_intro, writer_item_descriptions, writer_closing = _build_llm_writer_output(
            client=client,
            current_question=current_question,
            intent_terms=decision.extracted_keywords,
            selected_docs=selected_docs,
        )
        writer_intro = _replace_doc_id_mentions(writer_intro, selected_docs)
        writer_closing = _replace_doc_id_mentions(writer_closing, selected_docs)
        writer_intro = _sanitize_writer_text(writer_intro)
        writer_closing = _sanitize_writer_text(writer_closing)
        writer_item_descriptions = {
            sid: _sanitize_writer_text(_replace_doc_id_mentions(desc, selected_docs))
            for sid, desc in writer_item_descriptions.items()
        }

    answer = _render_link_focused_answer(
        current_question=current_question,
        intent_terms=decision.extracted_keywords,
        brief=brief,
        selected_ids=selected_ids,
        no_match=no_match,
        candidate_map=candidate_map,
        writer_intro=writer_intro,
        writer_item_descriptions=writer_item_descriptions,
        writer_closing=writer_closing,
    )
    selected_links = [
        {
            "id": sid,
            "title": str(candidate_map.get(sid, {}).get("title", "")),
            "url": str(candidate_map.get(sid, {}).get("url", "")),
        }
        for sid in selected_ids
        if sid in candidate_map
    ]
    return {
        "standalone_query": standalone_query,
        "answer": answer,
        "debug": {
            "selector_mode": "llm_selector" if use_llm_selector else "rank_fallback_selector",
            "writer_mode": "llm_writer" if use_llm_writer else "rule_writer",
            "llm_selector_raw": raw[:3000],
            "llm_selector_parsed": parsed,
            "llm_writer_intro": writer_intro,
            "llm_writer_item_count": len(writer_item_descriptions),
            "candidate_count": len(candidate_docs),
            "selected_ids": selected_ids,
            "selected_links": selected_links,
            "no_match": no_match,
        },
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
    # 예: "경비지급규정", "취업규칙", "운영방침", "시행지침", "관리부칙" 등
    raw = re.findall(r"[0-9A-Za-z가-힣]+(?:규정|규칙|부칙|기준|방침|준칙|정책|지침|제정)", text or "")
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
    client: LLMClient,
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
        # temperature=0.1,
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

    missing = missing[:RELATED_LINK_APPEND_LIMIT]
    lines = [answer.strip(), "", "추가로 확인할 관련 규정:"]
    for h in missing:
        lines.append(f"- [관련링크 : {h.title}]({h.source_url})")
    return "\n".join(lines).strip()


def append_sq_comment(answer: str, standalone_query: str) -> str:
    sq = (standalone_query or "").strip().replace("-->", "-- >")
    return f"{(answer or '').strip()}\n<!-- sq:{sq} -->"

