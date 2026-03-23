"""사규 데이터 적재 API."""
from __future__ import annotations

import json
import os
import re
import time
import uuid

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from src.chat_service import (
    INTENT_NON_REGULATION_THRESHOLD,
    NON_REGULATION_GUIDE_LINE,
    classify_intent,
    choose_search_query,
    extract_current_user_question,
    generate_non_regulation_answer,
    generate_answer_json,
    has_regulation_hints,
    is_meta_task_prompt,
    should_topic_lock,
)
from src.config import ANSWER_MODEL, PROJECT_WEAVIATE_CLASS
from src.ingest_pipeline import PipelineOptions, run_pipeline
from src.log_utils import write_log

app = FastAPI(title="Regulation Ingest API", version="1.0.0")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME", "zai-regulation")
NON_REGULATION_FALLBACK_MESSAGE = (
    f"사규 상담 전용 챗봇입니다. {NON_REGULATION_GUIDE_LINE}"
)


class IngestRequest(BaseModel):
    # 0이면 최신 필터 결과 전체 처리
    limit: int = Field(default=0, ge=0)
    use_llm_filter: bool = False
    cleanup_enabled: bool = True
    ingest_weaviate: bool = True
    replace_own_collection: bool = True


class ChatMessage(BaseModel):
    role: str
    content: object


class ChatRequest(BaseModel):
    model: str = "gpt-5.4"
    messages: list[ChatMessage]
    stream: bool = True
    use_llm_selector: bool = True
    use_llm_writer: bool = True
    shadow_compare: bool = False


def _extract_link_urls(text: str) -> list[str]:
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    urls: list[str] = []
    seen: set[str] = set()
    for m in link_re.finditer(text or ""):
        url = m.group(1).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


@app.post("/regulation/ingest")
def regulation_ingest(req: IngestRequest) -> dict:
    try:
        result = run_pipeline(
            PipelineOptions(
                limit=req.limit,
                use_llm_filter=req.use_llm_filter,
                cleanup_enabled=req.cleanup_enabled,
                ingest_weaviate=req.ingest_weaviate,
                weaviate_class=PROJECT_WEAVIATE_CLASS,
                replace_own_collection=req.replace_own_collection,
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    docs = result.get("documents", [])
    items = [
        {
            "title": d.get("title", ""),
            "source_url": d.get("source_url", ""),
            "summary_text": d.get("summary_text", ""),
        }
        for d in docs
    ]
    return {
        "run_id": result.get("run_id"),
        "counts": result.get("counts", {}),
        "ingested_items": items,
    }


@app.get("/v1/models")
def openai_models() -> dict:
    """OpenAI 호환 모델 목록 엔드포인트."""
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_DISPLAY_NAME,
                "object": "model",
                "created": now,
                "owned_by": "zai-regulation",
            }
        ],
    }


def _stream_openai_delta(content: str, *, model: str) -> StreamingResponse:
    created = int(time.time())
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    def gen():
        # role chunk
        first = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(first, ensure_ascii=False)}\n\n"

        step = 24
        for i in range(0, len(content), step):
            piece = content[i : i + step]
            chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        last = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(last, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


def _chat_completion_payload(*, model: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


@app.post("/regulation/chat")
@app.post("/v1/chat/completions")
def regulation_chat(req: ChatRequest, request: Request):
    # 스트리밍 여부는 요청의 stream 플래그를 그대로 따른다.
    effective_stream = req.stream
    try:
        messages = [m.model_dump() for m in req.messages]
        current_question = extract_current_user_question(messages)
        if not current_question:
            raise HTTPException(status_code=400, detail="no user question found in messages")

        intent = classify_intent(current_question)
        topic_lock = should_topic_lock(messages, current_question)
        meta_task = is_meta_task_prompt(current_question) or intent.reason == "meta_task_prompt_guard"
        regulation_hint_override = (not meta_task) and has_regulation_hints(current_question)
        route = "regulation"
        if (
            intent.intent == "non_regulation"
            and intent.confidence >= INTENT_NON_REGULATION_THRESHOLD
            and not topic_lock
            and not regulation_hint_override
        ):
            route = "non_regulation"

        if route == "non_regulation":
            fallback_answer = generate_non_regulation_answer(current_question)
            write_log(
                "chat_intent_routed",
                {
                    "question": current_question,
                    "intent": intent.intent,
                    "confidence": intent.confidence,
                    "reason": intent.reason,
                    "meta_task": meta_task,
                    "topic_lock": topic_lock,
                    "regulation_hint_override": regulation_hint_override,
                    "route": route,
                    "fallback_answer_preview": fallback_answer[:160],
                    "stream": effective_stream,
                },
            )
            if effective_stream:
                return _stream_openai_delta(fallback_answer, model=MODEL_DISPLAY_NAME)
            return JSONResponse(_chat_completion_payload(model=MODEL_DISPLAY_NAME, content=fallback_answer))

        decision = choose_search_query(messages, current_question)
        out = generate_answer_json(
            messages=messages,
            current_question=current_question,
            decision=decision,
            use_llm_selector=req.use_llm_selector,
            use_llm_writer=req.use_llm_writer,
        )
        answer = out.get("answer", "").strip()
        standalone_query = out.get("standalone_query", "").strip()
        answer_links = _extract_link_urls(answer)
        # 사용자 노출 응답에는 sq 주석을 포함하지 않는다.
        assistant_content = answer
        write_log(
            "chat_query_processed",
            {
                "question": current_question,
                "chosen_query": decision.chosen_query,
                "normalized_query": decision.normalized_query,
                "raw_extracted_keywords": decision.raw_extracted_keywords,
                "extracted_keywords": decision.extracted_keywords,
                "keyword_source": decision.keyword_source,
                "keyword_reason": decision.keyword_reason,
                "search_queries": decision.search_queries,
                "rerank_query": decision.rerank_query,
                "top_score": decision.score_a,
                "search_mode": decision.result.mode,
                "hit_count": len(decision.result.hits),
                "request_modes": {
                    "use_multi_query": False,
                    "use_llm_selector": req.use_llm_selector,
                    "use_llm_writer": req.use_llm_writer,
                    "shadow_compare": req.shadow_compare,
                },
                "search_debug": decision.debug,
                "selector_debug": out.get("debug", {}),
                "answer_link_count": len(answer_links),
                "answer_links": answer_links,
                "top_hits": [
                    {
                        "rank": i + 1,
                        "title": h.title,
                        "reg_date": h.reg_date,
                        "score": h.score,
                        "source_url": h.source_url,
                    }
                    for i, h in enumerate(decision.result.hits[:5])
                ],
                "standalone_query": standalone_query,
                "stream": effective_stream,
                "intent": intent.intent,
                "intent_confidence": intent.confidence,
                "intent_reason": intent.reason,
                "meta_task": meta_task,
                "topic_lock": topic_lock,
                "regulation_hint_override": regulation_hint_override,
                "route": route,
            },
        )

        if req.shadow_compare:
            shadow_cases = [
                {"name": "selector_on", "use_llm_selector": True, "use_llm_writer": req.use_llm_writer},
                {"name": "selector_off", "use_llm_selector": False, "use_llm_writer": req.use_llm_writer},
            ]
            shadow_results: list[dict] = []
            for case in shadow_cases:
                if case["use_llm_selector"] == req.use_llm_selector:
                    continue
                d = choose_search_query(messages, current_question)
                o = generate_answer_json(
                    messages=messages,
                    current_question=current_question,
                    decision=d,
                    use_llm_selector=case["use_llm_selector"],
                    use_llm_writer=case["use_llm_writer"],
                )
                candidate_answer = str(o.get("answer", "")).strip()
                candidate_links = _extract_link_urls(candidate_answer)
                shadow_results.append(
                    {
                        "name": case["name"],
                        "use_multi_query": False,
                        "use_llm_selector": case["use_llm_selector"],
                        "use_llm_writer": case["use_llm_writer"],
                        "top_score": d.score_a,
                        "search_queries": d.search_queries,
                        "search_debug": d.debug,
                        "selector_debug": o.get("debug", {}),
                        "answer_preview": candidate_answer[:220],
                        "answer_link_count": len(candidate_links),
                        "answer_links": candidate_links,
                    }
                )
            write_log(
                "chat_shadow_compare",
                {
                    "question": current_question,
                    "primary_modes": {
                        "use_multi_query": False,
                        "use_llm_selector": req.use_llm_selector,
                        "use_llm_writer": req.use_llm_writer,
                    },
                    "primary_top_score": decision.score_a,
                    "primary_answer_link_count": len(answer_links),
                    "primary_answer_links": answer_links,
                    "comparisons": shadow_results,
                },
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if effective_stream:
        return _stream_openai_delta(assistant_content, model=MODEL_DISPLAY_NAME)

    payload = _chat_completion_payload(model=MODEL_DISPLAY_NAME, content=assistant_content)
    return JSONResponse(payload)

