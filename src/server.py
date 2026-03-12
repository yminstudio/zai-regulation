"""사규 데이터 적재 API."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse, StreamingResponse

from src.chat_service import (
    INTENT_NON_REGULATION_THRESHOLD,
    NON_REGULATION_GUIDE_LINE,
    append_sq_comment,
    classify_intent,
    choose_search_query,
    extract_current_user_question,
    generate_non_regulation_answer,
    generate_answer_json,
)
from src.config import PROJECT_WEAVIATE_CLASS
from src.ingest_pipeline import PipelineOptions, run_pipeline
from src.log_utils import write_log

app = FastAPI(title="Regulation Ingest API", version="1.0.0")
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
    model: str = "gpt-4o-mini"
    messages: list[ChatMessage]
    stream: bool = True


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
def regulation_chat(req: ChatRequest):
    try:
        messages = [m.model_dump() for m in req.messages]
        current_question = extract_current_user_question(messages)
        if not current_question:
            raise HTTPException(status_code=400, detail="no user question found in messages")

        intent = classify_intent(current_question)
        route = "regulation"
        if intent.intent == "non_regulation" and intent.confidence >= INTENT_NON_REGULATION_THRESHOLD:
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
                    "route": route,
                    "fallback_answer_preview": fallback_answer[:160],
                    "stream": req.stream,
                },
            )
            if req.stream:
                return _stream_openai_delta(fallback_answer, model=req.model)
            return JSONResponse(_chat_completion_payload(model=req.model, content=fallback_answer))

        decision = choose_search_query(messages, current_question)
        out = generate_answer_json(
            messages=messages,
            current_question=current_question,
            decision=decision,
        )
        answer = out.get("answer", "").strip()
        standalone_query = out.get("standalone_query", "").strip()
        assistant_content = append_sq_comment(answer, standalone_query)
        write_log(
            "chat_query_processed",
            {
                "question": current_question,
                "chosen_query": decision.chosen_query,
                "top_score": decision.score_a,
                "search_mode": decision.result.mode,
                "hit_count": len(decision.result.hits),
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
                "stream": req.stream,
                "intent": intent.intent,
                "intent_confidence": intent.confidence,
                "intent_reason": intent.reason,
                "route": route,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if req.stream:
        return _stream_openai_delta(assistant_content, model=req.model)

    payload = _chat_completion_payload(model=req.model, content=assistant_content)
    return JSONResponse(payload)

