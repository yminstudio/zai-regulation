# 사내 규정 검색 기반 챗봇 (`zai-regulation`)

사내 그룹웨어 규정 문서를 수집/요약/벡터화하여, OpenAI 호환 Chat API로 질의응답을 제공하는 프로젝트입니다.

- 규정 문서 수집: 그룹웨어 게시글 + 첨부파일
- 2단계 요약: 첨부 요약 + 게시글 통합 요약
- 벡터 저장: Weaviate
- 질의응답: FastAPI (`/v1/chat/completions`, `/regulation/chat`)
- LLM: Ollama 계열 엔드포인트(기본 `gpt-oss-20b-128k:latest`)
- 임베딩: OpenAI 또는 로컬 sentence-transformers (`bge-m3`, `e5-large` 등)

---

## 1) 프로젝트 구조

```text
zai-regulation/
├─ src/
│  ├─ server.py               # API 서버 (OpenAI 호환 포함)
│  ├─ ingest_pipeline.py      # collect->filter->summarize->weaviate 통합 파이프라인
│  ├─ collect_documents.py    # 게시글/첨부 수집
│  ├─ summarize_documents.py  # 1차/2차 요약
│  ├─ weaviate_ingest.py      # 스키마/벡터 적재
│  ├─ weaviate_search.py      # 검색 유틸
│  ├─ chat_service.py         # 질의 라우팅/검색/답변 생성
│  ├─ llm_client.py           # Ollama API 래퍼(OpenAI-like)
│  ├─ embedding_client.py     # 임베딩 백엔드 래퍼(OpenAI/local)
│  ├─ reembed_ingest.py       # 동일 문서 재임베딩 적재 유틸
│  └─ config.py               # 환경/설정 로딩
├─ bin/
│  ├─ run-api
│  ├─ run-collect
│  ├─ run-summarize
│  ├─ run-ingest
│  └─ run-reembed
├─ config/
│  ├─ base.yaml
│  ├─ dev.yaml
│  └─ prod.yaml
├─ data/                      # run 산출물/중간 결과
├─ logs/
├─ .env.example
└─ requirements.txt
```

---

## 2) 설정 로딩 우선순위

`src/config.py` 기준으로 아래 순서로 설정이 적용됩니다.

1. OS 환경변수
2. `.env` / `.venv/.env`
3. `config/<env>.yaml` (`APP_ENV`, 기본 `dev`)
4. `config/base.yaml`
5. 코드 기본값

---

## 3) 빠른 시작

### 3-1. 설치

```bash
pip install -r requirements.txt
cp .env.example .env
```

### 3-2. 필수 환경변수 확인

`.env`에서 최소 아래 항목을 설정하세요.

- `OLLAMA_BASE_URL` (예: `http://localhost:11434` 또는 사내 게이트웨이)
- `ANSWER_MODEL` (기본: `gpt-oss-20b-128k:latest`)
- `SUMMARIZE_MODEL` (기본: `gpt-oss-20b-128k:latest`)
- `EMBEDDING_BACKEND` (`openai` | `local_sentence_transformers`, 기본: `local_sentence_transformers`)
- `OPENAI_API_KEY` (OpenAI 임베딩 사용 시)
- `OPENAI_EMBEDDING_MODEL` (기본: `text-embedding-3-small`)
- `LOCAL_EMBEDDING_MODEL` (기본: `BAAI/bge-m3`)
- `API_PORT` (개발 기본값 예시: `8012`)

### 3-3. API 실행

```bash
bash bin/run-api
```

### 3-4. 파이프라인 실행 예시

```bash
# 최신 필터 결과 중 3건만 처리 + Weaviate 적재
bash bin/run-ingest --limit 3 --ingest-weaviate
```

---

## 4) 실행 스크립트

- `bin/run-api`: FastAPI 서버 실행
- `bin/run-collect`: 게시글/첨부 수집
- `bin/run-summarize`: 수집 결과 요약
- `bin/run-ingest`: 통합 파이프라인 실행
- `bin/run-reembed`: 기존 요약 JSON을 임베딩만 바꿔 재적재

---

## 5) API 엔드포인트

### 모델 목록 (OpenAI 호환)

- `GET /v1/models`

### 채팅 (OpenAI 호환)

- `POST /v1/chat/completions`
- `POST /regulation/chat` (동일 기능 별칭)

### 수집/요약/적재 실행

- `POST /regulation/ingest`

요청 예시:

```bash
curl -X POST "http://localhost:${API_PORT:-8012}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-regulation",
    "stream": false,
    "messages": [
      {"role":"user","content":"경비 지급 기준 알려줘"}
    ]
  }'
```

---

## 6) 파이프라인 개요

`src/ingest_pipeline.py` 기준:

1. 그룹웨어 게시글 목록 수집
2. 최신 규정 필터링(rule 기반, 옵션으로 LLM refine)
3. 본문/첨부 수집 및 텍스트 추출
4. 첨부 1차 요약 + 게시글 2차 통합 요약
5. 선택한 임베딩 백엔드(OpenAI/local)로 벡터 생성 후 Weaviate 적재

Run 산출물은 `data/runs/<run_id>/` 아래에 저장됩니다.

---

## 7) 운영 메모

- 모델 교체는 코드 수정 없이 `.env`/환경변수만 변경
- Weaviate 클래스명은 `ZaiRegulation` prefix 정책 사용
- OpenAI 임베딩 사용 시에만 `OPENAI_API_KEY` 필요
- 사내 게이트웨이 인증이 필요한 경우 `OLLAMA_API_KEY` 사용

---

## 8) 5문서 미니셋 임베딩 A/B 예시

같은 문서 5개를 유지한 채 임베딩만 바꿔 비교하려면, 수집/요약을 1회 수행한 뒤 `run-reembed`를 2회 실행하세요.

```bash
# 1) 문서 5개를 수집/요약 (적재는 생략)
bash bin/run-ingest --limit 5 --no-cleanup

# 2) 위 run의 결과 JSON 경로 확인 후 재임베딩 적재
# 예: data/runs/<run_id>/result/07_final_for_ingest.json

# bge-m3 class
bash bin/run-reembed \
  --input-json "data/runs/<run_id>/result/07_final_for_ingest.json" \
  --weaviate-class ZaiRegulation_db_bgem3 \
  --replace-own-collection \
  --embedding-backend local_sentence_transformers \
  --embedding-model BAAI/bge-m3

# e5 class
bash bin/run-reembed \
  --input-json "data/runs/<run_id>/result/07_final_for_ingest.json" \
  --weaviate-class ZaiRegulation_db_e5 \
  --replace-own-collection \
  --embedding-backend local_sentence_transformers \
  --embedding-model intfloat/multilingual-e5-large-instruct
```

---

## 9) 개발 환경 프로파일

- `APP_ENV=dev` -> `config/dev.yaml`
- `APP_ENV=prod` -> `config/prod.yaml`
- 필요 시 `CONFIG_FILE`로 특정 YAML 직접 지정 가능
