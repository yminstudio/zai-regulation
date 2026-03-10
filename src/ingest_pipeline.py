"""Run 단위 수집-필터-요약 파이프라인.

현재 범위:
- collect -> latest filter -> summarize -> result JSON 생성
- VectorDB 업로드는 아직 수행하지 않음 (다음 단계)
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from openai import OpenAI

from src.config import DATA_DIR, OPENAI_API_KEY, PROJECT_WEAVIATE_CLASS
from src.extractors import extract_text
from src.filter_latest import llm_refine_filter, rule_based_filter
from src.gw_downloader import GWDownloader
from src.gw_list_fetcher import fetch_board_list
from src.log_utils import write_log
from src.summarize_documents import _doc_summary, _file_summary
from src.weaviate_ingest import ensure_collection, upsert_documents


@dataclass
class RunPaths:
    run_id: str
    root: Path
    raw: Path
    parsed: Path
    interim: Path
    result: Path
    logs: Path
    log_file: Path
    manifest: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _safe_name(name: str) -> str:
    keep = []
    for ch in (name or ""):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "file"


def _build_run_paths(run_id: str) -> RunPaths:
    root = DATA_DIR / "runs" / run_id
    raw = root / "raw"
    parsed = root / "parsed"
    interim = root / "interim"
    result = root / "result"
    logs = root / "logs"
    for p in (raw, parsed, interim, result, logs):
        p.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        run_id=run_id,
        root=root,
        raw=raw,
        parsed=parsed,
        interim=interim,
        result=result,
        logs=logs,
        log_file=logs / "pipeline.jsonl",
        manifest=root / "manifest.json",
    )


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(
    paths: RunPaths,
    *,
    status: str,
    started_at: str,
    finished_at: str | None = None,
    counts: dict | None = None,
    options: dict | None = None,
    error_summary: str = "",
    cleanup_applied: bool = False,
) -> None:
    payload = {
        "run_id": paths.run_id,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "counts": counts or {},
        "options": options or {},
        "error_summary": error_summary,
        "cleanup_applied": cleanup_applied,
        "paths": {
            "root": str(paths.root),
            "raw": str(paths.raw),
            "parsed": str(paths.parsed),
            "interim": str(paths.interim),
            "result": str(paths.result),
            "logs": str(paths.logs),
            "log_file": str(paths.log_file),
        },
    }
    _write_json(paths.manifest, payload)


def collect_and_filter(
    *,
    paths: RunPaths,
    limit: int,
    use_llm_filter: bool,
) -> list[dict]:
    gw = GWDownloader(download_dir=paths.raw)
    board = [x.to_dict() for x in fetch_board_list(gw)]
    _write_json(paths.interim / "01_board_full_list.json", board)
    write_log(
        "board_fetched",
        {"count": len(board), "output": str(paths.interim / "01_board_full_list.json")},
        log_path=paths.log_file,
    )

    rule = rule_based_filter(board)
    rule_payload = {
        "input_count": rule.input_count,
        "keep_count": rule.keep_count,
        "regulation_list": rule.regulation_list,
        "keep_source_urls": rule.keep_source_urls,
    }
    _write_json(paths.interim / "02_latest_rule.json", rule_payload)
    write_log(
        "latest_filtered_rule",
        {"keep_count": rule.keep_count, "output": str(paths.interim / "02_latest_rule.json")},
        log_path=paths.log_file,
    )

    keep_urls = rule.keep_source_urls
    if use_llm_filter:
        llm = llm_refine_filter(board, use_llm=True)
        llm_payload = {
            "input_count": llm.input_count,
            "keep_count": llm.keep_count,
            "regulation_list": llm.regulation_list,
            "keep_source_urls": llm.keep_source_urls,
        }
        _write_json(paths.interim / "03_latest_llm.json", llm_payload)
        write_log(
            "latest_filtered_llm",
            {"keep_count": llm.keep_count, "output": str(paths.interim / "03_latest_llm.json")},
            log_path=paths.log_file,
        )
        if llm.keep_source_urls:
            keep_urls = llm.keep_source_urls

    keep_set = set(keep_urls)
    latest_board = [b for b in board if b.get("source_url", "") in keep_set]
    selected = latest_board[: max(1, limit)]
    _write_json(paths.interim / "04_latest_selected_posts.json", selected)
    write_log(
        "latest_selected_posts",
        {
            "limit": limit,
            "selected_count": len(selected),
            "output": str(paths.interim / "04_latest_selected_posts.json"),
        },
        log_path=paths.log_file,
    )

    docs: list[dict] = []
    for idx, post in enumerate(selected, 1):
        source_url = post.get("source_url", "")
        title = post.get("title", "")
        source_text = gw.fetch_source_text(source_url)
        attachments = gw.get_attachments(source_url)
        write_log(
            "post_collect_start",
            {
                "index": idx,
                "title": title,
                "source_url": source_url,
                "attachments": len(attachments),
            },
            log_path=paths.log_file,
        )

        file_info: list[dict] = []
        for att in attachments:
            try:
                raw_path = gw.download_file(att)
                text, method = extract_text(raw_path)
                parsed_name = f"{uuid5(NAMESPACE_URL, source_url + '::' + att.file_name)}_{_safe_name(att.file_name)}.txt"
                parsed_path = paths.parsed / parsed_name
                parsed_path.write_text(text or "", encoding="utf-8")
                file_info.append(
                    {
                        "name": att.file_name,
                        "summary": "",
                        "keywords": [],
                        "raw_path": str(raw_path),
                        "parsed_path": str(parsed_path),
                        "extract_method": method,
                    }
                )
                write_log(
                    "attachment_collected",
                    {
                        "source_url": source_url,
                        "file_name": att.file_name,
                        "extract_method": method,
                        "text_length": len(text or ""),
                    },
                    log_path=paths.log_file,
                )
            except Exception as e:
                write_log(
                    "attachment_collect_failed",
                    {
                        "source_url": source_url,
                        "file_name": att.file_name,
                        "error": str(e),
                    },
                    log_path=paths.log_file,
                )

        docs.append(
            {
                "original_id": str(uuid5(NAMESPACE_URL, source_url)),
                "title": title,
                "reg_num": post.get("reg_num"),
                "reg_user": post.get("reg_user"),
                "reg_date": post.get("reg_date"),
                "source_url": source_url,
                "source_text": source_text,
                "file_info": file_info,
                "summary_text": "",
                "summary_keywords": [],
            }
        )

    _write_json(paths.interim / "05_schema_seed.json", docs)
    write_log(
        "collect_done",
        {
            "documents": len(docs),
            "attachments": sum(len(d.get("file_info", [])) for d in docs),
            "output": str(paths.interim / "05_schema_seed.json"),
        },
        log_path=paths.log_file,
    )
    return docs


def summarize_docs(*, paths: RunPaths, docs: list[dict]) -> list[dict]:
    client = OpenAI(api_key=(OPENAI_API_KEY or "").strip(), timeout=120.0)
    out_docs: list[dict] = []

    for doc in docs:
        source_text = doc.get("source_text", "") or ""
        files = doc.get("file_info", []) or []
        all_keywords: list[str] = []
        file_rows: list[dict] = []

        for f in files:
            name = f.get("name") or f.get("file_name") or ""
            parsed_path = Path(f.get("parsed_path", "")) if f.get("parsed_path") else None
            file_text = ""
            if parsed_path and parsed_path.exists():
                file_text = parsed_path.read_text(encoding="utf-8")
            file_summary, file_keywords = _file_summary(
                client,
                title=doc.get("title", "") or "",
                source_text=source_text,
                file_name=name,
                file_text=file_text,
            )
            for kw in file_keywords:
                if kw and kw not in all_keywords:
                    all_keywords.append(kw)
            file_rows.append(
                {
                    "name": name,
                    "summary": file_summary,
                    "keywords": file_keywords,
                    "parsed_path": f.get("parsed_path", ""),
                    "extract_method": f.get("extract_method", ""),
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

    _write_json(paths.interim / "06_summaries_interim.json", out_docs)
    _write_json(paths.result / "07_final_for_ingest.json", out_docs)
    write_log(
        "summarize_done",
        {
            "documents": len(out_docs),
            "output_interim": str(paths.interim / "06_summaries_interim.json"),
            "output_result": str(paths.result / "07_final_for_ingest.json"),
        },
        log_path=paths.log_file,
    )
    return out_docs


def cleanup_run(paths: RunPaths) -> None:
    for target in (paths.raw, paths.parsed, paths.interim):
        if target.exists():
            shutil.rmtree(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="run 기반 collect->filter->summarize 파이프라인")
    parser.add_argument("--limit", type=int, default=3, help="최신 필터 결과에서 처리할 게시글 수")
    parser.add_argument("--run-id", type=str, default="", help="지정하지 않으면 현재 시각 run_id 자동 생성")
    parser.add_argument("--use-llm-filter", action="store_true", help="최신 필터 2차 LLM 보정 사용")
    parser.add_argument("--no-cleanup", action="store_true", help="성공 후 raw/parsed/interim 삭제 비활성화")
    parser.add_argument("--ingest-weaviate", action="store_true", help="요약 결과를 Weaviate에 적재")
    parser.add_argument(
        "--weaviate-class",
        type=str,
        default=PROJECT_WEAVIATE_CLASS,
        help="적재 대상 Weaviate class (안전 prefix 강제)",
    )
    parser.add_argument(
        "--replace-own-collection",
        action="store_true",
        help="우리 프로젝트 class만 삭제 후 재생성",
    )
    args = parser.parse_args()

    run_id = args.run_id.strip() or _new_run_id()
    paths = _build_run_paths(run_id)
    started_at = _utc_now()
    options = {
        "limit": args.limit,
        "use_llm_filter": bool(args.use_llm_filter),
        "cleanup_default_on": True,
        "cleanup_enabled": not args.no_cleanup,
        "weaviate_ingest_enabled": bool(args.ingest_weaviate),
        "weaviate_class": args.weaviate_class,
        "replace_own_collection": bool(args.replace_own_collection),
    }
    _write_manifest(paths, status="running", started_at=started_at, options=options)
    write_log("run_started", {"run_id": run_id, "options": options}, log_path=paths.log_file)

    try:
        docs = collect_and_filter(paths=paths, limit=args.limit, use_llm_filter=args.use_llm_filter)
        final_docs = summarize_docs(paths=paths, docs=docs)
        ingest_report = {"input_count": 0, "error_count": 0}
        if args.ingest_weaviate:
            ensure_collection(
                args.weaviate_class,
                replace_own_collection=bool(args.replace_own_collection),
                allowed_replace_classes=[PROJECT_WEAVIATE_CLASS],
            )
            ingest_report = upsert_documents(
                class_name=args.weaviate_class,
                run_id=run_id,
                docs=final_docs,
            )
            _write_json(paths.result / "08_weaviate_ingest_report.json", ingest_report)
            write_log(
                "weaviate_ingest_done",
                {
                    "class_name": args.weaviate_class,
                    "input_count": ingest_report.get("input_count", 0),
                    "error_count": ingest_report.get("error_count", 0),
                    "report": str(paths.result / "08_weaviate_ingest_report.json"),
                },
                log_path=paths.log_file,
            )
            if ingest_report.get("error_count", 0) > 0:
                raise RuntimeError(
                    f"weaviate ingest had errors: {ingest_report.get('error_count', 0)}"
                )

        cleaned = False
        if not args.no_cleanup:
            cleanup_run(paths)
            cleaned = True
            write_log("cleanup_done", {"removed": ["raw", "parsed", "interim"]}, log_path=paths.log_file)

        counts = {
            "board_selected": len(docs),
            "attachments": sum(len(d.get("file_info", [])) for d in docs),
            "result_documents": len(final_docs),
            "weaviate_ingested": ingest_report.get("input_count", 0),
            "weaviate_error_count": ingest_report.get("error_count", 0),
        }
        _write_manifest(
            paths,
            status="success",
            started_at=started_at,
            finished_at=_utc_now(),
            counts=counts,
            options=options,
            cleanup_applied=cleaned,
        )
        print(f"run_root: {paths.root}")
        print(f"result:   {paths.result / '07_final_for_ingest.json'}")
        print(f"log:      {paths.log_file}")
    except Exception as e:
        _write_manifest(
            paths,
            status="failed",
            started_at=started_at,
            finished_at=_utc_now(),
            options=options,
            error_summary=str(e),
        )
        write_log("run_failed", {"error": str(e)}, log_path=paths.log_file)
        raise


if __name__ == "__main__":
    main()

