"""첨부파일 텍스트 추출기.

지원 형식:
- PDF (텍스트 우선, 부족하면 페이지 OCR)
- HWP/HWPX (가능 시 추출, 실패 시 빈 문자열)
- Excel (xls/xlsx)
- Word (docx/doc)
- 이미지 (OCR)
"""
from __future__ import annotations

import io
import re
import subprocess
import zipfile
from pathlib import Path

import pytesseract
import xlrd
from PIL import Image
from docx import Document
from openpyxl import load_workbook
from pypdf import PdfReader
import pypdfium2 as pdfium

OCR_LANG = "kor+eng"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def _clean_text(value: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", (value or "").strip())


def _ocr_image_file(path: Path) -> str:
    img = Image.open(path)
    return _clean_text(pytesseract.image_to_string(img, lang=OCR_LANG))


def _extract_pdf_text(path: Path) -> tuple[str, str]:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    text = _clean_text("\n".join(parts))
    if len(text) >= 50:
        return text, "pdf_text"

    # 텍스트가 거의 없으면 이미지 PDF로 판단하여 OCR 수행
    doc = pdfium.PdfDocument(str(path))
    ocr_parts: list[str] = []
    for idx in range(len(doc)):
        page = doc[idx]
        pil = page.render(scale=2).to_pil()
        ocr_parts.append(pytesseract.image_to_string(pil, lang=OCR_LANG))
    return _clean_text("\n".join(ocr_parts)), "pdf_ocr"


def _extract_docx_text(path: Path) -> str:
    doc = Document(str(path))
    lines = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return _clean_text("\n".join(lines))


def _extract_doc_text(path: Path) -> str:
    # antiword가 있으면 사용
    try:
        result = subprocess.run(
            ["antiword", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return _clean_text(result.stdout)
    except Exception:
        return ""


def _extract_xlsx_text(path: Path) -> str:
    wb = load_workbook(filename=str(path), data_only=True, read_only=True)
    out: list[str] = []
    for ws in wb.worksheets:
        out.append(f"[Sheet] {ws.title}")
        for row in ws.iter_rows(values_only=True):
            vals = [str(v).strip() for v in row if v is not None and str(v).strip()]
            if vals:
                out.append(" | ".join(vals))
    return _clean_text("\n".join(out))


def _extract_xls_text(path: Path) -> str:
    wb = xlrd.open_workbook(str(path))
    out: list[str] = []
    for sheet in wb.sheets():
        out.append(f"[Sheet] {sheet.name}")
        for r in range(sheet.nrows):
            vals = []
            for c in range(sheet.ncols):
                v = sheet.cell_value(r, c)
                if str(v).strip():
                    vals.append(str(v).strip())
            if vals:
                out.append(" | ".join(vals))
    return _clean_text("\n".join(out))


def _extract_hwpx_text(path: Path) -> str:
    # HWPX는 zip 기반 XML. 텍스트 노드만 단순 추출.
    try:
        with zipfile.ZipFile(path, "r") as zf:
            lines: list[str] = []
            for name in zf.namelist():
                if not name.endswith(".xml"):
                    continue
                data = zf.read(name).decode("utf-8", errors="ignore")
                lines.extend(re.findall(r">([^<>]+)<", data))
            return _clean_text("\n".join(lines))
    except Exception:
        return ""


def _extract_hwp_text(path: Path) -> str:
    # hwp5txt 명령이 설치된 경우에만 사용
    try:
        result = subprocess.run(
            ["hwp5txt", str(path)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return _clean_text(result.stdout)
    except Exception:
        return ""


def extract_text(path: str | Path) -> tuple[str, str]:
    """파일 텍스트 추출.

    Returns:
        (text, method)
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf_text(p)
    if ext in IMAGE_EXTS:
        return _ocr_image_file(p), "image_ocr"
    if ext == ".docx":
        return _extract_docx_text(p), "docx"
    if ext == ".doc":
        return _extract_doc_text(p), "doc"
    if ext == ".xlsx":
        return _extract_xlsx_text(p), "xlsx"
    if ext == ".xls":
        return _extract_xls_text(p), "xls"
    if ext == ".hwpx":
        return _extract_hwpx_text(p), "hwpx"
    if ext == ".hwp":
        return _extract_hwp_text(p), "hwp"

    # 텍스트 계열 fallback
    try:
        data = Path(path).read_text(encoding="utf-8", errors="ignore")
        return _clean_text(data), "plain_text"
    except Exception:
        return "", "unsupported"

