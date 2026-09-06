"""Text extraction for uploaded documents. PDF and plain text only."""
from __future__ import annotations

import io

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB


class ExtractionError(Exception):
    """Raised when an upload cannot be turned into text."""


def extract_text(filename: str, blob: bytes) -> str:
    if len(blob) > MAX_UPLOAD_BYTES:
        raise ExtractionError(
            f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not blob:
        raise ExtractionError("File is empty.")

    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _extract_pdf(blob)
    if name.endswith((".txt", ".md", ".csv")):
        return _extract_text(blob)
    raise ExtractionError(
        "Unsupported file type. Practical 1 accepts .pdf, .txt, .md or .csv.")


def _extract_text(blob: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return blob.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("Could not decode the file as text.")


def _extract_pdf(blob: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError(
            "pypdf is not installed. Run: pip install -r requirements-app.txt"
        ) from exc

    try:
        reader = PdfReader(io.BytesIO(blob))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise ExtractionError(f"Could not read the PDF: {exc}") from exc

    text = "\n".join(pages).strip()
    if not text:
        raise ExtractionError(
            "No selectable text found. This looks like a scanned PDF; "
            "OCR is out of scope for Practical 1.")
    return text
