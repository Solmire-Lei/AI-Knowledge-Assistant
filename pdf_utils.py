from pathlib import Path
from typing import List


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Extract text from a PDF file.

    Uses pypdf first because it is lightweight and easy to install. If pypdf
    cannot read a file well, it tries PyMuPDF as a fallback when available.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    texts: List[str] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                texts.append(page_text.strip())
    except Exception:
        texts = []

    # Fallback: PyMuPDF, imported as fitz.
    if not texts:
        try:
            import fitz

            doc = fitz.open(str(path))
            for page in doc:
                page_text = page.get_text() or ""
                if page_text.strip():
                    texts.append(page_text.strip())
            doc.close()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "PDF parsing failed. Please install pypdf or PyMuPDF: pip install pypdf PyMuPDF"
            ) from exc

    text = "\n".join(texts)
    return clean_text(text)


def clean_text(text: str) -> str:
    """Clean extracted PDF text."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    if not text or not text.strip():
        return []

    text = clean_text(text)
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break
        start = max(end - overlap, start + 1)

    return chunks
