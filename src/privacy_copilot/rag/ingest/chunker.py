# src/privacy_copilot/rag/ingest/chunker.py
from pathlib import Path
import re, json, hashlib
from typing import Iterable, Dict

# Optional parsers
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None


def normalize(text: str) -> str:
    """Collapse whitespace; keep it readable."""
    return re.sub(r"\s+", " ", (text or "")).strip()


def chunks_from_pdf(path: Path, min_words: int = 60) -> Iterable[Dict]:
    if PdfReader is None:
        raise RuntimeError("pypdf not installed. pip install pypdf")
    reader = PdfReader(str(path))
    for i, page in enumerate(reader.pages):
        text = normalize(page.extract_text())
        # If the PDF has article headings, split on them; else fall back to page-chunks
        parts = re.split(r"(?im)(?=^Article\s+\d+\b|^Section\b|^Chapitre\b)", text) or [text]
        for chunk in parts:
            if len(chunk.split()) >= min_words:
                yield {"text": chunk, "source_file": path.name, "page": i + 1}


def chunks_from_html(path: Path, min_words: int = 15) -> Iterable[Dict]:
    if BeautifulSoup is None:
        raise RuntimeError("bs4/lxml not installed. pip install beautifulsoup4 lxml")
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    # Grab headings + paragraphs + list items
    for node in soup.select("h1,h2,h3,p,li"):
        txt = normalize(node.get_text(" "))
        if len(txt.split()) >= min_words:
            yield {"text": txt, "source_file": path.name, "tag": node.name}


def build_corpus(raw_dir: str = "data/public/raw", out: str = "data/public/chunks.jsonl") -> int:
    raw = Path(raw_dir)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out_p.open("w", encoding="utf-8") as f:
        for p in raw.glob("**/*"):
            gen = None
            if p.suffix.lower() == ".pdf":
                gen = chunks_from_pdf(p)
            elif p.suffix.lower() in {".html", ".htm"}:
                gen = chunks_from_html(p)
            if not gen:
                continue
            for item in gen:
                # Stable ID for traceability
                key = f"{item.get('source_file','')}-{item.get('page', item.get('tag',''))}-{item['text'][:120]}"
                item["id"] = hashlib.md5(key.encode("utf-8")).hexdigest()
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    n = build_corpus()
    print(f"Wrote {n} chunks → data/public/chunks.jsonl")
