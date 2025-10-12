import os
import sys
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable
from dotenv import load_dotenv

# allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.weaviate_setup import (
    ensure_collections,
    close_client,
    class_from_key,     # 'FB1'|'FB2'|'FB3' -> class name
    KEY_TO_CLASS,       # mapping of keys to class names
    get_glossary_collection,  # for structured glossary entries
    client,             # v4 client (we'll use it for direct inserts)
)
from app.services.embedder import embed_and_store  # existing path (kept)

load_dotenv()

ROOT = Path(__file__).resolve().parent
PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "docs"))

PDF_FB1_ENV = os.getenv("PDF_FB1")
PDF_FB2_ENV = os.getenv("PDF_FB2")
PDF_FB3_ENV = os.getenv("PDF_FB3")

BOT_KEYS = ["FB1", "FB2", "FB3"]

# Toggle the new hi-res extraction path (default ON)
USE_HIRES = os.getenv("USE_HIRES_EXTRACTION", "1") not in ("0", "false", "False")

# ------------------------ PDF text extraction (legacy, page-wise) ------------------------

def _read_pdf_pages(pdf_path: Path) -> List[str]:
    """
    Return a list of page texts.
    Tries pdfplumber (better layout/text), falls back to PyPDF2.
    NOTE: This is used mainly for glossary scraping (fast path).
    """
    texts: List[str] = []

    # Try pdfplumber first (better for text extraction)
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                texts.append(txt)
        if texts:
            return texts
    except Exception:
        texts = []  # reset and try fallback

    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            texts.append(txt)
    except Exception:
        return []

    return texts

# ------------------------ NEW: Layout-aware extraction + chunking ------------------------

def _extract_blocks_hires(pdf_path: Path):
    """
    Use unstructured partitioner with layout-aware parsing.
    Falls back to OCR-only if text layer is messy.
    """
    try:
        from unstructured.partition.pdf import partition_pdf  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "unstructured is not installed. "
            "Run: pip install \"unstructured[pdf,pytesseract]\" pillow"
        ) from e

    try:
        # Try high-res layout aware first
        return partition_pdf(
            filename=str(pdf_path),
            strategy="hi_res",
            infer_table_structure=True,
            include_page_breaks=True,
        )
    except Exception:
        # Fallback to OCR-only if needed
        return partition_pdf(
            filename=str(pdf_path),
            strategy="ocr_only",
            infer_table_structure=True,
            include_page_breaks=True,
        )

def _blocks_to_spans(blocks) -> List[Dict[str, Optional[str]]]:
    """
    Convert unstructured blocks into plain spans with page info.
    """
    spans: List[Dict[str, Optional[str]]] = []
    for el in blocks:
        txt = getattr(el, "text", None)
        if not txt:
            continue
        md = getattr(el, "metadata", None)
        page_no = getattr(md, "page_number", None)
        spans.append({"text": txt.strip(), "page": int(page_no) if page_no else None})
    return spans

def _clean_text(s: str) -> str:
    # Remove common cid garbage and normalize spaces/dashes/superscripts
    s = re.sub(r"\(\s*cid\s*:\s*\d+\s*\)", "", s, flags=re.I)
    s = s.replace("\u2013", "-").replace("\u2212", "-").replace("\u00A0", " ")
    s = s.replace("m³", "m3")
    return re.sub(r"[ \t]+", " ", s).strip()

def _chunk_spans(
    spans: List[Dict[str, Optional[str]]],
    max_len: int = 1000,
    overlap: int = 150,
) -> List[Dict[str, Optional[str]]]:
    """
    Simple length-based chunking across spans, preserving approximate page locality.
    """
    chunks: List[Dict[str, Optional[str]]] = []
    buf = ""
    first_page = None

    def flush():
        nonlocal buf, first_page
        if buf.strip():
            chunks.append({"text": buf.strip(), "page": first_page})
        buf = ""
        first_page = None

    for sp in spans:
        t = _clean_text(sp.get("text") or "")
        if not t:
            continue
        candidate = (t + "\n")
        if not buf:
            first_page = sp.get("page")

        if len(buf) + len(candidate) <= max_len:
            buf += candidate
        else:
            # flush current
            flush()
            # start new with overlap from previous
            tail = (buf[-overlap:] if buf else "")
            buf = (tail + candidate) if tail else candidate
            first_page = sp.get("page")

    flush()
    return [c for c in chunks if c.get("text")]

def _detect_room(text: str) -> Optional[str]:
    # Heuristic keywords; expand as you like
    t = text.lower()
    if "cnc-fräserei" in t or "cnc fräserei" in t or "cnc-fraeserei" in t or "cnc fraeserei" in t:
        return "CNC-Fräserei"
    if "werkzeug" in t and "schleif" in t:
        return "Werkzeug- & Schleifraum"
    if "elektro" in t or "schaltschrank" in t:
        return "Elektro / Schaltschrank"
    return None

def _detect_machine(text: str) -> Optional[str]:
    t = text.lower()
    if "handschleifer" in t:
        return "Handschleifer"
    if "bearbeitungszentrum" in t or "fräszentrum" in t or "fraeszentrum" in t:
        return "Bearbeitungszentrum"
    return None

def _ingest_hires_chunks(pdf_path: Path, class_name: str) -> int:
    """
    Extract hi-res chunks and upsert them directly (BM25-ready).
    No vectors required (your collection's vectorizer is None).
    Returns number of objects inserted.
    """
    print(f"🔎 hi-res extracting: {pdf_path.name}")
    blocks = _extract_blocks_hires(pdf_path)
    spans = _blocks_to_spans(blocks)
    chunks = _chunk_spans(spans, max_len=1000, overlap=150)
    if not chunks:
        print("ℹ️ hi-res produced no chunks; skipping.")
        return 0

    coll = client.collections.get(class_name)
    n = 0
    for ch in chunks:
        text = ch["text"] or ""
        page = int(ch["page"]) if ch.get("page") else 1
        props = {
            "text": text,
            "source": pdf_path.name,
            "page": page,
            "room": _detect_room(text) or "",
            "machine": _detect_machine(text) or "",
        }
        try:
            coll.data.insert(properties=props)
            n += 1
        except Exception as e:
            print(f"⚠️ insert failed (page {page}): {e}")
    print(f"📚 hi-res chunks inserted: {n}")
    return n

# ------------------------ Glossary (Abkürzungsverzeichnis) -------------------

LINE_RX = re.compile(
    r"(?im)^[\s\-•]*([A-ZÄÖÜ0-9\/\-]{2,})\s*(?:[:=–\-]\s+|\s+)(.+?)\s*$"
)
MEASUREMENT_EARLY_RX = re.compile(
    r"\b(mg\/?m[23]|dB\(A\)|\bdB\b|LAeq|LAFmax|%|°C|V|A)\b", re.I
)

def _is_definition_like(defn: str) -> bool:
    d = (defn or "").strip()
    if not d:
        return False
    if re.match(r"^[\d\.\,\s]+", d):  # starts numeric
        return False
    if MEASUREMENT_EARLY_RX.search(d[:32]):
        return False
    return True

def _extract_glossary_entries_from_pages(
    pages: List[str],
) -> List[Tuple[str, str, int]]:
    results: List[Tuple[str, str, int]] = []
    if not pages:
        return results

    # Identify likely glossary pages
    likely_glossary_idx: set[int] = set()
    for idx, text in enumerate(pages):
        tl = (text or "").lower()
        if "abkürzungsverzeichnis" in tl or "abkuerzungsverzeichnis" in tl or "glossar" in tl:
            likely_glossary_idx.add(idx)

    def scan_page(text: str, page_no_1b: int) -> Iterable[Tuple[str, str, int]]:
        clean = re.sub(r"\(\s*cid\s*:\s*\d+\s*\)", "", text or "", flags=re.I)
        for m in LINE_RX.finditer(clean):
            term = (m.group(1) or "").strip()
            definition = (m.group(2) or "").strip(" \u00a0:–-")
            if not term or not definition:
                continue
            if not _is_definition_like(definition):
                continue
            yield (term.upper(), definition, page_no_1b)

    # Pass 1: likely glossary pages (desc order)
    for idx in sorted(likely_glossary_idx, reverse=True):
        for triple in scan_page(pages[idx], idx + 1):
            results.append(triple)

    # Pass 2: all pages
    for idx, text in enumerate(pages):
        if idx in likely_glossary_idx:
            continue
        for triple in scan_page(text, idx + 1):
            results.append(triple)

    # Deduplicate by TERM, prefer higher page numbers
    best: Dict[str, Tuple[str, str, int]] = {}
    for term_upper, definition, page_no in results:
        prev = best.get(term_upper)
        if prev is None or page_no > prev[2]:
            best[term_upper] = (term_upper, definition, page_no)

    return list(best.values())

def _upsert_glossary_entries(
    pdf_path: Path,
    entries: List[Tuple[str, str, int]],
) -> int:
    if not entries:
        return 0
    gc = get_glossary_collection()
    count = 0
    for term_upper, definition, page_no in entries:
        try:
            gc.data.insert({
                "term": term_upper,
                "definition": definition,
                "source": pdf_path.name,
                "page": int(page_no),
            })
            count += 1
        except Exception as e:
            print(f"⚠️  Glossary insert failed for {term_upper}: {e}")
    return count

# ------------------------ PDF discovery & mapping ----------------------------

def _discover_pdfs(folder: Path) -> List[Path]:
    if not folder.exists():
        print(f"❌ PDF folder not found: {folder}")
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() == ".pdf"])

def _pick_by_env_or_pattern(pdfs: List[Path]) -> Dict[str, Optional[Path]]:
    mapping: Dict[str, Optional[Path]] = {"FB1": None, "FB2": None, "FB3": None}
    name_to_path = {p.name.lower(): p for p in pdfs}

    env_map = {"FB1": PDF_FB1_ENV, "FB2": PDF_FB2_ENV, "FB3": PDF_FB3_ENV}
    for key, env_path in env_map.items():
        if not env_path:
            continue
        cand = Path(env_path)
        if cand.is_absolute() and cand.exists():
            mapping[key] = cand
        else:
            lower = cand.name.lower()
            if lower in name_to_path:
                mapping[key] = name_to_path[lower]
            else:
                for nm, p in name_to_path.items():
                    if lower in nm:
                        mapping[key] = p
                        break

    for key in BOT_KEYS:
        if mapping[key] is not None:
            continue
        needle = key.lower()
        for p in pdfs:
            if needle in p.name.lower():
                mapping[key] = p
                break

    remaining = [p for p in pdfs if p not in mapping.values() and p.exists()]
    for key in BOT_KEYS:
        if mapping[key] is None and remaining:
            mapping[key] = remaining.pop(0)

    return mapping

# ------------------------ Ingestion orchestration ----------------------------

def _ingest(pdf_path: Path, class_name: str):
    """
    1) (Existing) Call your embedder pipeline (kept for backward-compat).
    2) (New) Hi-res extraction + direct inserts for dense BM25 coverage.
    3) Extract abbreviations/glossary into GlossaryEntry (structured).
    """
    os.environ["WEAVIATE_CLASS"] = class_name   # for backward-compatible embedder

    print(f"📥 Indexing: {pdf_path}  ->  {class_name}")
    try:
        embed_and_store(str(pdf_path))  # existing path (may create few coarse objects)
    except Exception as e:
        print(f"⚠️ embed_and_store failed (continuing with hi-res path): {e}")

    # New: hi-res extraction & direct upserts
    if USE_HIRES:
        try:
            inserted = _ingest_hires_chunks(pdf_path, class_name)
            if inserted == 0:
                print("ℹ️ No hi-res chunks inserted.")
        except Exception as e:
            print(f"⚠️ hi-res ingestion failed: {e}")
    else:
        print("ℹ️ USE_HIRES_EXTRACTION=0 → skipped hi-res ingestion.")

    # Glossary extraction (fast page-wise)
    try:
        pages = _read_pdf_pages(pdf_path)
        entries = _extract_glossary_entries_from_pages(pages)
        if entries:
            n = _upsert_glossary_entries(pdf_path, entries)
            print(f"🔤 Glossary: stored {n} entries from {pdf_path.name}")
        else:
            print(f"ℹ️ No glossary-like entries found in {pdf_path.name}")
    except Exception as e:
        print(f"⚠️  Glossary extraction failed for {pdf_path.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into isolated Weaviate classes and extract glossary entries.")
    parser.add_argument("pdf", nargs="?", help="Path to a PDF (two-arg mode).")
    parser.add_argument("target", nargs="?", help="FB1|FB2|FB3 or exact class name (two-arg mode).")
    args = parser.parse_args()

    try:
        ensure_collections()

        # ----- Two-arg mode -----
        if args.pdf and args.target:
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                print(f"❌ PDF not found: {pdf_path}")
                return

            t = args.target.strip()
            if t.upper() in KEY_TO_CLASS:
                class_name = class_from_key(t.upper())
            else:
                class_name = t  # treat as explicit class name

            _ingest(pdf_path, class_name)
            print("✅ Done.")
            return

        # ----- Directory mode (no args) -----
        pdfs = _discover_pdfs(PDF_DIR)
        if not pdfs:
            print(f"ℹ️ No PDFs in: {PDF_DIR}")
            return

        mapping = _pick_by_env_or_pattern(pdfs)
        print("🗂️  Ingestion plan:")
        for key in BOT_KEYS:
            cn = class_from_key(key)
            print(f"  - {key} -> {cn}: {mapping.get(key) or '(no file)'}")

        any_ingested = False
        for key in BOT_KEYS:
            pdf_path = mapping.get(key)
            if not pdf_path:
                print(f"⚠️  Skipping {key}: no PDF assigned.")
                continue
            class_name = class_from_key(key)
            _ingest(pdf_path, class_name)
            any_ingested = True

        print("✅ Done." if any_ingested else "ℹ️ Nothing ingested.")

    finally:
        try:
            close_client()
        except Exception:
            pass

if __name__ == "__main__":
    main()

# Usage:
# python load_documents.py "docs/2025-08-18_FB 1 NEU.pdf" Chatbot_FB1
# or, using the key:
# python load_documents.py "docs/2025-08-18_FB 1 NEU.pdf" FB1
# Toggle new path off if needed:
# USE_HIRES_EXTRACTION=0 python load_documents.py "docs/2025-08-18_FB 1 NEU.pdf" FB1
