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
    get_glossary_collection,  # NEW: for structured glossary entries
)
from app.services.embedder import embed_and_store  # unchanged


load_dotenv()

ROOT = Path(__file__).resolve().parent
PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "docs"))

PDF_FB1_ENV = os.getenv("PDF_FB1")
PDF_FB2_ENV = os.getenv("PDF_FB2")
PDF_FB3_ENV = os.getenv("PDF_FB3")

BOT_KEYS = ["FB1", "FB2", "FB3"]


# ------------------------ PDF text extraction helpers ------------------------

def _read_pdf_pages(pdf_path: Path) -> List[str]:
    """
    Return a list of page texts.
    Tries pdfplumber (best layout/text), falls back to PyPDF2.
    """
    texts: List[str] = []

    # Try pdfplumber first (better for text extraction)
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                texts.append(txt)
        # if at least 1 page extracted, we're good
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
        # give up, return empty list
        return []

    return texts


# ------------------------ Glossary (Abkürzungsverzeichnis) -------------------

LINE_RX = re.compile(
    r"(?im)^[\s\-•]*([A-ZÄÖÜ0-9\/\-]{2,})\s*(?:[:=–\-]\s+|\s+)(.+?)\s*$"
)
MEASUREMENT_EARLY_RX = re.compile(
    r"\b(mg\/?m[23]|dB\(A\)|\bdB\b|LAeq|LAFmax|%|°C|V|A)\b", re.I
)

def _is_definition_like(defn: str) -> bool:
    """
    Accept textual definitions; reject measurement-style leading content.
    """
    d = (defn or "").strip()
    if not d:
        return False
    # reject if starts numerically
    if re.match(r"^[\d\.\,\s]+", d):
        return False
    # reject if measurement tokens appear *early*
    if MEASUREMENT_EARLY_RX.search(d[:32]):
        return False
    return True


def _extract_glossary_entries_from_pages(
    pages: List[str],
) -> List[Tuple[str, str, int]]:
    """
    Parse glossary-like lines from all pages.
    Prefer pages that look like Abkürzungsverzeichnis/Glossar
    but still collect from any page to be robust.
    Returns list of (TERM_UPPER, DEFINITION, PAGE_NUMBER_1_BASED).
    """
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
        # scrub common cid noise
        clean = re.sub(r"\(\s*cid\s*:\s*\d+\s*\)", "", text or "", flags=re.I)
        for m in LINE_RX.finditer(clean):
            term = (m.group(1) or "").strip()
            definition = (m.group(2) or "").strip(" \u00a0:–-")
            if not term or not definition:
                continue
            if not _is_definition_like(definition):
                continue
            yield (term.upper(), definition, page_no_1b)

    # Pass 1: scan likely glossary pages first (sorted by descending page number)
    for idx in sorted(likely_glossary_idx, reverse=True):
        for triple in scan_page(pages[idx], idx + 1):
            results.append(triple)

    # Pass 2: broad scan (all pages), but still useful if headings were not ingested
    for idx, text in enumerate(pages):
        if idx in likely_glossary_idx:
            continue  # already scanned
        for triple in scan_page(text, idx + 1):
            results.append(triple)

    # Deduplicate by TERM, prefer higher page numbers (more likely last-page glossary)
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
    """
    Write entries into `GlossaryEntry` collection:
    { term: UPPER, definition, source: pdf_filename, page }
    Returns number of entries written.
    """
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
            # keep going; log minimally to stdout
            print(f"⚠️  Glossary insert failed for {term_upper}: {e}")
    return count


# ------------------------ Existing ingestion plumbing ------------------------

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


def _ingest(pdf_path: Path, class_name: str):
    # 1) Ingest normal RAG chunks (unchanged path)
    os.environ["WEAVIATE_CLASS"] = class_name   # for backward-compatible embedder
    print(f"📥 Indexing: {pdf_path}  ->  {class_name}")
    embed_and_store(str(pdf_path))

    # 2) Extract & upsert Glossary (Abkürzungsverzeichnis) entries
    #    This makes abbreviation lookup instant & reliable at query time.
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
        close_client()


if __name__ == "__main__":
    main()

# python load_documents.py "docs/2025-08-18_FB 1 NEU.pdf" Chatbot_FB1
# or, using the key:
# python load_documents.py "docs/2025-08-18_FB 1 NEU.pdf" FB1
