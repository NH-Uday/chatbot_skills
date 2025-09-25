import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.weaviate_setup import (
    ensure_collections,
    close_client,
    class_from_key,     # 'FB1'|'FB2'|'FB3' -> class name
    KEY_TO_CLASS,       # mapping of keys to class names
)
from app.services.embedder import embed_and_store  # unchanged

load_dotenv()

ROOT = Path(__file__).resolve().parent
PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "docs"))

PDF_FB1_ENV = os.getenv("PDF_FB1")
PDF_FB2_ENV = os.getenv("PDF_FB2")
PDF_FB3_ENV = os.getenv("PDF_FB3")

BOT_KEYS = ["FB1", "FB2", "FB3"]


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
    os.environ["WEAVIATE_CLASS"] = class_name   # for backward-compatible embedder
    print(f"📥 Indexing: {pdf_path}  ->  {class_name}")
    embed_and_store(str(pdf_path))


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into isolated Weaviate classes.")
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


#python load_documents.py "docs/2025-08-18_FB 1 NEU.pdf" Chatbot_FB1
# or, using the key:
#python load_documents.py "docs/2025-08-18_FB 1 NEU.pdf" FB1
