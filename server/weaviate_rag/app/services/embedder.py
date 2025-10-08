import os
import math
import re
from typing import Optional
import pdfplumber
from typing import List, Dict, Iterable
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

from app.services.weaviate_setup import (
    ensure_collections,
    get_collection,
    class_from_key,
    client,
)
from app.services.format_math_equation import format_equations_for_mathjax

load_dotenv()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
PDF_DIR = os.getenv("PDF_DIR", "docs")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

oa = OpenAI()

# ---------------------------
# NEW: Room & Machine helpers
# ---------------------------

# Matches lines like: "06:55 Uhr – CNC-Fräserei"
ROOM_HDR = re.compile(r"^\s*\d{2}:\d{2}\s*Uhr\s*[–-]\s*(.+)$")

# Extend/adjust these keyword->label rules to your data as needed
MACHINE_RULES: List[tuple[re.Pattern, str]] = [
    (re.compile(r"\bBearbeitungszentrum|\bCNC\b|\bFräs", re.I), "CNC-Bearbeitungszentrum"),
    (re.compile(r"\bSchleif|Handschleif", re.I), "Schleifen/Handschleifer"),
    (re.compile(r"\bSchrumpfger", re.I), "Schrumpfgerät"),
    (re.compile(r"\bPrüfstand", re.I), "Prüfstand"),
    (re.compile(r"\bLOTO|Schaltschrank|400\s*V|63\s*A", re.I), "Schaltschrank/LOTO"),
    (re.compile(r"\bAbsaug|KSS|Aerosol|Nebel|Nassschleif", re.I), "Absaugung/KSS"),
]

def _detect_room_from_page_text(text: str, fallback: Optional[str] = None) -> str:
    
    for line in text.splitlines():
        m = ROOM_HDR.match(line.strip())
        if m:
            return m.group(1).strip()
    return fallback or "Unklar"

def _detect_machine_from_chunk(snippet: str) -> str:
    
    for rx, label in MACHINE_RULES:
        if rx.search(snippet):
            return label
    return "Allgemein"


def _read_pdf_with_rooms(path: str) -> List[Dict]:
    
    out: List[Dict] = []
    with pdfplumber.open(path) as pdf:
        current_room: Optional[str] = None
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            if not raw.strip():
                continue

            lines = raw.splitlines()
            sections: List[Tuple[str, List[str]]] = []  # [(room, lines)]
            room_here: Optional[str] = None
            buf: List[str] = []

            def flush_buffer():
                if buf:
                    text_block = "\n".join(buf).strip()
                    if text_block:
                        sections.append((room_here or current_room or "Unklar", [text_block]))

            for ln in lines:
                m = ROOM_HDR.match(ln.strip())
                if m:
                    # new section header → flush previous buffer
                    flush_buffer()
                    # update room for this section
                    room_here = m.group(1).strip()
                    # carry forward for subsequent sections across pages
                    current_room = room_here
                    buf = []
                else:
                    buf.append(ln)

            # leftover lines on the page
            flush_buffer()

            # Build output records
            for room_label, blocks in sections:
                out.append({"page": i, "text": "\n".join(blocks), "room": room_label})

            # If no header found at all, still emit the page with fallback room
            if not sections:
                out.append({"page": i, "text": raw, "room": current_room or "Unklar"})

    return out


# ---------------------------
# (Existing) helpers — unchanged
# ---------------------------

def _read_pdf(path: str) -> List[Dict]:
    out = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                out.append({"page": i, "text": text})
    return out

def _tokenize(text: str, model: str) -> List[int]:
    enc = tiktoken.encoding_for_model(model) if model in tiktoken.list_encoding_names() else tiktoken.get_encoding("cl100k_base")
    return enc.encode(text)

def _detok(tokens: List[int]) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    return enc.decode(tokens)

def _chunk_text(text: str, model: str, chunk_tokens=CHUNK_TOKENS, overlap=CHUNK_OVERLAP) -> Iterable[str]:
    toks = _tokenize(text, model)
    n = len(toks)
    if n <= chunk_tokens:
        yield _detok(toks)
        return
    step = chunk_tokens - overlap
    for start in range(0, n, step):
        window = toks[start : min(start + chunk_tokens, n)]
        yield _detok(window)
        if start + chunk_tokens >= n:
            break

def _embed_texts(texts: List[str]) -> List[List[float]]:
    resp = oa.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _target_class_from_env_or_key(target: Optional[str]) -> str:
    if target:
        t = target.strip()
        if t.upper() in {"FB1", "FB2", "FB3"}:
            return class_from_key(t.upper())
        return t
    return os.getenv("WEAVIATE_CLASS", "LectureChunk")

# ---------------------------
# Ingestion — minimally updated to include room & machine
# ---------------------------

def embed_and_store(pdf_path: str, target: Optional[str] = None):
    
    # Make sure all collections exist
    ensure_collections()

    # Pick collection at *runtime*
    class_name = _target_class_from_env_or_key(target)
    coll = get_collection(class_name)
    print(f"[embedder] Inserting into: {class_name}")

    # --- Read & chunk (room-aware) ---
    # NOTE: We keep _read_pdf for compatibility, but use the new room-aware reader here.
    docs = _read_pdf_with_rooms(pdf_path)
    basename = os.path.basename(pdf_path)

    raw_chunks: List[Dict] = []
    for d in docs:
        for ch in _chunk_text(d["text"], model="gpt-4o-mini"):
            pretty = format_equations_for_mathjax(ch)
            machine = _detect_machine_from_chunk(pretty)
            raw_chunks.append({
                "text": pretty,
                "source": basename,
                "page": d["page"],
                "room": d.get("room") or "Unklar",
                "machine": machine,
            })

    if not raw_chunks:
        print(f"⚠️ No text in {basename}")
        return

    # --- Embed & batch insert (manual vectors) ---
    batch_size = 64
    for i in range(0, len(raw_chunks), batch_size):
        batch = raw_chunks[i : i + batch_size]
        vectors = _embed_texts([b["text"] for b in batch])

        with coll.batch.dynamic() as b:
            for item, vec in zip(batch, vectors):
                b.add_object(
                    properties=item,   # includes text/source/page/room/machine
                    vector=vec,
                )

    print(f"✅ Indexed {len(raw_chunks)} chunks from {basename}")

def load_all_pdfs():

    if not os.path.exists(PDF_DIR):
        print(f"❌ Folder not found: {PDF_DIR}")
        return

    # Ensure collections exist (this replaces old 'init_schema' call)
    ensure_collections()

    for f in os.listdir(PDF_DIR):
        if f.lower().endswith(".pdf"):
            embed_and_store(os.path.join(PDF_DIR, f))

    client.close()

