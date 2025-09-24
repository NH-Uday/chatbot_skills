import os
import math
import pdfplumber
from typing import List, Dict, Iterable
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

from app.services.weaviate_setup import init_schema, client
from app.services.weaviate_setup import CLASS_NAME
from app.services.format_math_equation import format_equations_for_mathjax

load_dotenv()
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
PDF_DIR = os.getenv("PDF_DIR", "docs")
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

oa = OpenAI()

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

def embed_and_store(pdf_path: str):
    init_schema()
    coll = client.collections.get(CLASS_NAME)

    docs = _read_pdf(pdf_path)
    basename = os.path.basename(pdf_path)

    # Build chunks with page metadata
    raw_chunks = []
    for d in docs:
        for ch in _chunk_text(d["text"], model="gpt-4o-mini"):  # tokenization baseline
            # normalize math for better downstream rendering
            pretty = format_equations_for_mathjax(ch)
            raw_chunks.append({"text": pretty, "source": basename, "page": d["page"]})

    if not raw_chunks:
        print(f"⚠️ No text in {basename}")
        return

    # Batch insert with manual vectors
    batch_size = 64
    for i in range(0, len(raw_chunks), batch_size):
        batch = raw_chunks[i : i + batch_size]
        vectors = _embed_texts([b["text"] for b in batch])

        with coll.batch.dynamic() as b:
            for item, vec in zip(batch, vectors):
                b.add_object(
                    properties=item,
                    vector=vec,
                )

    print(f"✅ Indexed {len(raw_chunks)} chunks from {basename}")

def load_all_pdfs():
    if not os.path.exists(PDF_DIR):
        print(f"❌ Folder not found: {PDF_DIR}")
        return
    init_schema()
    for f in os.listdir(PDF_DIR):
        if f.lower().endswith(".pdf"):
            embed_and_store(os.path.join(PDF_DIR, f))

    client.close()  

