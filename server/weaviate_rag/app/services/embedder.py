import os
import pdfplumber
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from app.services.weaviate_setup import init_schema, client
from app.services.format_math_equation import format_equations_for_mathjax


load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PDF_FOLDER = "docs" 
CHUNK_SIZE = 500  # words
CHUNK_OVERLAP = 50

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def embed_and_store(pdf_path: str):
    filename = os.path.basename(pdf_path)
    collection = client.collections.get("LectureChunk")

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            chunks = chunk_text(page_text)

            for chunk in chunks:
                if not chunk.strip():
                    continue

                embedding = openai.embeddings.create(
                    input=chunk,
                    model="text-embedding-3-small"
                ).data[0].embedding

                collection.data.insert(
                    properties={
                        "text": chunk,
                        "source": filename,
                        "page": page_number  # ✅ new metadata
                    },
                    vector=embedding,
                )

    print(f"✅ Finished indexing: {filename}")
    
def load_all_pdfs():
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Folder not found: {PDF_FOLDER}")
        return

    init_schema()

    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            embed_and_store(pdf_path)

    client.close()  # 🔒 Close connection after all inserts

