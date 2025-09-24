import os
import sys
from dotenv import load_dotenv

# allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.weaviate_setup import init_schema, close_client  # type: ignore
from app.services.embedder import embed_and_store  # type: ignore

load_dotenv()

PDF_DIR = os.getenv("PDF_DIR", os.path.join(os.path.dirname(__file__), "docs"))

def main():
    try:
        init_schema()
        if not os.path.exists(PDF_DIR):
            print(f"❌ PDF folder not found: {PDF_DIR}")
            return
        pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
        if not pdfs:
            print(f"ℹ️ No PDFs in: {PDF_DIR}")
            return
        for name in pdfs:
            path = os.path.join(PDF_DIR, name)
            print(f"📥 Indexing: {path}")
            embed_and_store(path)
        print("✅ Done.")
    finally:
        close_client()

if __name__ == "__main__":
    main()
