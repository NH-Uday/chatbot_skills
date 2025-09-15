import os
from app.services.embedder import embed_and_store
from app.services.weaviate_setup import init_schema, client


PDF_FOLDER = os.path.join(os.path.dirname(__file__), "docs")

def load_all_pdfs():
    if not os.path.exists(PDF_FOLDER):
        print(f"❌ Folder not found: {PDF_FOLDER}")
        return

    init_schema()

    for file in os.listdir(PDF_FOLDER):
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_FOLDER, file)
            embed_and_store(pdf_path)
            
    client.close()

if __name__ == "__main__":
    load_all_pdfs()
