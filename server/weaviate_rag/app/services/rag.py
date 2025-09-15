import os
from dotenv import load_dotenv
from openai import OpenAI
from app.services.weaviate_setup import client

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
WEAVIATE_COLLECTION = "LectureChunk"

oa = OpenAI(api_key=OPENAI_API_KEY)

# --- system prompt for the assistant ---
SYSTEM_PROMPT = """
You are a helpful and inspiring study assistant for engineering/science topics.
Use ONLY the provided context. If the answer is not in the context, say "This question is out of my knowledge domain."

Answer every question very straight forward.
"""

def _retrieve_chunks(query: str, k: int = 6):
    embedded_query = oa.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    ).data[0].embedding

    coll = client.collections.get(WEAVIATE_COLLECTION)

    res = coll.query.hybrid(
        query=query,
        vector=embedded_query,
        limit=k,
        alpha=0.5
    )

    # ✅ Now returning full metadata
    return [
        {
            "text": obj.properties.get("text", "").strip(),
            "source": obj.properties.get("source", "unknown"),
            "page": obj.properties.get("page", "?")
        }
        for obj in res.objects
    ]


def retrieve_answer(question: str) -> str:
    retrieved_chunks = _retrieve_chunks(question)
    chunks = [c["text"] for c in retrieved_chunks if c["text"]]

    if not chunks:
        return (

            "I don't know based on the provided materials."
        )

    context = "\n\n---\n\n".join(chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nContext:\n{context}"}
    ]

    resp = oa.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0
    )

    answer = resp.choices[0].message.content

    sources = "\n".join(
        f"- From **{c['source']}**, page {c['page']}" for c in retrieved_chunks
    )

    #return f"{answer}\n\n---\n**Sources:**\n{sources}"
    return answer

