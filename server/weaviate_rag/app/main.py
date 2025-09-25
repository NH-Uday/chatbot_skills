from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.services.rag import retrieve_answer
from app.services.weaviate_setup import (
    ensure_collections,
    KEY_TO_CLASS,
    close_client,
)

class ChatRequest(BaseModel):
    question: str
    target: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collections()
    yield
    close_client()

app = FastAPI(title="PDF RAG Chat", version="1.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Generic endpoint (accepts 'target')
@app.post("/chat")
async def chat(req: ChatRequest):
    answer = retrieve_answer(req.question, target=req.target)
    return {"answer": answer}

# Dedicated endpoints
@app.post("/chat/fb1")
async def chat_fb1(req: ChatRequest):
    answer = retrieve_answer(req.question, target="FB1")
    return {"answer": answer}

@app.post("/chat/fb2")
async def chat_fb2(req: ChatRequest):
    answer = retrieve_answer(req.question, target="FB2")
    return {"answer": answer}

@app.post("/chat/fb3")
async def chat_fb3(req: ChatRequest):
    answer = retrieve_answer(req.question, target="FB3")
    return {"answer": answer}

# Utility
@app.get("/bots")
async def list_bots():
    return {"bots": KEY_TO_CLASS}

@app.get("/health")
async def health():
    return {"status": "ok"}

from fastapi import Query
from app.services.rag import _resolve_collection, _vector_search, _bm25_search, _embed
from weaviate.classes.query import MetadataQuery

@app.get("/debug/top")
async def debug_top(q: str = Query(..., description="question"),
                    target: str = Query("FB1"),
                    k: int = Query(8)):
    coll = _resolve_collection(target)
    qvec = _embed(q)

    vec = _vector_search(coll, qvec, k=k)
    bm  = _bm25_search(coll, q, k=k)

    def pack(res, kind):
        out = []
        if not res: return out
        for o in res.objects:
            props = getattr(o, "properties", {}) or {}
            out.append({
                "kind": kind,
                "uuid": o.uuid,
                "distance": getattr(getattr(o, "metadata", None), "distance", None),
                "score": getattr(getattr(o, "metadata", None), "score", None),
                "source": props.get("source"),
                "page": props.get("page"),
                "text": (props.get("text") or "")[:260],
            })
        return out

    return {"target": target, "vec": pack(vec, "vec"), "bm25": pack(bm, "bm25")}
