from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.services.rag import retrieve_answer
from app.services.weaviate_setup import (
    ensure_collections,
    KEY_TO_CLASS,
    close_client,
)

# --- NEW: session memory primitives -----------------------------------------
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import time

SESSION_TTL_SECONDS = 60 * 60  # 1 hour; adjust as needed

@dataclass
class SessionState:
    last_room: Optional[str] = None
    last_machine: Optional[str] = None
    last_noise_kind: Optional[str] = None
    qas: List[tuple[str, str]] = field(default_factory=list)
    touched_at: float = field(default_factory=lambda: time.time())

SESSIONS: Dict[str, SessionState] = {}

def _get_state(session_id: str) -> SessionState:
    now = time.time()
    st = SESSIONS.get(session_id)
    if not st:
        st = SessionState()
        SESSIONS[session_id] = st
    st.touched_at = now
    # TTL cleanup
    for sid, s in list(SESSIONS.items()):
        if now - s.touched_at > SESSION_TTL_SECONDS:
            SESSIONS.pop(sid, None)
    return st

# Try to import fine-grained detectors from rag; fall back gracefully if absent
try:
    from app.services.rag import detect_room_from_question, pick_machine_from_question  # type: ignore
except Exception:
    def detect_room_from_question(_q: str) -> Optional[str]: return None  # type: ignore
    def pick_machine_from_question(_q: str) -> Optional[str]: return None  # type: ignore

def _detect_noise_kind(q: str) -> Optional[str]:
    ql = q.lower()
    if any(k in ql for k in ["laeq", "leq", "durchschnitt", "average"]): return "laeq"
    if any(k in ql for k in ["peak", "lafmax", "spitze", "spitzenwert", "maximal"]): return "peak"
    if any(k in ql for k in ["grenzwert", "expositionsgrenzwert", "am ohr", "ohr"]): return "limit"
    if any(k in ql for k in ["raum", "raumpegel", "im raum", "hintergrundpegel"]): return "room"
    return None

def _apply_session_context(raw_q: str, st: SessionState) -> str:
    q = raw_q.strip()
    asked_room = detect_room_from_question(q) or st.last_room
    asked_machine = pick_machine_from_question(q) or st.last_machine
    nk = _detect_noise_kind(q) or st.last_noise_kind

    prefix = []
    if asked_room:    prefix.append(asked_room)
    if asked_machine: prefix.append(asked_machine)
    if nk == "laeq":    prefix.append("LAeq")
    elif nk == "peak":  prefix.append("Peak LAFmax")
    elif nk == "limit": prefix.append("Expositionsgrenzwert am Ohr")
    elif nk == "room":  prefix.append("Raumpegel")

    return (": ".join(prefix) + ": " + q) if prefix else q

def _update_session_from_question_and_answer(q: str, a: str, st: SessionState):
    room = detect_room_from_question(q)
    mach = pick_machine_from_question(q)
    nk = _detect_noise_kind(q)

    if room: st.last_room = room
    if mach: st.last_machine = mach
    if nk:   st.last_noise_kind = nk

    st.qas.append((q, a))
    if len(st.qas) > 8:
        st.qas[:] = st.qas[-8:]
# -----------------------------------------------------------------------------


class ChatRequest(BaseModel):
    question: str
    target: str | None = None
    session_id: str | None = None  # NEW


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
    st = _get_state(req.session_id or "default")
    augmented = _apply_session_context(req.question, st)
    answer = retrieve_answer(augmented, target=req.target)
    _update_session_from_question_and_answer(req.question, answer, st)
    return {"answer": answer}

# Dedicated endpoints
@app.post("/chat/fb1")
async def chat_fb1(req: ChatRequest):
    st = _get_state(req.session_id or "default")
    augmented = _apply_session_context(req.question, st)
    answer = retrieve_answer(augmented, target="FB1")
    _update_session_from_question_and_answer(req.question, answer, st)
    return {"answer": answer}

@app.post("/chat/fb2")
async def chat_fb2(req: ChatRequest):
    st = _get_state(req.session_id or "default")
    augmented = _apply_session_context(req.question, st)
    answer = retrieve_answer(augmented, target="FB2")
    _update_session_from_question_and_answer(req.question, answer, st)
    return {"answer": answer}

@app.post("/chat/fb3")
async def chat_fb3(req: ChatRequest):
    st = _get_state(req.session_id or "default")
    augmented = _apply_session_context(req.question, st)
    answer = retrieve_answer(augmented, target="FB3")
    _update_session_from_question_and_answer(req.question, answer, st)
    return {"answer": answer}

# Utility
@app.get("/bots")
async def list_bots():
    return {"bots": KEY_TO_CLASS}

@app.get("/health")
async def health():
    return {"status": "ok"}

# --- NEW: reset current session memory ---------------------------------------
@app.post("/session/reset")
async def reset_session(session_id: str | None = Query(None, description="session id to reset")):
    sid = session_id or "default"
    SESSIONS.pop(sid, None)
    return {"ok": True}
# -----------------------------------------------------------------------------


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
