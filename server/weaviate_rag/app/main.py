from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from app.services.rag import retrieve_answer_with_meta
from app.services.weaviate_setup import (
    ensure_collections,
    KEY_TO_CLASS,
    close_client,
)

from typing import Optional, Dict, List
from dataclasses import dataclass, field
import time
import re

SESSION_TTL_SECONDS = 60 * 60  

@dataclass
class SessionState:
    last_room: Optional[str] = None
    last_machine: Optional[str] = None
    last_noise_kind: Optional[str] = None
    qas: List[tuple[str, str]] = field(default_factory=list)
    touched_at: float = field(default_factory=lambda: time.time())
    pending_glossary_term: Optional[str] = None
    last_glossary_term: Optional[str] = None

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

# --- Special-case list-request blocker --------------------------------------
SPECIAL_LIST_REPLY = (
    "Tut mir leid, diese Frage kann ich dir so nicht beantworten. "
    "Bitte gib einen konkreten Arbeitsbereich oder eine Tätigkeit an, "
    "auf die sich die Exposition bezieht."
)

_LIST_VERBS = r"(?:auflisten|liste(?:n)?|nennen|zeigen|aufzählen|aufzaehlen)"
_ALL = r"(?:alle|sämtliche|saemtliche)"
_EXPO = r"(?:exposition(?:en)?)"
_COLOR = r"(?:farblich\s+markiert(?:e|en)?)"
_COLOR_WORDS = r"(?:rot(?:e|en)?|gelb(?:e|en)?|gr[üu]n(?:e|en)?)"
_AREAS = r"(?:arbeitsbereich(?:e)?|tätigkeit(?:en)?|taetigkeit(?:en)?)"
_ROOM_HINTS  = r"(?:im\s+bereich|bereich|raum|cnc|fr[äa]serei|fraeserei|werkzeug|schleif|montage|pr[üu]fstand|pruefstand)"
_AUFF = r"(?:auffälligkeit(?:en)?|auffaelligkeit(?:en)?|auffällige\s+exposition(?:en)?|auffaellige\s+exposition(?:en)?)"
_ITEMS = r"(?:exposition(?:en)?|expostion(?:en)?|stelle(?:n)?)"
_LIST_ANY = r"(?:auflisten|aufz(?:ä|ae)hlen|liste(?:n)?|liste\s+der|lister)"
_DOCUMENT_HINTS = r"(?:im\s+dokument|dokument|pdf|datei)"


_SPECIAL_PATTERNS = [
    # a) very broad “alle … Exposition(en) …”
    re.compile(rf"\b{_ALL}.*{_EXPO}\b.*(?:{_LIST_VERBS})?", re.I),
    re.compile(rf"\b(?:{_LIST_VERBS}).*{_ALL}.*{_EXPO}\b", re.I),

    # b) “alle farblich markierten …”
    re.compile(rf"\b{_ALL}.*{_COLOR}.*(?:{_EXPO}|{_AREAS})\b.*(?:{_LIST_VERBS})?", re.I),
    re.compile(rf"\b(?:{_LIST_VERBS}).*{_ALL}.*{_COLOR}.*(?:{_EXPO}|{_AREAS})\b", re.I),

    # c) “alle Auffälligkeiten …”
    re.compile(rf"\b{_ALL}.*{_AUFF}\b.*(?:{_LIST_VERBS})?", re.I),
    re.compile(rf"\b(?:{_LIST_VERBS}).*{_ALL}.*{_AUFF}\b", re.I),

    # d) color-scoped lists, with or without “alle”
    re.compile(rf"\b(?:{_LIST_VERBS}).*\b{_COLOR_WORDS}\b.*\b{_ITEMS}\b", re.I),
    re.compile(rf"\b{_COLOR_WORDS}\b.*\b{_ITEMS}\b.*(?:{_LIST_VERBS})?", re.I),

    # e) “im Bereich …” scoped lists of Expositionen
    re.compile(rf"\b(?:{_LIST_VERBS}).*\b{_EXPO}\b.*\b{_ROOM_HINTS}\b", re.I),
    re.compile(rf"\b{_EXPO}\b.*\b{_ROOM_HINTS}\b.*(?:{_LIST_VERBS})?", re.I),

    # f) “im Dokument” / PDF-scoped lists
    re.compile(rf"\b(?:{_LIST_VERBS})\b.*\b{_ITEMS}\b.*\b{_DOCUMENT_HINTS}\b", re.I),
    re.compile(rf"\b{_ITEMS}\b.*\b{_DOCUMENT_HINTS}\b.*(?:{_LIST_VERBS})?", re.I),

    # g) Room/area scopes (unchanged logic, but use _ITEMS to also cover “Stellen”)
    re.compile(rf"\b(?:{_LIST_VERBS}).*\b{_ITEMS}\b.*\b{_ROOM_HINTS}\b", re.I),
    re.compile(rf"\b{_ITEMS}\b.*\b{_ROOM_HINTS}\b.*(?:{_LIST_VERBS})?", re.I),

    # h) split-verb forms like “listen ... auf” (with optional “Sie”)
    re.compile(rf"\blisten(?:\s+sie)?\b.*\b{_COLOR_WORDS}\b.*\b{_ITEMS}\b.*\bauf\b", re.I),
    re.compile(rf"\blisten(?:\s+sie)?\b.*\b{_ITEMS}\b.*\bauf\b", re.I),

    # i) "Gib/Gebe mir eine Liste ..." style requests (list-style)
    re.compile(r"\b(?:gib|gebe)\s+(?:mir|uns)?\s+(?:eine\s+)?liste\b.*\b{_ITEMS}\b", re.I),
    re.compile(r"\b(?:gib|gebe)\s+(?:mir|uns)?\s+(?:eine\s+)?liste\b.*\bexposition", re.I),
    re.compile(r"\bliste\s+der\b.*\bexposition", re.I),

    # j) modal/indirect forms like “ob du … auflisten kannst / kannst du … auflisten …”
    re.compile(rf"\b(?:ob\s+)?(?:kann|kannst|können|koennen|würde|wuerde|würdest|wuerdest)\b.*\b{_ITEMS}\b.*\b(?:auflisten|aufzaehlen|aufzählen)\b", re.I),
    re.compile(rf"\b(?:ob\s+)?(?:kann|kannst|können|koennen|würde|wuerde|würdest|wuerdest)\b.*\b(?:auflisten|aufzaehlen|aufzählen)\b.*\b{_ITEMS}\b", re.I),

    # broad, order-agnostic “list … items” catch-alls
    re.compile(rf"\b{_LIST_ANY}\b.*\b{_ITEMS}\b", re.I),
    re.compile(rf"\b{_ITEMS}\b.*\b{_LIST_ANY}\b", re.I),

    # “Gib/Gebe mir eine Liste/Lister …” noun-phrase style
    re.compile(rf"\b(?:gib|gebe)\s+(?:mir|uns)?\s+(?:eine\s+)?(?:liste|lister)\b.*\b{_ITEMS}\b", re.I),

    # modal/indirect forms (“ob du … auflisten kannst / kannst du … auflisten …”)
    re.compile(
        rf"\b(?:ob\s+)?(?:kann|kannst|können|koennen|würde|wuerde|würdest|wuerdest)\b"
        rf".*\b{_ITEMS}\b.*\b(?:auflisten|aufz(?:ä|ae)hlen)\b",
        re.I
    ),


]


def _is_blocked_list_request(raw_q: str) -> bool:
    q = (raw_q or "").strip().lower()
    q = re.sub(r"^[^:]{1,40}:\s*", "", q)  # keep your existing prefix trim

    return any(p.search(q) for p in _SPECIAL_PATTERNS)

def _apply_session_context(raw_q: str, st: SessionState) -> str:
    q = raw_q.strip()

    # --- Case 1: "Ja"/"Yes" after glossary prompt ---
    if re.fullmatch(r"(ja|yes)[\.\!\s]*", q, re.I) and st.pending_glossary_term:
        q = f"{st.pending_glossary_term} Abkürzungsverzeichnis"

    # --- Case 2: "Nein"/"No" cancels glossary clarification ---
    if re.fullmatch(r"(nein|no)[\.\!\s]*", q, re.I):
        st.pending_glossary_term = None
        # fall through → regular context search

    # --- Case 3: Bare glossary-like term (word only) ---
    if re.fullmatch(r"[A-ZÄÖÜa-zäöüß0-9\/\-]{2,}$", q, re.I):
        # Ask if user wants glossary info about that term
        st.pending_glossary_term = q.upper()
        return (
            f"Meinen Sie das Abkürzungsverzeichnis für '{q.upper()}'?\n"
            f"- Ja, **{q.upper()} Abkürzungsverzeichnis**\n"
            f"- Nein, Messungen/Kontext anzeigen"
        )

    # --- Existing prefixing logic (rooms, noise kinds, etc.) ---
    asked_room = detect_room_from_question(q) or st.last_room
    asked_machine = pick_machine_from_question(q) or st.last_machine
    nk = _detect_noise_kind(q) or st.last_noise_kind

    prefix = []
    if asked_room:
        prefix.append(asked_room)
    if asked_machine:
        prefix.append(asked_machine)
    if nk == "laeq":
        prefix.append("LAeq")
    elif nk == "peak":
        prefix.append("Peak LAFmax")
    elif nk == "limit":
        prefix.append("Expositionsgrenzwert am Ohr")
    elif nk == "room":
        prefix.append("Raumpegel")

    return (": ".join(prefix) + ": " + q) if prefix else q

def _update_session_from_question_and_answer(q: str, a: str | None, st: SessionState):
    room = detect_room_from_question(q)
    mach = pick_machine_from_question(q)
    nk = _detect_noise_kind(q)

    if room:
        st.last_room = room
    if mach:
        st.last_machine = mach
    if nk:
        st.last_noise_kind = nk

    # --- Safely handle None answers & glossary tracking ---
    if isinstance(a, str) and a:
        # Capture glossary clarification prompt
        m = re.search(r"Abk(?:ürz|uerz)ungsverzeichnis\s+für\s+'([^']+)'", a, re.I)
        if m:
            st.pending_glossary_term = m.group(1).strip().upper()

        # Clear pending if a definition was given
        if re.search(r"'\s*([A-ZÄÖÜ0-9\/\-]{2,})\s*'\s*(?:bedeutet|means)\s*:", a, re.I):
            st.pending_glossary_term = None

        # Track last glossary term seen in answer (helps handle bare terms)
        g = re.search(r"'([A-ZÄÖÜa-zäöüß0-9\/\-]+)'\s*(?:bedeutet|means)", a)
        if g:
            st.last_glossary_term = g.group(1).strip()
    else:
        # If answer is None or empty, clear pending glossary safely
        st.pending_glossary_term = None

    st.qas.append((q, a or ""))
    if len(st.qas) > 8:
        st.qas[:] = st.qas[-8:]


CONFIRM_RX = re.compile(
    r"^\s*ist\s+d(?:as|er|ie)\b.*\bin\s+der\b\s*(?P<room>[^?!.]+)\??\s*$",
    re.I
)

def _try_coref_confirmation(raw_q: str, st: SessionState) -> str | None:
    """
    Intercept follow-ups like:
      "Ist das der Handschleifer in der CNC-Fräserei?"
    and answer Yes/No using session meta without hitting retrieval.
    """
    m = CONFIRM_RX.search(raw_q or "")
    if not m:
        return None
    asked_room = (m.group("room") or "").strip()
    last_room = (st.last_room or "").strip()
    last_machine = (st.last_machine or "").strip()

    if not last_room and not last_machine:
        return None  # nothing to confirm

    if asked_room and last_room:
        if asked_room.lower() == last_room.lower():
            if last_machine:
                return f"Ja – bezog sich auf **{last_machine}** in **{last_room}**."
            return f"Ja – bezog sich auf **{last_room}**."
        else:
            if last_machine:
                return f"Nein – der zuletzt genannte Wert bezog sich auf **{last_machine}** in **{last_room}**, nicht auf **{asked_room}**."
            return f"Nein – der zuletzt genannte Wert bezog sich auf **{last_room}**, nicht auf **{asked_room}**."
    return None


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

# Helper to call retriever, then update meta-driven session state
def _answer_and_update(req: ChatRequest, target: str | None, st: SessionState):
    augmented = _apply_session_context(req.question, st)

    # ⬇️ use the meta-enabled retriever
    answer, meta_room, meta_machine = retrieve_answer_with_meta(augmented, target=target)

    # Update session memory from META first (what the model actually used)
    if meta_room:
        st.last_room = meta_room
    if meta_machine:
        st.last_machine = meta_machine

    # Then run the usual lightweight updater (from user's raw question & answer)
    _update_session_from_question_and_answer(req.question, answer, st)

    return answer

# Generic endpoint (accepts 'target')
@app.post("/chat")
async def chat(req: ChatRequest):
    # --- Early guard for “list all …” special cases
    if _is_blocked_list_request(req.question):
        return {"answer": SPECIAL_LIST_REPLY}

    st = _get_state(req.session_id or "default")

    # 1) Intercept yes/no coref confirmation
    coref = _try_coref_confirmation(req.question, st)
    if coref:
        _update_session_from_question_and_answer(req.question, coref, st)
        return {"answer": coref}

    # 2) Normal flow
    answer = _answer_and_update(req, req.target, st)
    return {"answer": answer}

# Dedicated endpoints
@app.post("/chat/fb1")
async def chat_fb1(req: ChatRequest):
    if _is_blocked_list_request(req.question):
        return {"answer": SPECIAL_LIST_REPLY}

    st = _get_state(req.session_id or "default")

    coref = _try_coref_confirmation(req.question, st)
    if coref:
        _update_session_from_question_and_answer(req.question, coref, st)
        return {"answer": coref}

    answer = _answer_and_update(req, "FB1", st)
    return {"answer": answer}

@app.post("/chat/fb2")
async def chat_fb2(req: ChatRequest):
    if _is_blocked_list_request(req.question):
        return {"answer": SPECIAL_LIST_REPLY}

    st = _get_state(req.session_id or "default")

    coref = _try_coref_confirmation(req.question, st)
    if coref:
        _update_session_from_question_and_answer(req.question, coref, st)
        return {"answer": coref}

    answer = _answer_and_update(req, "FB2", st)
    return {"answer": answer}

@app.post("/chat/fb3")
async def chat_fb3(req: ChatRequest):
    if _is_blocked_list_request(req.question):
        return {"answer": SPECIAL_LIST_REPLY}

    st = _get_state(req.session_id or "default")

    coref = _try_coref_confirmation(req.question, st)
    if coref:
        _update_session_from_question_and_answer(req.question, coref, st)
        return {"answer": coref}

    answer = _answer_and_update(req, "FB3", st)
    return {"answer": answer}

# Utility
@app.get("/bots")
async def list_bots():
    return {"bots": KEY_TO_CLASS}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/session/reset")
async def reset_session(session_id: str | None = Query(None, description="session id to reset")):
    sid = session_id or "default"
    SESSIONS.pop(sid, None)
    return {"ok": True}


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
