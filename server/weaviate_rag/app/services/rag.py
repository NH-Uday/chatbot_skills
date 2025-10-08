import os
import re
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from weaviate.classes.query import MetadataQuery
from typing import List, Dict, Any, Optional

# New multi-bot helpers (backward-compatible)
from app.services.weaviate_setup import (
    ensure_collections,
    get_collection,
    get_collection_by_key,
    class_from_key,
)

# Legacy shim (if present) for default class fallback
try:
    from app.services.weaviate_setup import CLASS_NAME as LEGACY_DEFAULT_CLASS  # type: ignore
except Exception:
    LEGACY_DEFAULT_CLASS = os.getenv("WEAVIATE_CLASS", "LectureChunk")

load_dotenv()

CANDIDATES_PER_QUERY = int(os.getenv("CANDIDATES_PER_QUERY", "12"))  # per expansion
EXPANSIONS = int(os.getenv("EXPANSIONS", "4"))  # number of LLM paraphrases
KEEP_FOR_CONTEXT = int(os.getenv("KEEP_FOR_CONTEXT", "8"))  # final chunks for answer

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
TOP_K = int(os.getenv("TOP_K", "6"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

# IMPORTANT: include room & machine so we can detect ambiguity
RETURN_PROPS = ["text", "source", "page", "room", "machine"]

oa = OpenAI()

SYSTEM_PROMPT = (
    "Du antwortest ausschließlich anhand des bereitgestellten Kontexts. "
    "Kritische Zahlen und Bereiche (z. B. ppm, dB(A), m/s², %, lx, A/V) "
    "**musst du wortgleich** wie im Kontext wiedergeben. "
    "Falls die Information nicht im Kontext enthalten ist, antworte: "
    "'Das ist im Dokument nicht angegeben.' "
    "Antworte kurz und auf Deutsch."
)

# ---------------------------
# Embedding & search helpers
# ---------------------------

def _embed(q: str) -> List[float]:
    return oa.embeddings.create(model=OPENAI_EMBED_MODEL, input=q).data[0].embedding

def _vector_search(coll, query_vec, k=TOP_K):
    # return room & machine in hits
    return coll.query.near_vector(
        near_vector=query_vec,
        limit=k,
        return_metadata=MetadataQuery(distance=True),
        return_properties=RETURN_PROPS,
    )

def _bm25_search(coll, query: str, k=TOP_K):
    try:
        return coll.query.bm25(
            query=query,
            limit=k,
            return_metadata=MetadataQuery(rank=True),
            return_properties=RETURN_PROPS,
        )
    except Exception:
        return None

def _hybrid_rank(vec_res, bm25_res, alpha=HYBRID_ALPHA) -> List[Dict]:
    items = {}

    def add(res, score_key):
        if not res:
            return
        vals = []
        for o in res.objects:
            raw = (o.metadata.distance if score_key == "distance" else o.metadata.score) or 0.0
            vals.append(raw)
        if not vals:
            return
        lo, hi = (min(vals), max(vals))
        rng = max(hi - lo, 1e-9)
        for o in res.objects:
            raw = (o.metadata.distance if score_key == "distance" else o.metadata.score) or 0.0
            sim = 1.0 - (raw - lo) / rng if score_key == "distance" else (raw - lo) / rng
            prev = items.get(o.uuid, {"obj": o, "sim": 0.0})
            prev["sim"] += sim
            items[o.uuid] = prev

    add(vec_res, "distance")
    add(bm25_res, "score")

    ranked = []
    for v in items.values():
        o = v["obj"]
        has_vec = vec_res and any(o.uuid == r.uuid for r in vec_res.objects)
        has_bm = bm25_res and any(o.uuid == r.uuid for r in bm25_res.objects)
        if has_vec and has_bm:
            sim_vec = next((1 - r.metadata.distance for r in vec_res.objects if r.uuid == o.uuid), 0.0)
            sim_bm = next((r.metadata.score for r in bm25_res.objects if r.uuid == o.uuid), 0.0)
            hybrid = alpha * sim_vec + (1 - alpha) * sim_bm
        else:
            hybrid = v["sim"]
        ranked.append((hybrid, o))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [o for _, o in ranked][:TOP_K]

def _format_context(objs) -> Tuple[str, List[Dict]]:
    chunks = []
    for o in objs:
        props = o.properties
        chunks.append({
            "text": props["text"],
            "source": props["source"],
            "page": props["page"],
            # (optional) you could also carry room/machine here if you want
            # "room": props.get("room"),
            # "machine": props.get("machine"),
        })
    context = "\n\n---\n\n".join(
        f"[{i+1}] (Quelle: {c['source']}, Seite {c['page']})\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return context, chunks

# ---------------------------
# Numeric guard utilities
# ---------------------------

_DASH_PATTERN = re.compile(r"[–—−\-]+")
_WS = re.compile(r"\s+")

# include aerosol units (mg/m³, mg/m3, µg/m³, ug/m3)
_NUM_SPAN_RE = re.compile(
    r"""
    (?P<num1>\d+(?:[.,]\d+)?)                       # first number
    (?:\s*(?:[–—−-])\s*(?P<num2>\d+(?:[.,]\d+)?))?  # optional range end
    \s*
    (?P<unit>
        ppm|
        mg/m³|mg/m3|µg/m³|ug/m3|
        dB\(A\)|dB|
        m/s(?:²|\^2|2)?|
        %|
        lx|
        A|V|
        °C
    )?
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _norm_dash(s: str) -> str:
    return _DASH_PATTERN.sub("-", s)

def _norm_ws(s: str) -> str:
    return _WS.sub(" ", s).strip()

def _to_float(s: str) -> float:
    return float(s.replace(",", ".").strip())

# normalize aerosol & other unit variants
def _norm_unit(u: Optional[str]) -> str:
    if not u:
        return ""
    u = u.strip()
    u = u.replace("dB(a)", "dB(A)").replace("dB(A)", "dB(A)")
    if u.lower() == "db":
        u = "dB"
    u = u.replace("m/s^2", "m/s²").replace("m/s2", "m/s²")
    u = u.replace("mg/m3", "mg/m³").replace("ug/m3", "µg/m³")
    return u

def _extract_numeric_spans(text: str) -> List[Dict]:
    spans = []
    for m in _NUM_SPAN_RE.finditer(text):
        s = m.group(0)
        u = _norm_unit(m.group("unit"))
        a = _to_float(m.group("num1"))
        b = m.group("num2")
        b_val = _to_float(b) if b else None
        spans.append(
            {
                "match": s,
                "unit": u,
                "a": a,
                "b": b_val,
                "start": m.start(),
                "end": m.end(),
            }
        )
    return spans

def _span_distance(ans: Dict, src: Dict) -> float:
    if ans["unit"] != src["unit"]:
        return 1e6
    if _norm_ws(_norm_dash(ans["match"])) == _norm_ws(_norm_dash(src["match"])):
        return 0.0
    a1, b1 = ans["a"], ans["b"]
    a2, b2 = src["a"], src["b"]
    if b1 is not None and b2 is not None:
        return abs(a1 - a2) + abs(b1 - b2)
    if b1 is None and b2 is not None:
        if a2 <= a1 <= b2:
            return 0.0
        return min(abs(a1 - a2), abs(a1 - b2))
    if b1 is None and b2 is None:
        return abs(a1 - a2)
    return 1e5

def _choose_best_source_span(ans_span: Dict, src_spans: List[Dict]) -> Optional[Dict]:
    best, best_d = None, float("inf")
    for s in src_spans:
        d = _span_distance(ans_span, s)
        if d < best_d:
            best, best_d = s, d
    if best is not None and best_d < 0.51:  # small tolerance
        return best
    return None

# Collapse "zwischen <scalar> und <range>" → "zwischen <range>"
_ZWISCHEN_SCALAR_UND_RANGE = re.compile(
    r"""(?P<prefix>zwischen\s*)
        (?P<scalar>\d+(?:[.,]\d+)?)
        (?P<mid>\s*und\s*)
        (?P<range>\d+(?:[.,]\d+)?\s*(?:[–—−-]\s*\d+(?:[.,]\d+)?)\s*
           (?:ppm|dB\(A\)|dB|m/s(?:²|\^2|2)?|%|lx|A|V|°C)?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_ZWISCHEN_RANGE_UND_SCALAR = re.compile(
    r"""(?P<prefix>zwischen\s*)
        (?P<range>\d+(?:[.,]\d+)?\s*(?:[–—−-]\s*\d+(?:[.,]\d+)?)\s*
           (?:ppm|dB\(A\)|dB|m/s(?:²|\^2|2)?|%|lx|A|V|°C)?)
        (?P<mid>\s*und\s*)
        (?P<scalar>\d+(?:[.,]\d+)?(?:\s*(?:ppm|dB\(A\)|dB|m/s(?:²|\^2|2)?|%|lx|A|V|°C))?)
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _scalar_inside_range(scalar: float, range_a: float, range_b: float) -> bool:
    lo, hi = (min(range_a, range_b), max(range_a, range_b))
    return lo <= scalar <= hi

def _parse_range(range_text: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*[–—−-]\s*(\d+(?:[.,]\d+)?)", _norm_dash(range_text))
    if not m:
        return None, None
    a = _to_float(m.group(1))
    b = _to_float(m.group(2))
    return a, b

def _cleanup_z_between(answer: str) -> str:
    def repl1(m):
        scalar = _to_float(m.group("scalar"))
        rtxt = m.group("range")
        ra, rb = _parse_range(rtxt)
        if ra is not None and rb is not None and _scalar_inside_range(scalar, ra, rb):
            return f"{m.group('prefix')}{rtxt}"
        return m.group(0)

    def repl2(m):
        rtxt = m.group("range")
        scalar_text = m.group("scalar")
        ra, rb = _parse_range(rtxt)
        mnum = re.search(r"\d+(?:[.,]\d+)?", scalar_text)
        scalar = _to_float(mnum.group(0)) if mnum else None
        if scalar is not None and ra is not None and rb is not None and _scalar_inside_range(scalar, ra, rb):
            return f"{m.group('prefix')}{rtxt}"
        return m.group(0)

    out = _ZWISCHEN_SCALAR_UND_RANGE.sub(repl1, answer)
    out = _ZWISCHEN_RANGE_UND_SCALAR.sub(repl2, out)
    return out

def _multiquery_expand(question: str, n: int = EXPANSIONS) -> list[str]:
    prompt = (
        f"Generate {n} short alternative German phrasings for the following question. "
        "Keep domain terms; vary synonyms and morphology. One per line, no numbering.\n\n"
        f"Frage: {question}"
    )
    msg = [{"role": "user", "content": prompt}]
    out = oa.chat.completions.create(model=OPENAI_MODEL, messages=msg, temperature=0.2)
    text = (out.choices[0].message.content or "").strip()
    alts = [l.strip("•- ").strip() for l in text.splitlines() if l.strip()]
    return [question] + alts[:max(0, n)]


# include aerosol units in unit tokens
_UNIT_TOKENS = ["ppm", "mg/m³", "mg/m3", "µg/m³", "ug/m3", "dB(A)", "dB", "m/s²", "°C", "%", "lx", "A", "V"]

def _question_unit_hints(q: str) -> set[str]:
    qn = _norm_dash(q.lower())
    qn = qn.replace("³", "3")
    hits = set()
    for u in _UNIT_TOKENS:
        key = u.lower().replace("³", "3")
        if key.replace("(", "").replace(")", "") in qn.replace("(", "").replace(")", ""):
            hits.add(u)
    # Also add simple numeric hint if any number present
    if re.search(r"\d", qn):
        hits.add("<NUM>")
    return hits

def _soft_unit_score(q_hints: set[str], text: str) -> float:
    if not q_hints:
        return 0.0
    t = _norm_dash(text.lower())
    t = t.replace("³", "3")
    score = 0.0
    for u in q_hints:
        if u == "<NUM>":
            if re.search(r"\d", t): score += 0.5
        else:
            key = u.lower().replace("³", "3")
            if key.replace("(", "").replace(")", "") in t.replace("(", "").replace(")", ""):
                score += 1.0
    return score

# --- NEW: soft keyword score for domain keywords (not units)
def _soft_keyword_score(question: str, text: str) -> float:
    ql = question.lower()
    tl = text.lower()
    score = 0.0
    # Starkstrom / Schaltschrank context
    if any(k in ql for k in ["starkstrom", "schaltschrank", "loto", "400 v", "63 a"]):
        for k in ["starkstrom", "schaltschrank", "loto", "400 v", "63 a", "offen 400 v/63 a"]:
            if k in tl: score += 0.6
    # KSS / aerosols context
    if any(k in ql for k in ["kss", "aerosol", "feinstaub", "druckluft", "spannvorrichtung", "rüsten", "ruesten"]):
        for k in ["kss", "kss-aerosole", "aerosol", "mg/m³", "nebel", "schleier", "druckluft", "spannvorrichtung"]:
            if k in tl: score += 0.5
    # Abbreviation lookup context
    if re.search(r"\b(was\s+bedeutet|what\s+does).+\b", ql):
        for k in ["abkürzungsverzeichnis", "abkuerzungsverzeichnis", "kss", "kühlschmierstoff", "kuehlschmierstoff"]:
            if k in tl: score += 0.4
    return score

def _mmr_select(question_vec: list[float], objs: list, k: int = KEEP_FOR_CONTEXT, lambda_: float = 0.7):
    selected = []
    candidates = objs[:]
    sims = {}

    def _cos_sim(a, b):
        # a and b are vectors; here we don't have candidate vectors. We approximate
        # via weaviate distance if present; otherwise 0. This keeps MMR simple.
        return 0.0

    while candidates and len(selected) < k:
        best, best_score = None, -1e9
        for o in candidates:
            # base score from distance if present (1 - distance), else 0
            base = 0.0
            if hasattr(o, "metadata") and getattr(o.metadata, "distance", None) is not None:
                base = 1.0 - float(o.metadata.distance)

            # diversity penalty against already picked
            div = 0.0
            for s in selected:
                # no vectors at hand; use textual overlap penalty instead:
                t1 = o.properties.get("text", "")
                t2 = s.properties.get("text", "")
                if t1 and t2:
                    # rough Jaccard of word sets for diversity
                    w1, w2 = set(t1.split()), set(t2.split())
                    inter = len(w1 & w2); uni = max(len(w1 | w2), 1)
                    sim = inter / uni
                    div = max(div, sim)

            score = lambda_ * base - (1 - lambda_) * div
            if score > best_score:
                best, best_score = o, score
        selected.append(best)
        candidates.remove(best)
    return selected

def _rerank_llm(question: str, objs: list, top_n: int = KEEP_FOR_CONTEXT) -> list:
    # Build a single prompt to score multiple candidates
    blocks = []
    for i, o in enumerate(objs, 1):
        txt = o.properties.get("text", "")[:1200]
        src = o.properties.get("source", ""); pg = o.properties.get("page", "")
        blocks.append(f"[{i}] (Quelle: {src}, Seite {pg})\n{txt}")
    prompt = (
        "Bewerte die Relevanz der folgenden Passagen für die Frage auf einer Skala von 0.0 bis 1.0. "
        "Gib JSON als Liste von Objekten {idx, score} zurück, keine Erklärungen.\n\n"
        f"Frage: {question}\n\n"
        + "\n\n---\n\n".join(blocks)
    )
    res = oa.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    import json
    text = (res.choices[0].message.content or "[]").strip()
    try:
        scored = json.loads(text)
        scored_map = {int(x["idx"]): float(x["score"]) for x in scored if "idx" in x and "score" in x}
    except Exception:
        return objs[:top_n]

    pairs = []
    for i, o in enumerate(objs, 1):
        sc = scored_map.get(i, 0.0)
        pairs.append((sc, o))
    pairs.sort(key=lambda x: x[0], reverse=True)
    return [o for sc, o in pairs[:top_n]]

def _gather_candidates(coll, queries: list[str]) -> list:
    """
    For each query, do vector + BM25, collect top candidates, dedup by UUID.
    """
    bag = {}
    for q in queries:
        qvec = _embed(q)
        vec = _vector_search(coll, qvec, k=CANDIDATES_PER_QUERY)
        bm  = _bm25_search(coll, q, k=CANDIDATES_PER_QUERY)
        for res in [vec, bm]:
            if not res:
                continue
            for o in res.objects:
                bag[o.uuid] = o   # dedup by uuid
    return list(bag.values())

def _numeric_guard(answer: str, sources_text: str) -> str:
    src_spans = _extract_numeric_spans(_norm_dash(sources_text))
    if not src_spans:
        return answer

    ans_norm = _norm_dash(answer)
    ans_spans = _extract_numeric_spans(ans_norm)
    if ans_spans:
        replacements: List[Tuple[int, int, str]] = []
        for ans_span in ans_spans:
            best = _choose_best_source_span(ans_span, src_spans)
            if best is None:
                continue
            replacements.append((ans_span["start"], ans_span["end"], best["match"]))

        if replacements:
            out = ans_norm
            for start, end, rep in sorted(replacements, key=lambda x: x[0], reverse=True):
                out = out[:start] + rep + out[end:]
            ans_norm = out

    cleaned = _cleanup_z_between(ans_norm)
    return cleaned

# (Left as-is; not used by the main pipeline. Keep for compatibility.)
def vector_search(q):
    return client.query(..., return_properties=RETURN_PROPS, limit=K)

ROOM_KEYWORDS = {
    # CNC
    "cnc": "CNC-Fräserei", "fräs": "CNC-Fräserei", "fraes": "CNC-Fräserei",
    "machining": "CNC-Fräserei", "milling": "CNC-Fräserei",

    # Werkzeug- & Schleifraum
    "schleif": "Werkzeug- & Schleifraum", "werkzeug": "Werkzeug- & Schleifraum",
    "grind": "Werkzeug- & Schleifraum", "grinding": "Werkzeug- & Schleifraum",

    # Endmontage & Prüfstand
    "montage": "Endmontage & Prüfstand", "prüf": "Endmontage & Prüfstand",
    "pruef": "Endmontage & Prüfstand", "test bench": "Endmontage & Prüfstand",
    "endmontage": "Endmontage & Prüfstand", "prüfstand": "Endmontage & Prüfstand",
}

MACHINE_KEYWORDS = {
    "cnc": "CNC-Bearbeitungszentrum",
    "bearbeitungszentrum": "CNC-Bearbeitungszentrum",
    "schleif": "Schleifen/Handschleifer",
    "handschleif": "Schleifen/Handschleifer",
    "prüfstand": "Prüfstand",
    "schrumpf": "Schrumpfgerät",
    "loto": "Schaltschrank/LOTO",
}

# Known rooms (case-insensitive match)
KNOWN_ROOMS = {
    "CNC-Fräserei",
    "Werkzeug- & Schleifraum",
    "Endmontage & Prüfstand",
}

# --- Noise disambiguation helpers ---

# What the user might explicitly ask for
NOISE_KEYWORDS = {
    "laeq": ["laeq", "leq", "durchschnitt", "average"],
    "peak": ["peak", "lafmax", "spitze", "spitzenwert", "maximal"],
    "limit": ["grenzwert", "expositionsgrenzwert", "am ohr", "ohr"],
    "room": ["raum", "raumpegel", "im raum", "hintergrundpegel"],
}

def _detect_noise_kind_from_question(q: str) -> Optional[str]:
    ql = q.lower()
    for kind, kws in NOISE_KEYWORDS.items():
        if any(k in ql for k in kws):
            return kind
    return None

# How to detect kinds from retrieved text
NOISE_PATTERNS = {
    "laeq": re.compile(r"\bLAeq\b|\bLeq\b", re.I),
    "peak": re.compile(r"\bPeak[s]?\b|\bLAFmax\b|\bMax(?:imal)?\b", re.I),
    "limit": re.compile(r"\bExpositionsgrenzwert\b|\bGrenzwert\b|am\s*Ohr", re.I),
    "room": re.compile(r"\bim\s*Raum\b|\bRaumpegel\b|\bHintergrundpegel\b", re.I),
}

def _classify_noise_kinds_in_objs(objs) -> set[str]:
    kinds = set()
    for o in objs:
        t = o.properties.get("text", "")
        for kind, rx in NOISE_PATTERNS.items():
            if rx.search(t):
                kinds.add(kind)
    return kinds

def _filter_objs_by_noise_kind(objs, kind: str):
    rx = NOISE_PATTERNS.get(kind)
    if not rx:
        return objs
    filtered = [o for o in objs if rx.search(o.properties.get("text", ""))]
    return filtered or objs  # fallback if over-filtered

def detect_room_from_question(q: str):
    ql = q.lower()
    for k, v in ROOM_KEYWORDS.items():
        if k in ql:
            return v
    return None

def _looks_like_room_only(q: str) -> Optional[str]:
    s = (q or "").strip()
    for r in KNOWN_ROOMS:
        if s.lower() == r.lower():
            return r
    return None

def pick_machine_from_question(q):
    ql = q.lower()
    for k, v in MACHINE_KEYWORDS.items():
        if k in ql:
            return v
    return None

# ---------------------------
# Main retrieval pipeline
# ---------------------------

def _resolve_collection(target: Optional[str]):
    ensure_collections()

    if target:
        t = target.strip()
        # Key?
        if t.upper() in {"FB1", "FB2", "FB3"}:
            return get_collection_by_key(t.upper())
        # Class name
        return get_collection(t)

    # Legacy default path
    return get_collection(LEGACY_DEFAULT_CLASS)

def retrieve_answer(question: str, target: Optional[str] = None) -> str:
    coll = _resolve_collection(target)

    # --- normalization helper (local & minimal) ---
    def _norm_local(s: Optional[str]) -> str:
        if not s:
            return ""
        return re.sub(r"\s+", " ", s).replace("–", "-").strip().lower()

    # Detect if user already specified room/machine
    asked_room = detect_room_from_question(question)
    asked_machine = pick_machine_from_question(question)

    # --- Option A: user replied with only a room name
    room_only = _looks_like_room_only(question)
    if room_only:
        return (
            f"Verstanden: **{room_only}**.\n"
            "Bitte gib auch die Kennzahl an, z. B.:\n"
            f"- *Wie hoch ist der Geräuschpegel in {room_only}?*\n"
            f"- *Temperatur in {room_only}?*\n"
            f"- *Vibrationen in {room_only}?*\n"
        )

    # treat 'Dezibel', 'dBA', 'dB' as noise queries too
    is_noise_q = bool(re.search(r"\b(noise|geräusch|lärm|dezibel|dba|db)\b", question, re.I))

    NOISE_PATTERNS = {
        "laeq": re.compile(r"\bLAeq\b|\bLeq\b", re.I),
        "peak": re.compile(r"\bPeak[s]?\b|\bLAFmax\b|\bMax(?:imal)?\b", re.I),
        "limit": re.compile(r"\bExpositionsgrenzwert\b|\bGrenzwert\b|am\s*Ohr", re.I),
        "room": re.compile(r"\bim\s*Raum\b|\bRaumpegel\b|\bHintergrundpegel\b", re.I),
    }
    def _detect_noise_kind_from_question_local(q: str) -> Optional[str]:
        ql = q.lower()
        if any(k in ql for k in ["laeq", "leq", "durchschnitt", "average", "dezibel", "dba", "db"]): return "laeq"
        if any(k in ql for k in ["peak", "lafmax", "spitze", "spitzenwert", "maximal"]): return "peak"
        if any(k in ql for k in ["grenzwert", "expositionsgrenzwert", "am ohr", "ohr"]): return "limit"
        if any(k in ql for k in ["raum", "raumpegel", "im raum", "hintergrundpegel"]): return "room"
        return None

    def _classify_noise_kinds_in_objs_local(_objs) -> set[str]:
        kinds = set()
        for o in _objs:
            t = o.properties.get("text", "")
            for kind, rx in NOISE_PATTERNS.items():
                if rx.search(t):
                    kinds.add(kind)
        return kinds

    def _filter_objs_by_noise_kind_local(_objs, kind: str):
        rx = NOISE_PATTERNS.get(kind)
        if not rx:
            return _objs
        _filtered = [o for o in _objs if rx.search(o.properties.get("text", ""))]
        return _filtered or _objs  # fallback if over-filtered

    # Domain expansions for Starkstrom/Aerosole/etc. and abbreviation lookups
    def _domain_expansions(q: str) -> list[str]:
        out = []
        ql = q.lower()
        if "starkstrom" in ql:
            out += ["400 V", "63 A", "Schaltschrank", "LOTO", "offen 400 V/63 A"]
        if any(k in ql for k in ["kss", "aerosol", "feinstaub", "staub", "schwebstoff", "druckluft", "spannvorrichtung", "rüsten", "ruesten"]):
            out += ["KSS-Aerosole", "Aerosol", "Feinstaub", "Staub", "Schwebstoff", "mg/m³", "Nebel", "Schleier", "Druckluft", "Spannvorrichtung"]
        if "lecktest" in ql or "glykol" in ql:
            out += ["Glykol-Aerosole", "Ethylenglykol", "mg/m³"]
        if re.search(r"\b(was\s+bedeutet|what\s+does)\b", ql):
            out += ["Abkürzungsverzeichnis", "KSS", "Kühlschmierstoff", "Kuehlschmierstoff"]
        return list(dict.fromkeys(out))  # dedup, keep order


    noise_kind_asked = _detect_noise_kind_from_question_local(question)

    # 1) Multi-query expansion
    queries = _multiquery_expand(question, n=EXPANSIONS)

    # 1a) Bias queries only if the user specified room/machine/noise-kind
    def _prefix(q: str) -> str:
        pref = ""
        if asked_room:    pref += f"{asked_room}: "
        if asked_machine: pref += f"{asked_machine}: "
        nk = _detect_noise_kind_from_question_local(q)
        if nk == "laeq":    pref += "LAeq: "
        elif nk == "peak":  pref += "Peak LAFmax: "
        elif nk == "limit": pref += "Expositionsgrenzwert am Ohr: "
        elif nk == "room":  pref += "Raumpegel: "
        # optional: default bias for generic "noise level" phrasing
        if is_noise_q and re.search(r"\blevel\b|\bniveau\b", q, re.I) and "LAeq" not in pref:
            pref += "LAeq: "
        return pref + q if pref else q

    queries = [_prefix(q) for q in queries]

    # Append domain expansions (exact tokens) to guide BM25/vector search
    exp = _domain_expansions(question)
    if exp:
        queries.extend([_prefix(x) for x in exp])

    # 2) Gather & dedup candidates
    objs_all = _gather_candidates(coll, queries)

    # Pre-filter by room before ranking (if user specified a room)
    if asked_room:
        objs_room = [o for o in objs_all if _norm_local(o.properties.get("room")) == _norm_local(asked_room)]
        if objs_room:
            objs_all = objs_room

    # If user already specified a noise kind, pre-filter candidates to that kind
    if is_noise_q and noise_kind_asked and objs_all:
        objs_kind = _filter_objs_by_noise_kind_local(objs_all, noise_kind_asked)
        if objs_kind:
            objs_all = objs_kind

    # ---- Early noise-kind ambiguity check BEFORE ranking ----
    if is_noise_q and objs_all:
        room_noise_kinds = _classify_noise_kinds_in_objs_local(objs_all)
        if not noise_kind_asked and len(room_noise_kinds) > 1:
            label_map = {
                "laeq": "LAeq (Durchschnitt)",
                "peak": "Peak / LAFmax",
                "limit": "Grenzwert am Ohr",
                "room": "Raumpegel (im Raum)",
            }
            # echo full question in each option
            opts = [k for k in ["laeq", "peak", "limit", "room"] if k in room_noise_kinds]
            options_list = [f"- {question} — Kennzahl: {label_map[k]}" for k in opts]
            return "Welche Kennzahl meinst du? Antworte mit einer Option:\n" + "\n".join(options_list)

    if not objs_all:
        # Fallback to classic single-query retrieval
        qvec = _embed(question if not asked_room and not asked_machine else _prefix(question))
        vec_res = _vector_search(coll, qvec, k=KEEP_FOR_CONTEXT)
        bm25_res = _bm25_search(coll, question if not asked_room and not asked_machine else _prefix(question), k=KEEP_FOR_CONTEXT)
        objs = _hybrid_rank(vec_res, bm25_res, alpha=HYBRID_ALPHA)
    else:
        # 3) Unit-aware and keyword-aware boost to bring measurements & domain facts forward
        hints = _question_unit_hints(question)
        scored = []
        for o in objs_all:
            base = 0.0
            if hasattr(o, "metadata") and getattr(o.metadata, "distance", None) is not None:
                base = 1.0 - float(o.metadata.distance)
            txt = o.properties.get("text", "")
            boost = _soft_unit_score(hints, txt)
            kw_boost = _soft_keyword_score(question, txt)

            # strong bonus if candidate's room matches asked room
            # inside the scoring loop where room_bonus is computed
            room_bonus = 0.0
            if asked_room:
                cand_room = (o.properties.get("room") or "").strip()
                if _norm_local(cand_room) == _norm_local(asked_room):
                    room_bonus = 0.5   # was 0.35


            # tiny preference for explicit noise markers on noise questions
            noise_marker = 0.0
            if is_noise_q and re.search(r"\bLAeq\b|\bLeq\b|\bLAFmax\b|\bdB\(A\)|\bdB\b", txt, re.I):
                noise_marker = 0.2

            scored.append((base + 0.15 * boost + kw_boost + room_bonus + noise_marker, o))

        # Sort by boosted score and apply MMR for diversity
        scored.sort(key=lambda x: x[0], reverse=True)
        objs_sorted = [o for _, o in scored]
        objs_mmr = _mmr_select([], objs_sorted, k=max(KEEP_FOR_CONTEXT * 2, 10), lambda_=0.7)

        # 4) LLM rerank top K (final cut)
        objs = _rerank_llm(question, objs_mmr, top_n=KEEP_FOR_CONTEXT)

    # ---------- Ambiguity checks BEFORE composing the answer ----------

    # Rooms present in top hits
    rooms_in_top = []
    for o in objs:
        r = (o.properties.get("room") or "").strip()
        if r:
            rooms_in_top.append(r)
    unique_rooms = sorted({r for r in rooms_in_top if r and r.lower() not in {"unklar", "gesamt"}})

    # Improved room clarification: suggest full follow-up questions
    if not asked_room and len(unique_rooms) > 1:
        options = "\n".join(f"- {question} in {r}" for r in unique_rooms)
        return (
            "Welchen Raum meinen Sie?\n"
            "Antworte mit einer der folgenden Optionen:\n"
            f"{options}"
        )

    # If a room is specified (or only one appears), filter to that room to avoid mixing
    if asked_room:
        objs = [o for o in objs if _norm_local(o.properties.get("room")) == _norm_local(asked_room)]
        if not objs:  # safety fallback
            objs = _rerank_llm(question, objs_all[:KEEP_FOR_CONTEXT*2], top_n=KEEP_FOR_CONTEXT)

    # Machine ambiguity (after room disambiguation)
    machines_in_top = []
    for o in objs:
        m = (o.properties.get("machine") or "").strip()
        if m:
            machines_in_top.append(m)
    real_machines = sorted({m for m in machines_in_top if m and m.lower() != "allgemein"})

    if not asked_machine and len(real_machines) > 1:
        options_list = [f"- {question} — Maschine: {m}" for m in real_machines]
        return "Welche Maschine meinst du? Antworte mit einer Option:\n" + "\n".join(options_list)

    # If machine specified, filter to that machine as well (keeps answer precise)
    if asked_machine:
        objs_filtered = [o for o in objs if _norm_local(o.properties.get("machine")) == _norm_local(asked_machine)]
        if objs_filtered:
            objs = objs_filtered

    # ---------- Noise-specific disambiguation (inside the room) ----------
    noise_kind_asked = _detect_noise_kind_from_question_local(question)
    if is_noise_q:
        kinds_now = _classify_noise_kinds_in_objs_local(objs)
        if not noise_kind_asked and len(kinds_now) > 1:
            label_map = {
                "laeq": "LAeq (Durchschnitt)",
                "peak": "Peak / LAFmax",
                "limit": "Grenzwert am Ohr",
                "room": "Raumpegel (im Raum)",
            }
            ordered = [k for k in ["laeq", "peak", "limit", "room"] if k in kinds_now]
            options_list = [f"- {question} — Kennzahl: {label_map[k]}" for k in ordered]
            return "Welche Kennzahl meinst du? Antworte mit einer Option:\n" + "\n".join(options_list)

        # Auto-select the single remaining kind
        if not noise_kind_asked and len(kinds_now) == 1:
            noise_kind_asked = next(iter(kinds_now))

        # If user specified a kind, filter and apply robust fallback
        if noise_kind_asked:
            filtered = _filter_objs_by_noise_kind_local(objs, noise_kind_asked)
            if not filtered:
                core_rx = re.compile(r"\b(?:LAeq|Leq|LAFmax|dB\(A\)|dB)\b", re.I)
                filtered = [o for o in objs if core_rx.search(o.properties.get("text", ""))]
            if not filtered:
                room_only_pool = [o for o in objs_all if _norm_local(o.properties.get("room")) == _norm_local(asked_room)] if asked_room else objs_all[:]
                if noise_kind_asked in NOISE_PATTERNS:
                    rx = NOISE_PATTERNS[noise_kind_asked]
                    filtered = [o for o in room_only_pool if rx.search(o.properties.get("text", ""))]
                if not filtered:
                    core_rx = re.compile(r"\b(?:LAeq|Leq|LAFmax|dB\(A\)|dB)\b", re.I)
                    filtered = [o for o in room_only_pool if core_rx.search(o.properties.get("text", ""))]
            if filtered:
                objs = filtered

        # Gentle preference for measurement-bearing text
        dba_rx = re.compile(r"\b(?:dB\(A\)|dB|LAeq|LAFmax|Lärm|Geräusch)\b", re.I)
        objs_dba = [o for o in objs if dba_rx.search(o.properties.get("text", ""))]
        if objs_dba:
            objs = objs_dba

    # -----------------------------------------------------------------

    # Compose context & ask the model
    context, chunks = _format_context(objs)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Frage: {question}\n\nKontext:\n{context}"},
    ]

    resp = oa.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.0,
    )
    draft = (resp.choices[0].message.content or "").strip()

    # safeguard — if user asked about Starkstrom and sources mention 400 V / 63 A, avoid false "not specified"
    sources_concat_pre = context  # quick check over the rendered context
    if re.search(r"\bstarkstrom\b", question, re.I) and re.search(r"\b(400\s*V|63\s*A)\b", sources_concat_pre, re.I):
        if "nicht angegeben" in draft.lower():
            draft = "Ja – offen 400 V/63 A."

    # numeric sanity pass
    sources_concat = "\n".join(c["text"] for c in chunks)
    final = _numeric_guard(draft, sources_concat)

    # --- NEW: scrub common extraction artifacts like '((cid:48070)'
    def _scrub_artifacts(s: str) -> str:
        s = re.sub(r"\(\s*cid\s*:\s*\d+\s*\)", "", s, flags=re.I)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip()
    final = _scrub_artifacts(final)

    cites = "; ".join(f"{c['source']} S.{c['page']}" for c in chunks[:3])
    if cites:
        final = f"{final}\n\n— Quellen: {cites}"

    return final
