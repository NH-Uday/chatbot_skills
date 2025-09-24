# app/services/rag.py
import os
import re
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from weaviate.classes.query import MetadataQuery

from app.services.weaviate_setup import client, CLASS_NAME

load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
TOP_K = int(os.getenv("TOP_K", "6"))
HYBRID_ALPHA = float(os.getenv("HYBRID_ALPHA", "0.5"))

oa = OpenAI()

SYSTEM_PROMPT = (
    "Du antwortest ausschließlich anhand des bereitgestellten Kontexts. "
    "Kritische Zahlen und Bereiche (z. B. ppm, dB(A), m/s², %, lx, A/V) "
    "**musst du wortgleich** wie im Kontext wiedergeben. "
    "Falls die Information nicht im Kontext enthalten ist, antworte: "
    "'Das ist im Dokument nicht angegeben.' "
    "Antworte kurz und auf Deutsch."
)

def _embed(q: str) -> List[float]:
    return oa.embeddings.create(model=OPENAI_EMBED_MODEL, input=q).data[0].embedding

def _vector_search(coll, query_vec, k=TOP_K):
    return coll.query.near_vector(
        near_vector=query_vec, limit=k, return_metadata=MetadataQuery(distance=True)
    )

def _bm25_search(coll, query: str, k=TOP_K):
    try:
        return coll.query.bm25(query=query, limit=k, return_metadata=MetadataQuery(rank=True))
    except Exception:
        return None

def _hybrid_rank(vec_res, bm25_res, alpha=HYBRID_ALPHA) -> List[Dict]:
    items = {}
    def add(res, score_key):
        if not res: return
        vals = []
        for o in res.objects:
            raw = (o.metadata.distance if score_key == "distance" else o.metadata.score) or 0.0
            vals.append(raw)
        if not vals: return
        lo, hi = (min(vals), max(vals))
        rng = max(hi - lo, 1e-9)
        for o in res.objects:
            raw = (o.metadata.distance if score_key == "distance" else o.metadata.score) or 0.0
            sim = 1.0 - (raw - lo)/rng if score_key == "distance" else (raw - lo)/rng
            prev = items.get(o.uuid, {"obj": o, "sim": 0.0})
            prev["sim"] += sim
            items[o.uuid] = prev
    add(vec_res, "distance")
    add(bm25_res, "score")
    ranked = []
    for v in items.values():
        o = v["obj"]
        has_vec = vec_res and any(o.uuid == r.uuid for r in vec_res.objects)
        has_bm  = bm25_res and any(o.uuid == r.uuid for r in bm25_res.objects)
        if has_vec and has_bm:
            sim_vec = next((1 - r.metadata.distance for r in vec_res.objects if r.uuid == o.uuid), 0.0)
            sim_bm  = next((r.metadata.score       for r in bm25_res.objects if r.uuid == o.uuid), 0.0)
            hybrid  = alpha * sim_vec + (1 - alpha) * sim_bm
        else:
            hybrid = v["sim"]
        ranked.append((hybrid, o))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [o for _, o in ranked][:TOP_K]

def _format_context(objs) -> Tuple[str, List[Dict]]:
    chunks = []
    for o in objs:
        props = o.properties
        chunks.append({"text": props["text"], "source": props["source"], "page": props["page"]})
    context = "\n\n---\n\n".join(
        f"[{i+1}] (Quelle: {c['source']}, Seite {c['page']})\n{c['text']}" for i, c in enumerate(chunks)
    )
    return context, chunks


_DASH_PATTERN = re.compile(r"[–—−\-]+")  # en/em dash, minus, hyphen
_WS = re.compile(r"\s+")

_NUM_SPAN_RE = re.compile(
    r"""
    (?P<num1>\d+(?:[.,]\d+)?)                       # first number
    (?:\s*(?:[–—−-])\s*(?P<num2>\d+(?:[.,]\d+)?))?  # optional range end
    \s*
    (?P<unit>
        ppm|
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

def _norm_unit(u: Optional[str]) -> str:
    if not u: return ""
    u = u.strip()
    u = u.replace("dB(a)", "dB(A)").replace("dB(A)", "dB(A)")
    if u.lower() == "db": u = "dB"
    u = u.replace("m/s^2", "m/s²").replace("m/s2", "m/s²")
    return u

def _extract_numeric_spans(text: str) -> List[Dict]:
    spans = []
    for m in _NUM_SPAN_RE.finditer(text):
        s = m.group(0)
        u = _norm_unit(m.group("unit"))
        a = _to_float(m.group("num1"))
        b = m.group("num2")
        b_val = _to_float(b) if b else None
        spans.append({
            "match": s, "unit": u, "a": a, "b": b_val, "start": m.start(), "end": m.end()
        })
    return spans

def _span_distance(ans: Dict, src: Dict) -> float:
    if ans["unit"] != src["unit"]: return 1e6
    if _norm_ws(_norm_dash(ans["match"])) == _norm_ws(_norm_dash(src["match"])): return 0.0
    a1, b1 = ans["a"], ans["b"]
    a2, b2 = src["a"], src["b"]
    if b1 is not None and b2 is not None:
        return abs(a1 - a2) + abs(b1 - b2)
    if b1 is None and b2 is not None:
        if a2 <= a1 <= b2: return 0.0
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

# NEW: collapse "zwischen <scalar> und <range>" → "zwischen <range>" (if scalar is inside the range)
_ZWISCHEN_SCALAR_UND_RANGE = re.compile(
    r"""(?P<prefix>zwischen\s*)
        (?P<scalar>\d+(?:[.,]\d+)?)
        (?P<mid>\s*und\s*)
        (?P<range>\d+(?:[.,]\d+)?\s*(?:[–—−-]\s*\d+(?:[.,]\d+)?)\s*
           (?:ppm|dB\(A\)|dB|m/s(?:²|\^2|2)?|%|lx|A|V|°C)?)
    """,
    re.IGNORECASE | re.VERBOSE
)

_ZWISCHEN_RANGE_UND_SCALAR = re.compile(
    r"""(?P<prefix>zwischen\s*)
        (?P<range>\d+(?:[.,]\d+)?\s*(?:[–—−-]\s*\d+(?:[.,]\d+)?)\s*
           (?:ppm|dB\(A\)|dB|m/s(?:²|\^2|2)?|%|lx|A|V|°C)?)
        (?P<mid>\s*und\s*)
        (?P<scalar>\d+(?:[.,]\d+)?(?:\s*(?:ppm|dB\(A\)|dB|m/s(?:²|\^2|2)?|%|lx|A|V|°C))?)
    """,
    re.IGNORECASE | re.VERBOSE
)

def _scalar_inside_range(scalar: float, range_a: float, range_b: float) -> bool:
    lo, hi = (min(range_a, range_b), max(range_a, range_b))
    return lo <= scalar <= hi

def _parse_range(range_text: str) -> Tuple[Optional[float], Optional[float]]:
    # find first numeric range inside text
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*[–—−-]\s*(\d+(?:[.,]\d+)?)", _norm_dash(range_text))
    if not m: return None, None
    a = _to_float(m.group(1)); b = _to_float(m.group(2))
    return a, b

def _cleanup_z_between(answer: str) -> str:
    # case 1: zwischen <scalar> und <range>
    def repl1(m):
        scalar = _to_float(m.group("scalar"))
        rtxt   = m.group("range")
        ra, rb = _parse_range(rtxt)
        if ra is not None and rb is not None and _scalar_inside_range(scalar, ra, rb):
            return f"{m.group('prefix')}{rtxt}"
        return m.group(0)

    # case 2: zwischen <range> und <scalar>
    def repl2(m):
        rtxt   = m.group("range")
        scalar_text = m.group("scalar")
        ra, rb = _parse_range(rtxt)
        # try to pull scalar number out of its text
        mnum = re.search(r"\d+(?:[.,]\d+)?", scalar_text)
        scalar = _to_float(mnum.group(0)) if mnum else None
        if scalar is not None and ra is not None and rb is not None and _scalar_inside_range(scalar, ra, rb):
            return f"{m.group('prefix')}{rtxt}"
        return m.group(0)

    out = _ZWISCHEN_SCALAR_UND_RANGE.sub(repl1, answer)
    out = _ZWISCHEN_RANGE_UND_SCALAR.sub(repl2, out)
    return out

def _numeric_guard(answer: str, sources_text: str) -> str:

    # 1) replacement step
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

    # 2) cleanup "zwischen … und …" redundancy
    cleaned = _cleanup_z_between(ans_norm)
    return cleaned

# ---------------------------
# Main retrieval pipeline
# ---------------------------

def retrieve_answer(question: str) -> str:
    coll = client.collections.get(CLASS_NAME)

    qvec = _embed(question)
    vec_res = _vector_search(coll, qvec, k=TOP_K)
    bm25_res = _bm25_search(coll, question, k=TOP_K)

    objs = _hybrid_rank(vec_res, bm25_res, alpha=HYBRID_ALPHA)
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
    draft = resp.choices[0].message.content.strip()

    # numeric sanity pass
    sources_concat = "\n".join(c["text"] for c in chunks)
    final = _numeric_guard(draft, sources_concat)

    cites = "; ".join(f"{c['source']} S.{c['page']}" for c in chunks[:3])
    if cites:
        final = f"{final}\n\n— Quellen: {cites}"

    return final
