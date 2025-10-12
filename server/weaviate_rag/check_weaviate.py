import os, sys, json, unicodedata
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.weaviate_setup import client, close_client

# OPTIONAL: import your embedder if you have it
try:
    # adjust this import to whatever you actually expose
    from app.services.embedder import embed_text  # should return a list[float]
    HAVE_EMBEDDER = True
except Exception:
    HAVE_EMBEDDER = False
    embed_text = None

def normalize(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    return (
        s.replace("\u2013", "-")
         .replace("\u2212", "-")
         .replace("\u00A0", " ")
         .replace("m³", "m3")
    )

def list_with_counts():
    cols = client.collections.list_all()
    try:
        names = [c.name for c in cols]
    except Exception:
        names = list(cols)
    out = []
    for name in names:
        try:
            total = client.collections.get(name).aggregate.over_all().total_count
        except Exception:
            total = "n/a"
        out.append((name, total))
    return out

def print_schema(name="Chatbot_FB1"):
    try:
        coll = client.collections.get(name)
        cfg = coll.config.get()
        print("\nSchema properties for", name, ":")
        for p in cfg.properties:
            print(" -", p.name, p.data_type)
        print("Vectorizer:", getattr(cfg, "vectorizer_config", None))
    except Exception as e:
        print("Could not fetch schema:", e)

def show_results(tag, results):
    print(f"\n=== {tag} ===")
    if not results or not results.objects:
        print("No results.")
        return
    for i, obj in enumerate(results.objects, 1):
        meta = getattr(obj, "metadata", None)
        score = getattr(meta, "score", None) if meta else None
        distance = getattr(meta, "distance", None) if meta else None
        d = obj.properties or {}
        text = d.get("text") or d.get("content") or d.get("chunk") or ""
        text = " ".join(text.split())
        print(f"[{i}] score={score} distance={distance}")
        print("source:", d.get("source"))
        print("text:", (text[:300] + "…") if len(text) > 300 else text)
        print("-" * 60)

def q_bm25(coll, query):
    return coll.query.bm25(
        query=query,
        limit=8,
        return_properties=["source", "text"],
        return_metadata=["score"],
    )

def q_near_vector(coll, query):
    if not HAVE_EMBEDDER:
        return None
    try:
        vec = embed_text(query)
        return coll.query.near_vector(
            near_vector=vec,
            limit=8,
            return_properties=["source", "text"],
            return_metadata=["distance", "score"],
        )
    except Exception as e:
        print("nearVector failed:", e)
        return None

print("WEAVIATE_HTTP =", os.getenv("WEAVIATE_HTTP"))
print("WEAVIATE_PORT =", os.getenv("WEAVIATE_PORT"))
print("WEAVIATE_GRPC =", os.getenv("WEAVIATE_GRPC"))
print("WEAVIATE_GRPC_PORT =", os.getenv("WEAVIATE_GRPC_PORT"))

print_schema("Chatbot_FB1")

print("\nCollections (name, total_count):")
for name, cnt in list_with_counts():
    print(" -", name, cnt)

fb1 = client.collections.get("Chatbot_FB1")
queries = [
    "Feinstaub CNC Fräserei",
    "CNC-Fräserei KSS-Aerosole 2–8 mg/m³",
    "Kühlschmierstoff-Aerosole 2-8 m3",
    "Feinstaub Belastung KSS Aerosole",
]
for q in queries:
    qn = normalize(q)
    show_results(f'BM25: "{qn}"', q_bm25(fb1, qn))
    if HAVE_EMBEDDER:
        show_results(f'nearVector: "{qn}"', q_near_vector(fb1, qn))

# Always close to avoid resource warning
try:
    close_client()
except Exception:
    pass
