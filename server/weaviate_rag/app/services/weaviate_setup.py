import os
from urllib.parse import urlparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances

# ----- Load .env from server/.env (kept from your original) -----
THIS_FILE = Path(__file__).resolve()
ENV_PATH = THIS_FILE.parents[3] / ".env"   # server/.env
load_dotenv(dotenv_path=str(ENV_PATH))

# ======= Multi-bot class names =======
# You can override these in .env:
#   WEAVIATE_CLASS_FB1=Chatbot_FB1
#   WEAVIATE_CLASS_FB2=Chatbot_FB2
#   WEAVIATE_CLASS_FB3=Chatbot_FB3
CLASS_FB1 = (os.getenv("WEAVIATE_CLASS_FB1") or "Chatbot_FB1").strip()
CLASS_FB2 = (os.getenv("WEAVIATE_CLASS_FB2") or "Chatbot_FB2").strip()
CLASS_FB3 = (os.getenv("WEAVIATE_CLASS_FB3") or "Chatbot_FB3").strip()

# ======= NEW: Glossary class name =======
# Override with WEAVIATE_CLASS_GLOSSARY=GlossaryEntry (or your preferred name)
GLOSSARY_CLASS = (os.getenv("WEAVIATE_CLASS_GLOSSARY") or "GlossaryEntry").strip()

CLASS_NAMES: List[str] = [CLASS_FB1, CLASS_FB2, CLASS_FB3]

# Handy mapping for API/UI layers using dropdown keys
KEY_TO_CLASS: Dict[str, str] = {
    "FB1": CLASS_FB1,
    "FB2": CLASS_FB2,
    "FB3": CLASS_FB3,
}

# ======= Connection settings (kept from your original) =======
WEAVIATE_HTTP = os.getenv("WEAVIATE_HTTP", "http://localhost:8080").strip()
WEAVIATE_PORT_ENV = (os.getenv("WEAVIATE_PORT", "") or "").strip()
WEAVIATE_GRPC = os.getenv("WEAVIATE_GRPC", "localhost:50051").strip()
WEAVIATE_GRPC_PORT_ENV = (os.getenv("WEAVIATE_GRPC_PORT", "") or "").strip()

def _parse_host_port(http_env: str, port_env: str, default_port: int) -> Tuple[str, int]:
    if "://" not in http_env:
        http_env = f"http://{http_env}"
    parsed = urlparse(http_env)
    host = (parsed.hostname or "localhost").strip()
    port = None
    if port_env:
        try:
            port = int(port_env)
        except Exception:
            port = None
    if port is None:
        port = parsed.port or default_port
    return host, int(port)

def _parse_grpc(grpc_env: str, port_env: str, default_port: int) -> Tuple[str, int]:
    host = grpc_env
    port = None
    if ":" in grpc_env:
        host, _, maybe_port = grpc_env.partition(":")
        try:
            port = int(maybe_port.strip())
        except Exception:
            port = None
    if not host:
        host = "localhost"
    host = host.strip()
    if port is None and port_env:
        try:
            port = int(port_env.strip())
        except Exception:
            port = None
    if port is None:
        port = default_port
    return host, int(port)

HTTP_HOST, HTTP_PORT = _parse_host_port(WEAVIATE_HTTP, WEAVIATE_PORT_ENV, default_port=8080)
GRPC_HOST, GRPC_PORT = _parse_grpc(WEAVIATE_GRPC, WEAVIATE_GRPC_PORT_ENV, default_port=50051)

# Connect (same method you used previously)
client = weaviate.connect_to_local(
    host=HTTP_HOST,
    port=HTTP_PORT,
    grpc_port=GRPC_PORT,
    skip_init_checks=True,
)

def _list_collection_names() -> List[str]:
    cols = client.collections.list_all()
    try:
        return [c.name for c in cols]
    except Exception:
        # Older weaviate clients might already return List[str]
        return list(cols)

def _vector_index_config():
    # HNSW with COSINE distance (fallbacks for older client signatures)
    try:
        return Configure.VectorIndex.hnsw(distance=VectorDistances.COSINE)
    except TypeError:
        try:
            return Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
        except TypeError:
            return Configure.VectorIndex.hnsw()  # fall back to defaults

def _vectorizer_config():
    # Keep "none" to match your existing setup (hybrid/BM25 + your own embeddings if any)
    return Configure.Vectorizer.none()

def _create_collection_if_missing(name: str) -> None:
    existing = _list_collection_names()
    if name in existing:
        return
    client.collections.create(
        name=name,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
            Property(name="room", data_type=DataType.TEXT),    # optional metadata
            Property(name="machine", data_type=DataType.TEXT), # optional metadata
        ],
        vectorizer_config=_vectorizer_config(),
        vector_index_config=_vector_index_config(),
        description=f"RAG chunks for {name}",
    )
    print(f"✅ Created schema for '{name}'")

# ======= NEW: Glossary schema helpers =======

def _create_glossary_if_missing() -> None:
    existing = _list_collection_names()
    if GLOSSARY_CLASS in existing:
        return
    client.collections.create(
        name=GLOSSARY_CLASS,
        properties=[
            Property(name="term",       data_type=DataType.TEXT),  # store UPPERCASE normalized
            Property(name="definition", data_type=DataType.TEXT),
            Property(name="source",     data_type=DataType.TEXT),
            Property(name="page",       data_type=DataType.INT),
        ],
        vectorizer_config=_vectorizer_config(),
        vector_index_config=_vector_index_config(),
        description="Structured glossary (Abkürzungsverzeichnis) entries for fast lookups",
    )
    print(f"✅ Created schema for '{GLOSSARY_CLASS}'")

def ensure_collections() -> None:
    """Ensure all three isolated collections (FB1/FB2/FB3) exist, plus the glossary."""
    for name in CLASS_NAMES:
        _create_collection_if_missing(name)
    _create_glossary_if_missing()  
    _create_chatlog_if_missing()
    
def get_collection(name: str):
    """Return a collection by exact class name (ensuring it exists)."""
    ensure_collections()
    return client.collections.get(name)

def class_from_key(key: str) -> str:
    """Return class name for 'FB1' | 'FB2' | 'FB3'."""
    key = key.upper().strip()
    if key not in KEY_TO_CLASS:
        raise ValueError(f"Unknown key '{key}'. Expected one of {list(KEY_TO_CLASS.keys())}.")
    return KEY_TO_CLASS[key]

def get_collection_by_key(key: str):
    """Return a collection using the dropdown key ('FB1' | 'FB2' | 'FB3')."""
    return get_collection(class_from_key(key))

# ======= NEW: Glossary collection getters =======

def ensure_glossary_collection() -> None:
    """Ensure the glossary collection exists (called by ensure_collections, but usable standalone)."""
    _create_glossary_if_missing()

def get_glossary_collection():
    """Return the glossary collection object (ensuring it exists)."""
    ensure_glossary_collection()
    return client.collections.get(GLOSSARY_CLASS)

def close_client():
    try:
        client.close()
    except Exception:
        pass

# ======= Backward-compat shims =======
# Some legacy modules import `init_schema` and/or `CLASS_NAME`.
# We alias init_schema -> ensure_collections, and expose a harmless CLASS_NAME.
def init_schema() -> None:
    """Legacy alias — old code calls this; new code uses ensure_collections()."""
    ensure_collections()

# Legacy default; not used by multi-bot path but keeps old imports happy.
CLASS_NAME = os.getenv("WEAVIATE_CLASS", "LectureChunk")


# --- NEW: ChatLog class for admin-only chat history ---
CHATLOG_CLASS = (os.getenv("WEAVIATE_CLASS_CHATLOG") or "ChatLog").strip()

def _create_chatlog_if_missing() -> None:
    existing = _list_collection_names()
    if CHATLOG_CLASS in existing:
        return
    client.collections.create(
        name=CHATLOG_CLASS,
        properties=[
            Property(name="session_id",  data_type=DataType.TEXT),
            Property(name="season",      data_type=DataType.TEXT),  # e.g. 2025-Q4
            Property(name="ts",          data_type=DataType.INT),   # timestamp
            Property(name="bot_key",     data_type=DataType.TEXT),  # FB1 | FB2 | FB3
            Property(name="question",    data_type=DataType.TEXT),
            Property(name="answer",      data_type=DataType.TEXT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),
        vector_index_config=_vector_index_config(),
        description="Admin-only chat history grouped by season for analytics",
    )
    print(f"✅ Created schema for '{CHATLOG_CLASS}'")
