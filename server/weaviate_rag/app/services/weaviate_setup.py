# app/services/weaviate_setup.py
import os
from urllib.parse import urlparse
from pathlib import Path
from dotenv import load_dotenv
import weaviate
from weaviate.classes.config import Property, DataType, Configure, VectorDistances

# ----- Load .env from server/.env -----
THIS_FILE = Path(__file__).resolve()
ENV_PATH = THIS_FILE.parents[3] / ".env"   # server/.env
load_dotenv(dotenv_path=str(ENV_PATH))

CLASS_NAME = os.getenv("WEAVIATE_CLASS", "LectureChunk").strip()

WEAVIATE_HTTP = os.getenv("WEAVIATE_HTTP", "http://localhost:8080").strip()
WEAVIATE_PORT_ENV = (os.getenv("WEAVIATE_PORT", "") or "").strip()
WEAVIATE_GRPC = os.getenv("WEAVIATE_GRPC", "localhost:50051").strip()
WEAVIATE_GRPC_PORT_ENV = (os.getenv("WEAVIATE_GRPC_PORT", "") or "").strip()

def _parse_host_port(http_env: str, port_env: str, default_port: int) -> tuple[str, int]:
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

def _parse_grpc(grpc_env: str, port_env: str, default_port: int) -> tuple[str, int]:
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

client = weaviate.connect_to_local(
    host=HTTP_HOST,
    port=HTTP_PORT,
    grpc_port=GRPC_PORT,
    skip_init_checks=True,
)

def _list_collection_names():
    cols = client.collections.list_all()
    try:
        return [c.name for c in cols]
    except Exception:
        return list(cols)

def _vector_index_config():
    
    try:
        return Configure.VectorIndex.hnsw(distance=VectorDistances.COSINE)
    except TypeError:
        try:
            return Configure.VectorIndex.hnsw(distance_metric=VectorDistances.COSINE)
        except TypeError:
            return Configure.VectorIndex.hnsw()  # fall back to defaults

def init_schema() -> None:
    existing = _list_collection_names()
    if CLASS_NAME in existing:
        return
    client.collections.create(
        name=CLASS_NAME,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
        ],
        vectorizer_config=Configure.Vectorizer.none(),  # push vectors manually
        vector_index_config=_vector_index_config(),
        description="RAG chunks from uploaded PDFs with manual vectors",
    )
    print(f"✅ Created schema for '{CLASS_NAME}'")

def close_client():
    try:
        client.close()
    except Exception:
        pass
