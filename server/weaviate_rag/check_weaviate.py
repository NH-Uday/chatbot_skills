import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.weaviate_setup import client, close_client

def list_with_counts():
    cols = client.collections.list_all()
    names = []
    try:
        # Newer clients return objects with .name
        names = [c.name for c in cols]
    except Exception:
        # Older clients return list[str]
        names = list(cols)

    out = []
    for name in names:
        try:
            total = client.collections.get(name).aggregate.over_all().total_count
        except Exception:
            total = "n/a"
        out.append((name, total))
    return out

print("WEAVIATE_HTTP =", os.getenv("WEAVIATE_HTTP"))
print("WEAVIATE_PORT =", os.getenv("WEAVIATE_PORT"))
print("WEAVIATE_GRPC =", os.getenv("WEAVIATE_GRPC"))
print("WEAVIATE_GRPC_PORT =", os.getenv("WEAVIATE_GRPC_PORT"))

print("Collections (name, total_count):")
for name, cnt in list_with_counts():
    print(" -", name, cnt)

close_client()
