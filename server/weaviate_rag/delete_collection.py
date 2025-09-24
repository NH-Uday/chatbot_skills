import os, sys

# If your services are inside app/services:
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.weaviate_setup import client, CLASS_NAME  # type: ignore

def list_names():
    cols = client.collections.list_all()
    try:
        return [c.name for c in cols]
    except Exception:
        return list(cols)

def main():
    try:
        names = list_names()
        if CLASS_NAME in names:
            client.collections.delete(CLASS_NAME)
            print(f"✅ Deleted collection '{CLASS_NAME}'")
        else:
            print(f"ℹ️ Collection '{CLASS_NAME}' not found. Existing: {names}")
    finally:
        try:
            client.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
