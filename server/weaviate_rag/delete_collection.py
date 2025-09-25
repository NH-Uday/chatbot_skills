import os
import sys
import argparse

# If your services are inside app/services:
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

from app.services.weaviate_setup import (  # type: ignore
    client,
    KEY_TO_CLASS,     
    CLASS_FB1,
    CLASS_FB2,
    CLASS_FB3,
)

BOT_CLASSES = [CLASS_FB1, CLASS_FB2, CLASS_FB3]


def list_collections() -> list[str]:
    cols = client.collections.list_all()
    try:
        return [c.name for c in cols]
    except Exception:
        # Compatibility with older weaviate clients that return a list[str]
        return list(cols)


def delete_by_class_name(class_name: str) -> bool:
    names = list_collections()
    if class_name in names:
        client.collections.delete(class_name)
        print(f"✅ Deleted collection '{class_name}'")
        return True
    else:
        print(f"ℹ️ Collection '{class_name}' not found. Existing: {names}")
        return False


def delete_by_key(key: str) -> bool:
    key = key.upper().strip()
    if key not in KEY_TO_CLASS:
        print(f"❌ Unknown key '{key}'. Expected one of {list(KEY_TO_CLASS.keys())}.")
        return False
    return delete_by_class_name(KEY_TO_CLASS[key])


def delete_all_three() -> None:
    found_any = False
    for cn in BOT_CLASSES:
        if delete_by_class_name(cn):
            found_any = True
    if not found_any:
        print("ℹ️ None of the three bot collections were present.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Delete Weaviate collections for the 3 isolated PDF bots."
    )
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument("--key", choices=["FB1", "FB2", "FB3"], help="Delete by bot key.")
    g.add_argument("--class-name", help="Delete by exact class name.")
    g.add_argument("--all-three", action="store_true", help="Delete all three bot collections.")
    p.add_argument("--list", action="store_true", help="List existing collections and exit.")
    return p.parse_args()


def main():
    try:
        args = parse_args()

        if args.list:
            names = list_collections()
            print("📚 Collections:", names)
            return

        if args.all_three:
            delete_all_three()
            return

        if args.key:
            delete_by_key(args.key)
            return

        if args.class_name:
            delete_by_class_name(args.class_name)
            return

        # Default behavior if no flags: just list
        names = list_collections()
        print("📚 Collections:", names)

    finally:
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


#python delete_collection.py --list

#python delete_collection.py --key FB2

#python delete_collection.py --class-name Chatbot_FB1

#python delete_collection.py --all-three

#python delete_collection.py --class-name LectureChunk

