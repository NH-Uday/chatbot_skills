import weaviate
from weaviate.classes.config import Property, DataType
from dotenv import load_dotenv

load_dotenv()

client = weaviate.connect_to_local(
    host="localhost",
    port=8080,
    grpc_port=50051,
    skip_init_checks=True,
)

def init_schema():
    class_name = "LectureChunk"

    existing_collections = client.collections.list_all()

    if class_name in existing_collections:
        print(f"ℹ️ Schema '{class_name}' already exists.")
        return

    client.collections.create(
        name=class_name,
        properties=[
            Property(name="text", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="page", data_type=DataType.INT),
        ],
        # No vectorizer specified — this uses the default ('none' is not supported in 4.16.8)
        description="A chunk of a lecture or technical PDF",
    )

    print(f"✅ Created schema for '{class_name}'")
