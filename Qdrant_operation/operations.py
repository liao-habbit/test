from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
url = "http://localhost:6333"
def to_create_collection(url, name, vector_size = 768):
    client = QdrantClient(url = url)
    existing_collections = [c.name for c in client.get_collections().collections]
    if name in existing_collections:
        print(f"Collection {name} 已存在，跳過建立。")
    else:
        client.create_collection(
            collection_name=str(name),
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(f"Collection {name} 已建立。✅")

def to_delete_collection(url, name):
    client = QdrantClient(url = url)
    existing_collections = [c.name for c in client.get_collections().collections]
    if name in existing_collections:
        client.delete_collection(collection_name=name)
        print(f"Collection {name} 已刪除。✅")
    else:
        print(f"Collection {name} 不存在。")


def upload(url, embedding_list, metadata, collection):
    client = QdrantClient(url = url)
    import uuid
    def uuid64():
        return uuid.uuid4().int >> 64
    points = []
    for vec, meta in zip(embedding_list, metadata):
        points.append({
            "id": uuid64(),   # 新的唯一 ID
            "vector": vec,
            "payload": meta
        })
    client.upsert(
        collection_name=collection,
        points=points
    )
    print(f"資料已新增到 {collection}✅")

# 測試用
# to_create_collection(url = url, name = "test", vector_size = 768)
# to_delete_collection(url = url, name = "test")
# test_embedding_list = [
#     [0.11] * 768,
#     [0.22] * 768
# ]

# test_payloads = [
#     {
#         "filename": "test1.jpg",
#         "objects": [{"name": "leaf_rust"}]
#     },
#     {
#         "filename": "test2.jpg",
#         "objects": [{"name": "brown_spot"}]
#     }
# ]
# upload(url, test_embedding_list, test_payloads, "test")

