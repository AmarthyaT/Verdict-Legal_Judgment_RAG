import chromadb

from main import COLLECTION_NAME

CHROMA_PATH = r"C:/Users/tamar/Desktop/MUNN/chroma_db"
COLLECTION_NAME = "legal_docs_contextual"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

print(f"ChromaDB collection '{COLLECTION_NAME}' has {collection.count()} items.")

collection = chroma_client.get_collection(name=COLLECTION_NAME)
print(f"ChromaDB collection '{COLLECTION_NAME}' has {collection.count()} items.")

collection = chroma_client.get_collection(name="legal_docs_contextual")
print("Collection metadata:", collection.metadata)

import chromadb

CHROMA_PATH = r"C:/Users/tamar/Desktop/MUNN/chroma_db"
COLLECTION_NAME = "legal_docs_contextual"

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

# Print metadata to see the dimension
print("Collection name:", collection.name)
print("Collection metadata:", collection.metadata)



    
    
results = collection.get(include=["embeddings"], limit=1)

# Check if any embeddings were returned
if "embeddings" in results and len(results["embeddings"]) > 0:
    first_emb = results["embeddings"][0]
    print("Embedding dimension:", len(first_emb))
else:
    print("No embeddings found in collection.")