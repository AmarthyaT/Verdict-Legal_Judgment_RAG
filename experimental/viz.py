import chromadb
import pandas as pd
from renumics import spotlight

# 1. Connect to your existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_store_GONECASE2")
collection = client.get_collection("legal_judgments_collection")

# 2. Extract EVERYTHING (documents, embeddings, and metadata)
# Note: include=['embeddings'] is vital for the 3D map
data = collection.get(include=['embeddings', 'documents', 'metadatas'])

# 3. Flatten the metadata and create a DataFrame
df = pd.DataFrame({
    "id": data["ids"],
    "document": data["documents"],
    "embedding": data["embeddings"]
})

# Add metadata fields as separate columns for filtering
metadata_df = pd.DataFrame(data["metadatas"])
df = pd.concat([df, metadata_df], axis=1)

# 4. Launch Spotlight
# Spotlight will automatically detect the 'embedding' column 
# and generate a 3D Similarity Map using UMAP.
spotlight.show(df, dtype={"embedding": spotlight.Embedding})