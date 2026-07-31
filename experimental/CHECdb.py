import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# --- CONFIGURATION (Must match your embedding script's config) ---
CHROMA_PATH = "./chroma_store_GONECASE2"
CHROMA_COLLECTION_NAME = "legal_judgments_collection"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
# -----------------------------------------------------------------

def check_embedding_success(expected_count: int = 33824):
    """
    Verifies that the ChromaDB vector index was saved successfully and can be reloaded.

    Args:
        expected_count (int): The number of chunks expected to be found in the index.
    """
    print(f"--- Starting Embedding Success Verification for '{CHROMA_PATH}' ---")

    # 1. Check for Persistence (File System Check)
    if not os.path.exists(CHROMA_PATH):
        print(f"❌ FAILURE: ChromaDB folder '{CHROMA_PATH}' not found.")
        return

    # 2. Reload the Embedding Model
    try:
        # Use the consistent HuggingFaceEmbeddings wrapper for reliable reload
        embedding_model_reload = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"✅ SUCCESS: Embedding model '{EMBEDDING_MODEL_NAME}' loaded locally.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to load embedding model. Reason: {e}")
        print("Tip: If this is an SSL error, pre-download the model locally.")
        return

    # 3. Reload the ChromaDB Collection and Get Count
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

        # This reloads the vector store from disk
        reloaded_vectorstore = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model_reload,
        )

        # Get the actual count of vectors stored
        actual_count = chroma_client.get_collection(CHROMA_COLLECTION_NAME).count()

        if actual_count == expected_count:
            print(f"\n🎉 **EMBEDDING SUCCESSFUL!** 🎉")
            print(f"Index loaded correctly. Found {actual_count} chunks (Matches expected count).")
        else:
            print(f"\n⚠️ **WARNING: COUNT MISMATCH.**")
            print(f"Index loaded, but found {actual_count} chunks. Expected {expected_count}.")
            print("This may indicate some files failed during the initial embedding process.")

    except Exception as e:
        print(f"❌ FAILURE: Failed to reload ChromaDB index. Reason: {e}")


if __name__ == '__main__':
    # You reported 33824 chunks stored in your output, so we use that as the expected count.
    check_embedding_success(expected_count=33824)