import os
import sys
import chromadb
import statistics
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
COLLECTION_NAME = "supreme_court_judgments"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_store1")
MODEL_NAME = "all-MiniLM-L6-v2"  # A good default model for semantic search
# ensure the directory for the persistent store exists
os.makedirs(CHROMA_PATH, exist_ok=True)
# Initialize LLM with correct model name
try:
    llm = OllamaLLM(model="llama3")  # Changed from llama3 to llama2
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    exit()


# --- INITIALIZE MODELS AND DATABASE ---
# Initialize Chroma client first and handle errors separately
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
except Exception as e:
    print(f"❌ Failed to initialize Chroma client: {e}")
    exit()

# --- PROMPT TEMPLATE ---
template = """
You are an expert legal analyst. You will be provided with excerpts from a legal judgment retrieved based on a user's question.
Your task is to synthesize the information from these excerpts to answer the user's question and provide a structured summary based on the available text.

give you full scale summery of the document based on the excerpts.
{document}
---
**User's Question:**
{question}
---
**Your Answer and Summary:**
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | llm

# --- UTILITY FUNCTION ---
def embed_query(query_text):
    """Embeds the user's query using sentence-transformers."""
    try:
        # Initialize the model (will download if not present)
        model = SentenceTransformer(MODEL_NAME)
        # Generate embedding
        embedding = model.encode(query_text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        print(f"⚠️ Query embedding failed: {e}")
        return None

# --- DIAGNOSTICS ---
def diagnose_collection():
    """Print quick diagnostics: total count and a few sample entries (id, preview, metadata keys)."""
    try:
        cnt = collection.count()
        print(f"ℹ️ Diagnostic: collection '{COLLECTION_NAME}' contains {cnt} items.")
        sample_limit = min(5, max(0, cnt))
        if sample_limit == 0:
            print("ℹ️ No sample entries available (collection empty).")
            return
        sample = collection.get(limit=sample_limit, include=['ids', 'documents', 'metadatas'])
        ids = sample.get('ids', [])
        docs = sample.get('documents', [])
        metas = sample.get('metadatas', [])
        print("ℹ️ Sample entries:")
        for i in range(len(ids)):
            doc_preview = (docs[i] or "")[:120].replace("\n", " ") if i < len(docs) else ""
            meta_keys = list(metas[i].keys()) if i < len(metas) and isinstance(metas[i], dict) else metas[i]
            print(f"  {i+1}. id={ids[i]} preview='{doc_preview}' metadata_keys={meta_keys}")
    except Exception as e:
        print(f"⚠️ diagnose_collection failed: {e}")

# --- ADDITIONAL DIAGNOSTICS: FILESYSTEM ---
def diagnose_store_files():
    """Inspect CHROMA_PATH contents and attempt to identify persistent files (sqlite, etc.)."""
    print(f"ℹ️ Inspecting chroma store directory: {CHROMA_PATH}")
    try:
        if not os.path.exists(CHROMA_PATH):
            print(f"⚠️ Path does not exist: {CHROMA_PATH}")
            return

        # List top-level files and directories with sizes
        for root, dirs, files in os.walk(CHROMA_PATH):
            print(f"  {root}:")
            for d in dirs:
                print(f"    [dir]  {d}")
            for f in files:
                p = os.path.join(root, f)
                try:
                    size = os.path.getsize(p)
                    print(f"    [file] {f} ({size} bytes)")
                except Exception:
                    print(f"    [file] {f} (size unknown)")
            # show only top-level listing to keep output concise
            break

        # Top-level sqlite detection
        try:
            top_files = os.listdir(CHROMA_PATH)
        except Exception:
            top_files = []
        sqlite_candidates = [f for f in top_files if f.lower().endswith('.sqlite') or f.lower().endswith('.sqlite3')]
        if sqlite_candidates:
            print("ℹ️ Found sqlite files at top level:", sqlite_candidates)
        else:
            print("ℹ️ No sqlite files found at top-level of chroma store. Check subdirectories or whether data was persisted to a different path.")

        # If chroma client can list collections, call it
        if hasattr(chroma_client, "list_collections"):
            try:
                cols = chroma_client.list_collections()
                print("ℹ️ Collections reported by chroma client:", cols)
            except Exception as e:
                print(f"⚠️ chroma_client.list_collections() failed: {e}")
        else:
            print("ℹ️ chroma_client has no 'list_collections' method; cannot list collections programmatically.")
    except Exception as e:
        print(f"⚠️ diagnose_store_files failed: {e}")

# New helper: more thorough listing of collections (uses client API if available, filesystem fallback)
def list_collections_verbose():
    print("ℹ️ Listing collections (client API preferred, then filesystem fallback)...")
    # 1) Try client API
    try:
        if hasattr(chroma_client, "list_collections"):
            cols = chroma_client.list_collections()
            print("ℹ️ Collections reported by chroma client:")
            # handle common shapes (list of names or list of dicts)
            if isinstance(cols, list):
                for c in cols:
                    print(f"  - {c}")
            else:
                print(f"  {cols}")
            return
    except Exception as e:
        print(f"⚠️ chroma_client.list_collections() failed: {e}")

    # 2) Filesystem fallback: look for common Chroma persistent structures
    print("ℹ️ Fallback: inspecting CHROMA_PATH for possible collections and files:")
    try:
        if not os.path.exists(CHROMA_PATH):
            print(f"  ⚠️ CHROMA_PATH does not exist: {CHROMA_PATH}")
            return
        for root, dirs, files in os.walk(CHROMA_PATH):
            rel_root = os.path.relpath(root, CHROMA_PATH)
            print(f"  {rel_root if rel_root != '.' else CHROMA_PATH}:")
            for d in sorted(dirs):
                print(f"    [dir]  {d}")
            for f in sorted(files):
                p = os.path.join(root, f)
                try:
                    size = os.path.getsize(p)
                    print(f"    [file] {f} ({size} bytes)")
                except Exception:
                    print(f"    [file] {f} (size unknown)")
            # show only top-level entries for brevity
            break

        # common locations: look for a 'collections' dir or sqlite files
        top = sorted(os.listdir(CHROMA_PATH))
        collections_dir = [n for n in top if n.lower() == "collections" or n.lower().startswith("collection")]
        sqlite_candidates = [n for n in top if n.lower().endswith('.sqlite') or n.lower().endswith('.sqlite3') or n.lower().endswith('.db')]
        if collections_dir:
            print("  ℹ️ Possible collections directories:", collections_dir)
            # list subdirs if any
            for cdir in collections_dir:
                path = os.path.join(CHROMA_PATH, cdir)
                try:
                    subs = sorted([s for s in os.listdir(path) if os.path.isdir(os.path.join(path, s))])
                    if subs:
                        print(f"    subdirs in {cdir}: {subs}")
                except Exception:
                    pass
        if sqlite_candidates:
            print("  ℹ️ Found sqlite/db-like files at top-level:", sqlite_candidates)
        if not collections_dir and not sqlite_candidates:
            print("  ℹ️ No obvious collection files found; data may be stored in an alternate layout or a different path was used when persisting.")
    except Exception as e:
        print(f"⚠️ Filesystem inspection failed: {e}")

# call the helper to show what's in the store at startup
list_collections_verbose()

# Ensure collection exists (create if missing)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception:
    print(f"⚠️ Collection '{COLLECTION_NAME}' not found. Available entries in the store:")
    list_collections_verbose()
    try:
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
        print(f"ℹ️ Created new collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"❌ Failed to create collection '{COLLECTION_NAME}': {e}")
        exit()

# print basic collection info for quick diagnostics
try:
    total_items = collection.count()
    print(f"ℹ️ Collection '{COLLECTION_NAME}' contains {total_items} items.")
    if total_items == 0:
        # call filesystem-level diagnostic if the collection is empty
        try:
            diagnose_store_files()
        except NameError:
            # diagnose_store_files defined below; if not yet defined, it's fine - will be available later
            pass
except Exception:
    print("ℹ️ Could not read collection count at startup.")


# --- MAIN EXECUTION ---
def main():
    """Main loop to accept user questions and generate answers."""
    while True:
        question = input("Ask a question about your documents (or enter 'q' to quit): ")
        if question.lower() == 'q':
            print("Exiting...")
            break

        print("\nEmbedding your query...")
        query_embedding = embed_query(question)
        if query_embedding is None:
            print("⚠️ Could not embed query. Please try again.")
            continue

        print("🔍 Searching for relevant document chunks...")
        results = None
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=15  # Retrieve the top 15 most relevant chunks
            )
        except Exception as e:
            print(f"⚠️ Semantic query failed: {e}")

        documents = results.get("documents") if results else None
        if not documents or not documents[0]:
            # fallback: try text-based query (useful if embeddings were not created/compatible)
            print("⚠️ No semantic results. Attempting fallback text query...")
            try:
                text_results = collection.query(query_texts=[question], n_results=15)
                text_documents = text_results.get("documents")
                if text_documents and text_documents[0]:
                    documents = text_documents
                    print("✅ Text fallback returned results.")
                else:
                    print("⚠️ Text fallback returned no results.")
                    diagnose_collection()
                    continue
            except Exception as e:
                print(f"⚠️ Fallback text query failed: {e}")
                diagnose_collection()
                continue

        # Using the top N chunks is a robust strategy for context
        relevant_chunks = documents[0]
        combined_text = "\n\n---\n\n".join(relevant_chunks)

        print(f"✅ Found {len(relevant_chunks)} relevant chunks. Generating response...")

        try:
            response = chain.invoke({
                "document": combined_text,
                "question": question
            })
            print("\n" + "="*50 + " RESPONSE " + "="*50)
            print(response)
            print("="*110 + "\n")
        except Exception as e:
            print(f"⚠️ An error occurred while generating the response: {e}")

if __name__ == "__main__":
    # support quick check: python main.py --list-collections
    if len(sys.argv) > 1 and sys.argv[1] in ("--list-collections", "--list"):
        list_collections_verbose()
        sys.exit(0)
    main()


