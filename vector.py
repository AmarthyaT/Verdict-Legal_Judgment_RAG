import os
import chromadb
import tiktoken
from PyPDF2 import PdfReader
from nomic import embed  # ✅ Nomic embedding client

# --- CONFIG ---
PDF_FOLDER = "pdfs"
MAX_TOKENS = 500
OVERLAP_TOKENS = 90
COLLECTION_NAME = "legal-pdf-embeddings"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_store")

# --- INIT CHROMA ---
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)

# --- UTILS ---
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text_tokenwise(text, max_tokens=MAX_TOKENS, overlap=OVERLAP_TOKENS):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk = enc.decode(tokens[start:end])
        chunks.append(chunk)
        start += max_tokens - overlap
    return chunks

def embed_chunk(chunk):
    try:
        response = embed.text(
            texts=[chunk],
            model="nomic-embed-text-v1.5",
            task_type="search_document"
        )
        return response["embeddings"][0]
    except Exception as e:
        print(f"⚠️ Embedding failed: {e}")
        return None

# --- MAIN LOOP ---
doc_id = 0
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        path = os.path.join(PDF_FOLDER, filename)
        print(f"📄 Processing: {filename}")
        text = extract_text_from_pdf(path)
        chunks = chunk_text_tokenwise(text)

        for i, chunk in enumerate(chunks):
            embedding = embed_chunk(chunk)
            if embedding is None:
                continue

            uid = f"{filename}-{i}"
            collection.add(
                ids=[uid],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"source": filename, "chunk": i}]
            )
            doc_id += 1

print(f"\n✅ Embedded {doc_id} chunks into persistent ChromaDB collection at: {CHROMA_PATH}")