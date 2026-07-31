import os
import time
import argparse
import fitz  # PyMuPDF
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# --- CONFIGURATION ---
load_dotenv()

# 1. API KEY CHECK
if "GOOGLE_API_KEY" not in os.environ:
    # You can manually set it here if .env is failing:
    # os.environ["GOOGLE_API_KEY"] = "your_key_here"
    pass

if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("CRITICAL ERROR: GOOGLE_API_KEY not found. Set it in .env or environment variables.")

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# 2. SETTINGS
CHROMA_PATH = "C:/Users/tamar/Desktop/judg/pdfs"
COLLECTION_NAME = "legal_docs_contextual"
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-1.5-flash"
CHUNK_SIZE = 1000 
CHUNK_OVERLAP = 200

# --- DATABASE SETUP ---
print(f"--- Initializing Vector Database at {CHROMA_PATH} ---")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# --- HELPER FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """Fast text extraction using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def simple_chunker(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# --- AI LOGIC (CONTEXTUAL RETRIEVAL) ---

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(Exception)
)
def generate_chunk_context(document_text, chunk_text):
    """
    Uses Gemini to generate context for a chunk.
    """
    model = genai.GenerativeModel(GENERATION_MODEL)
    
    # Truncate doc context to ~50k chars to save tokens
    truncated_doc = document_text[:50] 

    prompt = f"""
    <document>
    {truncated_doc}
    ... (truncated) ...
    </document>

    <chunk>
    {chunk_text}
    </chunk>

    Please provide a short, specific context (1-2 sentences) that situates this chunk within the overall legal document. 
    Explain what this specific section refers to.
    """
    
    response = model.generate_content(prompt)
    return response.text.strip()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20)
)
def get_embedding(text):
    """Gets embedding from Gemini API with exponential backoff retry."""
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_document"
    )
    return result['embedding']

# --- MAIN PIPELINE ---

def process_directory(directory_path, use_contextual_rag=True):
    """
    Scans recursively for PDFs, chunks them, adds context, embeds, and stores.
    """
    
    # 1. FIND FILES (RECURSIVE)
    print(f"Scanning '{directory_path}' recursively for PDFs...")
    pdf_files_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files_paths.append(os.path.join(root, file))

    total_files = len(pdf_files_paths)
    print(f"Found {total_files} PDF files.")

    if total_files == 0:
        print("No PDFs found. Exiting.")
        return

    # 2. RESUME LOGIC
    try:
        existing_data = collection.get(include=[]) 
        existing_ids = set(existing_data['ids'])
        processed_filenames = set([eid.split('_chunk_')[0] for eid in existing_ids])
    except:
        processed_filenames = set()
    
    files_to_process = [
        p for p in pdf_files_paths 
        if os.path.basename(p) not in processed_filenames
    ]
    
    print(f"Skipping {total_files - len(files_to_process)} processed files.")
    print(f"Starting ingestion for {len(files_to_process)} new files...")
    print(f"Contextual RAG is: {'ENABLED' if use_contextual_rag else 'DISABLED'}")

    # 3. PROCESSING LOOP
    for file_path in tqdm(files_to_process, desc="Ingesting Docs"):
        filename = os.path.basename(file_path)
        
        # A. Extract
        full_text = extract_text_from_pdf(file_path)
        if not full_text or len(full_text) < 50:
            continue 

        # B. Chunk
        raw_chunks = simple_chunker(full_text)
        
        ids_batch = []
        documents_batch = []
        embeddings_batch = []
        metadatas_batch = []

        # C. Contextualize & Embed Loop
        for i, chunk in enumerate(raw_chunks):
            
            text_to_embed = chunk
            
            # --- CONTEXTUAL STEP ---
            if use_contextual_rag:
                try:
                    context = generate_chunk_context(full_text, chunk)
                    text_to_embed = f"{context}\n\n{chunk}"
                except Exception as e:
                    pass
            
            # D. Embed
            try:
                vector = get_embedding(text_to_embed)
                
                chunk_id = f"{filename}_chunk_{i}"
                
                ids_batch.append(chunk_id)
                documents_batch.append(chunk)
                embeddings_batch.append(vector)
                metadatas_batch.append({
                    "source": filename,
                    "path": file_path,
                    "chunk_index": i,
                    "contextualized": use_contextual_rag
                })
            except Exception as e:
                # Only print critical errors, skip chunks that fail
                pass

        # E. Save Batch to DB
        if ids_batch:
            collection.add(
                ids=ids_batch,
                embeddings=embeddings_batch,
                documents=documents_batch,
                metadatas=metadatas_batch
            )
        
        # F. Rate Limit Pause
        time.sleep(1.0) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Legal Doc Ingestion")
    parser.add_argument("folder", help="Path to the folder containing PDF files")
    parser.add_argument("--no-context", action="store_true", help="Disable AI context")
    
    args = parser.parse_args()

    if os.path.exists(args.folder):
        process_directory(args.folder, use_contextual_rag=not args.no_context)
    else:
        print(f"Error: Directory '{args.folder}' does not exist.")