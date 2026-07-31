import os
import time
import argparse
import fitz  # PyMuPDF
import chromadb
from dotenv import load_dotenv
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import torch
from transformers import AutoTokenizer, AutoModel

# --- CONFIGURATION ---
load_dotenv()

# HUGGINGFACE EMBEDDING MODEL
HUGGINGFACE_MODEL = "BAAI/bge-large-en"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading embedding model:", HUGGINGFACE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
model = AutoModel.from_pretrained(HUGGINGFACE_MODEL).to(device)

# PDF PATH
DIRECTORY_PATH = "C:/Users/tamar/Desktop/judg/pdfs"

# CHROMA
CHROMA_PATH = "C:/Users/tamar/Desktop/judg/chroma_db1"
COLLECTION_NAME = "legal_docs_contextual"

# CHUNKING SETTINGS
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# BATCH EMBEDDING SETTINGS
BATCH_SIZE = 32 # Define a batch size for embeddings

# --- DATABASE SETUP ---
print(f"--- Initializing Vector Database at {CHROMA_PATH} ---")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

# -----------------------------------------
# BATCH EMBEDDING FUNCTION
# -----------------------------------------
def embed_hf(texts: list[str]) -> list[list[float]]:
    # Tokenize the batch of texts
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(device)
    with torch.no_grad():
        # Get model output
        model_output = model(**encoded_input)
        # Pool embeddings using CLS token and normalize
        embeddings = model_output.last_hidden_state[:, 0].cpu().tolist()
    return embeddings

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_path):
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
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# -----------------------------------------
# CONTEXT GENERATION (Gemini removed)
# NOW MODEL-AGNOSTIC – simple heuristic context
# -----------------------------------------

def generate_chunk_context(document_text, chunk_text):
    """
    Replace Gemini with a simple contextual tag:
    - Finds first heading above the chunk
    - Produces 1–2 sentence summary
    """
    chunk_start = document_text.find(chunk_text[:100])

    # Simple heuristic: get nearby headings or section markers
    context_area = document_text[max(0, chunk_start - 2000): chunk_start]

    lines = context_area.split("\n")
    heading = None
    for line in reversed(lines):
        if line.strip().isupper() and 4 <= len(line.strip()) <= 120:
            heading = line.strip()
            break

    summary = f"This section relates to: {heading if heading else 'a legal discussion earlier in the document'}."

    return summary


# -----------------------------------------
# HF EMBEDDING WITH RETRIES
# -----------------------------------------
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    retry=retry_if_exception_type(Exception)
)
def get_embeddings_batched(texts: list[str]) -> list[list[float]]:
    return embed_hf(texts)


# -----------------------------------------
# MAIN PIPELINE
# -----------------------------------------
def process_directory(directory_path, use_contextual_rag=True, start_idx_for_queue=0):

    print(f"Scanning '{directory_path}' recursively for PDFs...")
    pdf_files_paths = []
    empty_files_count = 0

    for root, dirs, files in os.walk(directory_path):
        for f in files:
            if f.lower().endswith(".pdf"):
                file_path = os.path.join(root, f)
                if os.path.getsize(file_path) > 0: # Check if file is not empty
                    pdf_files_paths.append(file_path)
                else:
                    print(f"Skipping empty file: {file_path}")
                    empty_files_count += 1

    total_files = len(pdf_files_paths) + empty_files_count
    print(f"Found {total_files} PDF files ({len(pdf_files_paths)} non-empty, {empty_files_count} empty).")

    if len(pdf_files_paths) == 0:
        print("No non-empty PDFs found.")
        return

    # RESUME LOGIC
    try:
        existing_data = collection.get(include=[])
        existing_ids = set(existing_data["ids"])
        processed_filenames = set([eid.split("_chunk_")[0] for eid in existing_ids])
    except Exception as e:
        print(f"Error retrieving existing ChromaDB data: {e}. Assuming no existing data.")
        processed_filenames = set()

    files_to_process_initial = [p for p in pdf_files_paths if os.path.basename(p) not in processed_filenames]
    files_to_process_initial = files_to_process_initial[:500]  # Limiting for original task demonstration

    num_skipped_by_db = len(pdf_files_paths) - len(files_to_process_initial)

    print(f"Skipping {num_skipped_by_db} files already in ChromaDB.")

    files_to_process = []
    if start_idx_for_queue > 0 and start_idx_for_queue < len(files_to_process_initial):
        print(f"Starting ingestion from file index {start_idx_for_queue} in the remaining queue, skipping the first {start_idx_for_queue} files as requested.")
        files_to_process = files_to_process_initial[start_idx_for_queue:]
    elif start_idx_for_queue >= len(files_to_process_initial):
        print("Requested start index is beyond the number of files to process. No files will be processed in this run.")
    else: # start_idx_for_queue is 0 or negative, process all initial files
        files_to_process = files_to_process_initial

    print(f"Starting ingestion for {len(files_to_process)} new files...")
    print(f"Contextual RAG is: {'ENABLED' if use_contextual_rag else 'DISABLED'}")

    # -----------------------------------------
    # PROCESS EACH PDF
    # -----------------------------------------
    for file_path in tqdm(files_to_process, desc="Ingesting Docs"):
        filename = os.path.basename(file_path)

        # EXTRACT TEXT
        full_text = extract_text_from_pdf(file_path)
        if not full_text or len(full_text) < 50:
            continue

        # CHUNK
        raw_chunks = simple_chunker(full_text)

        # Temporary lists to hold data for batching
        current_batch_texts_to_embed = []
        current_batch_documents = []
        current_batch_ids = []
        current_batch_metadatas = []

        # LOOP OVER CHUNKS
        for i, chunk in enumerate(raw_chunks):

            text_to_embed = chunk

            # CONTEXTUAL RAG (REPLACES GEMINI)
            if use_contextual_rag:
                try:
                    context = generate_chunk_context(full_text, chunk)
                    text_to_embed = context + "\n\n" + chunk
                except Exception as e:
                    # Fallback if context generation fails
                    print(f"Warning: Context generation failed for chunk {i} of {filename}: {e}")
                    pass # Use original chunk if context generation fails

            chunk_id = f"{filename}_chunk_{i}"

            # Add to current batch lists
            current_batch_texts_to_embed.append(text_to_embed)
            current_batch_documents.append(chunk)
            current_batch_ids.append(chunk_id)
            current_batch_metadatas.append({
                "source": filename,
                "path": file_path,
                "chunk_index": i,
                "contextualized": use_contextual_rag
            })

            # If batch size is reached or it's the last chunk, process the batch
            if len(current_batch_texts_to_embed) >= BATCH_SIZE or (i == len(raw_chunks) - 1 and current_batch_texts_to_embed):
                try:
                    # EMBED BATCH
                    embeddings_batch = get_embeddings_batched(current_batch_texts_to_embed)

                    # SAVE TO DB
                    if embeddings_batch:
                        collection.add(
                            ids=current_batch_ids,
                            embeddings=embeddings_batch,
                            documents=current_batch_documents,
                            metadatas=current_batch_metadatas
                        )

                except Exception as e:
                    print(f"Error embedding or adding batch for {filename}: {e}")

                # Reset batch lists
                current_batch_texts_to_embed = []
                current_batch_documents = []
                current_batch_ids = []
                current_batch_metadatas = []


if __name__ == "__main__":
    # Re-running the full ingestion process with start_idx_for_queue=0
    # to ensure ChromaDB is populated after deletion and restart.
    process_directory(DIRECTORY_PATH, use_contextual_rag=True, start_idx_for_queue=0)