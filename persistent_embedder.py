import os
import hashlib
import torch
import json
from typing import List, Dict, Any
from pypdf import PdfReader, errors as pypdf_errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm 

# --- Libraries for ChromaDB Persistence ---
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LC_Document
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- CONFIGURATION ---
CHROMA_PATH = "./data/chroma_store" 
CHROMA_COLLECTION_NAME = "legal_judgments_collection"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 100
METADATA_JSON_PATH = "./data/structured_judgments.json" 

# --------------------------------------------------------------------------
# UTILITY FUNCTIONS
# --------------------------------------------------------------------------

def generate_doc_id(file_path: str) -> str:
    """Creates a unique, persistent identifier (DocID) based on file hash."""
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()[:12].upper()
    except Exception:
        return "ERROR_ID" 

def extract_metadata_from_json(
    file_name_key: str, 
    doc_id: str, 
    case_metadata_map: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """Extracts structured legal metadata from the pre-loaded JSON map."""
    # Use the normalized key for safe lookup
    judgment_data = case_metadata_map.get(file_name_key, {}) 
    
    petitioner = judgment_data.get("Petitioner", "N/A")
    respondent = judgment_data.get("Respondent", "N/A")
    year = judgment_data.get("Year", "Unknown")
    case_title = f"{petitioner} vs {respondent} ({year})"
    
    metadata = {
        "SourceID": doc_id,
        "CaseTitle": case_title, 
        "Year": year,
        "Petitioner": petitioner,
        "Respondent": respondent,
        
        "VerdictDate": judgment_data.get("VerdictDate", "N/A"), 
        "Bench": judgment_data.get("Bench", "N/A"), 
        "Citation": judgment_data.get("Citation", "N/A"), 
        "KeyLegalIssue": judgment_data.get("KeyLegalIssue", "N/A"),
        
        "EntityType": "CaseLaw",
        "IsHeadnote": True 
    }
    return metadata

def extract_metadata_from_filename(file_name: str, doc_id: str) -> Dict[str, Any]:
    """Creates generic metadata when using filename fallback."""
    raw_title = os.path.splitext(file_name)[0]
    case_title = raw_title.replace("-0__", "").replace("_", " ").strip().title()
    
    metadata = {
        "SourceID": doc_id,
        "CaseTitle": f"[UNSTRUCTURED] {case_title}",
        "Year": "N/A (Filename-based)",
        "Petitioner": case_title,
        "Respondent": "N/A",
        
        "VerdictDate": "N/A", 
        "Bench": "N/A",
        "Citation": "N/A", 
        "KeyLegalIssue": "N/A",
        
        "EntityType": "CaseLaw",
        "IsHeadnote": True 
    }
    return metadata

def clean_extracted_text(text: str) -> str:
    """Aggressively cleans text to remove encoding and non-ASCII issues."""
    if not text:
        return ""
    # Remove non-ASCII characters and clean up excessive whitespace/control characters
    cleaned_text = (
        text
        .encode('ascii', 'ignore')
        .decode('utf-8')
        .replace('\n\n\n', '\n\n')
        .replace('\r', '')
        .strip()
    )
    return cleaned_text


# --------------------------------------------------------------------------
# CORE EMBEDDING WORKFLOW FUNCTION
# --------------------------------------------------------------------------

def embed_legal_documents(pdf_folder: str, embedding_model: HuggingFaceEmbeddings, metadata_json_path: str):
    """
    Implements the persistent Parent/Child RAG Embedding Process with key normalization 
    and robust text cleaning.
    """
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"Error: No PDF files found in '{pdf_folder}'. Check the folder path.")
        return
    
    # 1. Load and Normalize JSON Metadata (CRITICAL FIX)
    try:
        with open(metadata_json_path, 'r', encoding='utf-8') as f:
            raw_metadata = json.load(f)
        
        # 🌟 FIX: Normalize all keys to lowercase for robust matching
        case_metadata_map = {k.lower(): v for k, v in raw_metadata.items()}
        print(f"Loaded and normalized {len(case_metadata_map)} case entries from the JSON.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load JSON metadata from '{metadata_json_path}'. Reason: {e}")
        return

    # 2. Initialize the Paragraph-based Splitter
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""] 
    )
    
    all_child_documents: List[LC_Document] = []
    parent_metadata_store: Dict[str, Dict[str, Any]] = {}

    # --- HYBRID METADATA LOGIC SETUP ---
    pdf_files.sort() 
    files_to_use_fallback = pdf_files[-4:] 
    
    print(f"Starting persistent batch embedding for {len(pdf_files)} PDF(s)...")

    for file_name in tqdm(pdf_files, desc="Embedding Documents"):
        file_path = os.path.join(pdf_folder, file_name)
        
        # 🌟 FIX: The key used for JSON lookup must be normalized to lowercase
        json_key = file_name.lower() 
        
        # --- Metadata Selection Logic ---
        use_fallback = False
        
        if file_name in files_to_use_fallback:
            use_fallback = True
        elif json_key not in case_metadata_map:
             tqdm.write(f"*** WARNING *** File: {file_name}. Key '{json_key}' not found in JSON. Falling back to filename.")
             use_fallback = True
        
        try:
            reader = PdfReader(file_path)
            
            if len(reader.pages) == 0:
                raise ValueError("PDF contains no readable pages.")

            # Text extraction
            headnote_text = reader.pages[0].extract_text() 
            main_judgment_text = "".join(page.extract_text() + "\n\n" for page in reader.pages[1:])

            # 🌟 FIX: Clean the text immediately after extraction (solves mapping corruption)
            headnote_text = clean_extracted_text(headnote_text)
            main_judgment_text = clean_extracted_text(main_judgment_text)
            
            if not headnote_text or len(main_judgment_text) < 50:
                 raise ValueError("Cleaned text extraction resulted in empty/minimal content.")

            doc_id = generate_doc_id(file_path)
            
            # --- A. PARENT DOCUMENT METADATA STORAGE ---
            if use_fallback:
                parent_metadata = extract_metadata_from_filename(file_name, doc_id)
                tqdm.write(f"    -> PARENT: Using UNSTRUCTURED metadata for {file_name}.")
            else:
                parent_metadata = extract_metadata_from_json(json_key, doc_id, case_metadata_map)
                tqdm.write(f"    -> PARENT: Successfully mapped {file_name} to {parent_metadata['CaseTitle']}.")
            
            parent_metadata_store[doc_id] = parent_metadata
            
            # --- B. CHILD DOCUMENT CHUNKING ---
            child_chunks: List[str] = child_splitter.split_text(main_judgment_text)
            
            if not child_chunks:
                 raise ValueError("Chunking resulted in zero chunks.")

            # Convert text chunks into LangChain Document objects
            for i, chunk_text in enumerate(child_chunks):
                child_doc = LC_Document(
                    page_content=chunk_text,
                    metadata={
                        "SourceID": doc_id,          
                        "ChunkIndex": i + 1,        
                        "ParentID": doc_id,          
                        "EntityType": "MainBodyChunk",
                        "CaseTitle": parent_metadata["CaseTitle"] 
                    }
                )
                all_child_documents.append(child_doc)
                
            tqdm.write(f"    -> CHILD: Parsed {len(child_chunks)} paragraph chunks (ID: {doc_id}).")
            
        except (pypdf_errors.PdfReadError, ValueError) as e: 
            tqdm.write(f"*** SKIPPING ERROR *** File: {file_name}. Reason: {e}")
            continue
        except Exception as e:
            tqdm.write(f"*** UNEXPECTED SKIPPING ERROR *** File: {file_name}. Reason: {e}")
            continue

    # 3. --- FINAL PERSISTENCE STEP: SAVING TO DISK ---
    print("\n--- Saving all embeddings and metadata to ChromaDB on disk ---")
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    # a) Save the Child Vectors (The main index)
    try:
        Chroma.from_documents(
            documents=all_child_documents,
            embedding=embedding_model,
            client=chroma_client,
            collection_name=CHROMA_COLLECTION_NAME,
        )
        print(f"SUCCESS: ChromaDB Vector Index saved to folder: {CHROMA_PATH}")
        print(f"Total embedded chunks stored: {len(all_child_documents)}")
        
    except Exception as e:
        print(f"FAILED to save ChromaDB index. Error: {e}")
        return

    # b) Save the Parent Metadata (The rich Headnote data) to a JSON file
    parent_metadata_file = os.path.join(CHROMA_PATH, "parent_metadata.json")
    with open(parent_metadata_file, 'w', encoding='utf-8') as f:
        json.dump(parent_metadata_store, f, indent=4)
    print(f"SUCCESS: Parent Metadata saved to file: {parent_metadata_file}")

    print("\nEmbedding process complete. Data is stored persistently in your folder.")

# --------------------------------------------------------------------------
# EXECUTION AND VERIFICATION BLOCK
# --------------------------------------------------------------------------

def verify_persistence():
    """Checks if the data was successfully saved to disk and can be reloaded."""
    if not os.path.exists(CHROMA_PATH):
        print(f"\n--- VERIFICATION FAILED: ChromaDB folder '{CHROMA_PATH}' not found. ---")
        return

    print(f"\n--- VERIFICATION: RELOADING EMBEDDING MODEL ({EMBEDDING_MODEL_NAME}) ---")
    try:
        embedding_model_reload = HuggingFaceEmbeddings(
             model_name=EMBEDDING_MODEL_NAME, 
             model_kwargs={'device': DEVICE}, 
             encode_kwargs={'normalize_embeddings': True, 'batch_size': 32} 
        )
    except Exception as e:
         print(f"FATAL: Failed to load HuggingFaceEmbeddings wrapper for verification. Error: {e}")
         return

    print("\n--- VERIFICATION: RELOADING PERSISTENT DATA FROM CHROMADB ---")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        
        reloaded_vectorstore = Chroma(
            client=chroma_client,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_model_reload,
        )
        
        count = chroma_client.get_collection(CHROMA_COLLECTION_NAME).count()
        print(f"SUCCESS: Reloaded ChromaDB collection '{CHROMA_COLLECTION_NAME}' with {count} total chunks.")
        
    except Exception as e:
        print(f"FAILED to reload ChromaDB index. Error: {e}")
        return

    # 3. Reload the Parent Metadata
    parent_metadata_file = os.path.join(CHROMA_PATH, "parent_metadata.json")
    try:
        with open(parent_metadata_file, 'r', encoding='utf-8') as f:
            reloaded_metadata = json.load(f)
        print(f"SUCCESS: Reloaded {len(reloaded_metadata)} Parent Documents (Headnotes).")
        
        first_doc_id = list(reloaded_metadata.keys())[0] if reloaded_metadata else "N/A"
        if first_doc_id != "N/A":
             print(f"Verified Case Title Example: {reloaded_metadata[first_doc_id]['CaseTitle']}")
        
    except Exception as e:
        print(f"FAILED to reload Parent Metadata. Error: {e}")
        return

if __name__ == '__main__':
    PDF_FOLDER = "C:/Users/tamar/Desktop/judgments"
    
    if not os.path.isdir(PDF_FOLDER):
        print(f"CRITICAL ERROR: The directory '{PDF_FOLDER}' was not found. Please ensure the path is correct.")
        exit()
    
    if not os.path.exists(METADATA_JSON_PATH):
        print(f"CRITICAL ERROR: The metadata file '{METADATA_JSON_PATH}' was not found. Please update the METADATA_JSON_PATH variable.")
        exit()
        
    print(f"Loading embedding model for pipeline: {EMBEDDING_MODEL_NAME}")
    
    try:
        embedding_model_instance = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, 
            model_kwargs={'device': DEVICE}, 
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32} 
        )
    except Exception as e:
        print(f"FATAL: Failed to load HuggingFaceEmbeddings wrapper. Error: {e}")
        exit()

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f"\nWARNING: The ChromaDB folder '{CHROMA_PATH}' already exists. Running the pipeline will OVERWRITE all existing data.")
        
    embed_legal_documents(PDF_FOLDER, embedding_model_instance, METADATA_JSON_PATH)
    
    verify_persistence()