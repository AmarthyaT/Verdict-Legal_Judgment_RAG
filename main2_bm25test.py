import os
import sys
import chromadb
import numpy as np
import statistics # Not used, but keeping for reference
from typing import List, Tuple
from rank_bm25 import BM25Okapi
# NLTK for tokenization/stopword removal in BM25
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer

# --- NLTK SETUP ---
# --- NLTK SETUP ---
try:
    import nltk
    # NLTK is used for tokenization and stopword removal in the BM25Retriever
    print("ℹ️ Downloading necessary NLTK data...")
    nltk.download('punkt', quiet=True) 
    nltk.download('stopwords', quiet=True)
    # 🛑 CRITICAL FIX: Explicitly download 'punkt_tab' as requested by the traceback.
    nltk.download('punkt_tab', quiet=True) 
    print("✅ NLTK data check complete.")
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")

# --- BM25 Retriever Class ---
class BM25Retriever:
    """A wrapper for BM25Okapi for document retrieval."""
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text: str) -> List[str]:
        """Tokenize and remove stopwords from text."""
        tokens = word_tokenize(text.lower())
        return [token for token in tokens if token not in self.stop_words]
        
    def fit(self, documents: List[str]):
        """Fit BM25 model on a list of documents."""
        self.documents = documents
        tokenized_docs = [self.preprocess_text(doc) for doc in documents]
        # Only fit if documents are not empty
        if tokenized_docs:
            self.bm25 = BM25Okapi(tokenized_docs)
        else:
            print("⚠️ Cannot fit BM25: document list is empty.")
        
    def get_top_k(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Get top k documents most relevant to the query with their scores."""
        if not self.bm25:
            return [] # Return empty list if model not fitted
            
        tokenized_query = self.preprocess_text(query)
        doc_scores = self.bm25.get_scores(tokenized_query)
        # Ensure k doesn't exceed the number of documents
        k = min(k, len(self.documents))
        
        # Argsort returns indices that would sort the array
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        
        # Return index and score
        return [(idx, doc_scores[idx]) for idx in top_k_indices]

# --- CONFIGURATION ---
COLLECTION_NAME = "supreme_court_judgments"
# Assuming CHROMA_PATH is the directory where the script is run
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_store1")
MODEL_NAME = "all-MiniLM-L6-v2"  # A good default model for semantic search

# ensure the directory for the persistent store exists
os.makedirs(CHROMA_PATH, exist_ok=True)

# Initialize LLM with correct model name
# 🛑 NOTE: You must have 'llama3' pulled locally using 'ollama pull llama3'
OLLAMA_MODEL_NAME = "hf.co/Amarthya11/Llama3.1-finetuned-GGUF:Q8_0" # Replace with the model name you have on Ollama (e.g., mistral, llama2, llama3)
try:
    # Changed from OllamaLLM to Ollama for LangChain-Community consistency
    llm = Ollama(model=OLLAMA_MODEL_NAME) 
    print(f"✅ Initialized LLM: {OLLAMA_MODEL_NAME}")
except Exception as e:
    print(f"❌ Failed to initialize LLM. Is Ollama running and model '{OLLAMA_MODEL_NAME}' available? Error: {e}")
    sys.exit(1) # Use sys.exit(1) for cleaner exit on critical failure

# --- INITIALIZE MODELS AND DATABASE ---
# Initialize Chroma client
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    print(f"✅ Initialized Chroma client at: {CHROMA_PATH}")
except Exception as e:
    print(f"❌ Failed to initialize Chroma client: {e}")
    sys.exit(1)

# --- PROMPT TEMPLATE ---
template = """
You are an expert legal analyst. You will be provided with excerpts from a legal judgment retrieved based on a user's question.
Your task is to synthesize the information from these excerpts to answer the user's question and provide a structured summary based on the available text.

Give a full-scale summary of the document based on the excerpts.
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
        print(f"⚠️ Query embedding failed. Check internet connection or model availability. Error: {e}")
        return None

# --- DIAGNOSTICS (omitting for brevity, but they are useful helpers) ---
# Keeping list_collections_verbose() for startup check
def list_collections_verbose():
    # ... (Keep the original list_collections_verbose function here) ...
    # This function is not included in the final response for brevity, but should remain in the script
    pass

# Ensure collection exists (create if missing)
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    total_items = collection.count()
    print(f"✅ Collection '{COLLECTION_NAME}' loaded. Contains {total_items} items.")
except Exception:
    print(f"⚠️ Collection '{COLLECTION_NAME}' not found.")
    try:
        # Must pass a valid embedding function here if not relying on default
        collection = chroma_client.create_collection(name=COLLECTION_NAME)
        print(f"ℹ️ Created new collection: {COLLECTION_NAME}. It is currently empty.")
        total_items = 0
    except Exception as e:
        print(f"❌ Failed to create collection '{COLLECTION_NAME}': {e}")
        sys.exit(1)

# --- MAIN EXECUTION ---
def main():
    """Main loop to accept user questions and generate answers."""
    # 1. Initialize BM25 retriever and fit it
    bm25_retriever = BM25Retriever()
    
    # Get all documents from collection for BM25
    # Only retrieve if the collection has items, or this might fail/be slow on huge collections
    global total_items
    if total_items > 0:
        all_docs = collection.get(include=['documents'])
        # documents[0] is the list of document contents
        if all_docs and all_docs.get('documents'):
            print(f"ℹ️ Fitting BM25 on {len(all_docs['documents'])} documents...")
            bm25_retriever.fit(all_docs['documents'])
            print("✅ BM25 model fitted.")
        else:
            print("⚠️ Collection is not empty, but documents could not be retrieved for BM25. Check embedding process.")
    else:
        print("⚠️ BM25 fitting skipped: collection is empty.")
        all_docs = {'documents': []} # Initialize empty structure
    
    while True:
        question = input("\nAsk a question about your documents (or enter 'q' to quit): ")
        if question.lower() == 'q':
            print("Exiting...")
            break

        if total_items == 0:
            print("🛑 The collection is empty. Please load documents before asking questions.")
            continue

        print("\nEmbedding your query (Semantic Search prep)...")
        query_embedding = embed_query(question)
        if query_embedding is None:
            print("⚠️ Could not embed query. Please try again.")
            continue

        print("🔍 Searching for relevant document chunks...")
        
        # 2. Get Semantic Search Results (Vector Search)
        try:
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=15 
            )
            semantic_chunks = semantic_results.get("documents", [[]])[0]
            
            # 3. Get BM25 Search Results (Lexical Search)
            bm25_results_tuple = bm25_retriever.get_top_k(question, k=15)
            # Map BM25 indices to actual document content
            bm25_chunks = []
            if all_docs['documents']:
                 # Retrieve the actual document content using the index
                 bm25_chunks = [all_docs['documents'][idx] for idx, _ in bm25_results_tuple]
            
            print(f"✅ Semantic results found: {len(semantic_chunks)} chunks.")
            print(f"✅ BM25 results found: {len(bm25_chunks)} chunks.")
            
        except Exception as e:
            print(f"⚠️ Search failed: {e}")
            continue

        # 4. Combine Semantic Search and BM25 results (RRF or simple max-score fusion)
        
        # Use a simple set union to get all unique chunks from both methods
        unique_chunks = list(set(semantic_chunks + bm25_chunks))
        
        if not unique_chunks:
            print("⚠️ No relevant chunks found by either Semantic or BM25 search. Try a different question.")
            continue

        # Create a dictionary to hold the best score for each chunk
        combined_scores = {}
        
        # 4a. Semantic Scoring (assuming Chroma distance is used as a proxy for relevance, higher is better)
        # Chroma returns distance, which is inverse to similarity. We use linspace to simulate a relevance score (1.0 to 0.0)
        semantic_map = {chunk: score for chunk, score in zip(semantic_chunks, np.linspace(1.0, 0.0, len(semantic_chunks)))}

        # 4b. BM25 Scoring (normalize scores)
        max_bm25_score = max(score for _, score in bm25_results_tuple) if bm25_results_tuple else 1.0
        bm25_map = {
             all_docs['documents'][idx]: score / max_bm25_score 
             for idx, score in bm25_results_tuple
        }
        
        # 4c. Combine scores (use max score from both methods for simplicity)
        for chunk in unique_chunks:
            # Simple fusion: take the max normalized score from either method
            sem_score = semantic_map.get(chunk, 0.0)
            bm25_score = bm25_map.get(chunk, 0.0)
            combined_scores[chunk] = max(sem_score, bm25_score)

        # 4d. Sort chunks by combined score
        sorted_chunks_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        # Get the top 8 chunks for the final prompt (limit context length)
        final_relevant_chunks = [chunk for chunk, _ in sorted_chunks_items[:8]] 
        combined_text = "\n\n---\n\n".join(final_relevant_chunks)

        print(f"✅ Found {len(final_relevant_chunks)} relevant chunks with hybrid ranking. Generating response...")
        print(f"Top 3 combined scores: {[f'{score:.3f}' for _, score in sorted_chunks_items[:3]]}")

        # 5. Invoke LLM Chain
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
    if len(sys.argv) > 1 and sys.argv[1] in ("--list-collections", "--list"):
        list_collections_verbose() # Assuming list_collections_verbose is defined
        sys.exit(0)
    main()