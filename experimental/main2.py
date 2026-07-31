import os
import json
import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchainretrievers import MergerRetriever              
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer, CrossEncoder
import ollama

# --------------------------------------------------------------------------
# REQUIREMENTS
# --------------------------------------------------------------------------
# This script is a self-contained Proof-of-Concept (POC)
# To run this, you will need the following packages:
#
# pip install langchain langchain-community langchain-chroma chromadb
# pip install sentence-transformers rank_bm25 ollama
#
# Also ensure Ollama is installed and running:
# ollama pull llama3.1
#
# --------------------------------------------------------------------------
# MODEL CONFIGURATION (as per the research report)
# --------------------------------------------------------------------------

# Section 3.1 Recommendation: BGE model for embeddings
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Section 3.4 Recommendation: Lightweight Cross-encoder for reranking
RERANKER_MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L6-v2'

# Ollama model configuration
OLLAMA_MODEL = "hf.co/Amarthya11/Llama3.1-finetuned-GGUF:Q8_0"  # Change this to your preferred model

CHROMA_DB_PATH = "C:/Users/tamar/Desktop/judg/chroma_db1"

# --------------------------------------------------------------------------
# PROMPT TEMPLATE (as per the research report)
# --------------------------------------------------------------------------

# This is the prompt from Section 5.2, ready to be used.
VIOLATION_PROMPT_TEMPLATE = """
SYSTEM: You are an expert legal analyst for the Indian legal system, developed from a Llama 3.1 model fine-tuned on the Indian Legal Corpus. Your analysis must be precise, factual, and based *solely* on the provided context. Do not include any information or legal knowledge not present in the context.

USER:
Based on the following retrieved context, analyze the user's query.

USER QUERY:
\"\"\"
{user_query}
\"\"\"

RETRIEVED CONTEXT:
---
{retrieved_context}
---

INSTRUCTIONS:
Complete the following two tasks and provide your output *only* in the requested JSON format.

TASK 1: Provide a brief, neutral summary of the core facts and legal principles discussed in the provided context.
TASK 2: Identify all specific laws, statutes, articles, or legal precedents that were violated, discussed, or established. For each, provide the specific text from the context that serves as evidence.

JSON_OUTPUT_FORMAT:
{{
  "case_summary": "A concise summary of the case facts, arguments, and ruling as described in the context.",
  "ratio_decidendi": {{
    "legal_principle": "The core legal principle established or applied by the court",
    "reasoning": "The court's reasoning that led to this principle",
    "supporting_evidence_quote": "A direct, verbatim quote from the retrieved context that contains the ratio decidendi"
  }},
  "analysis_of_violations": [
    {{
      "law_or_precedent": "",
      "violation_or_principle": "[e.g., Establishment of the 'basic structure' doctrine]",
      "description": "[Explain how this law was used or what principle was established from it, based *only* on the text]",
      "supporting_evidence_quote": "[A direct, verbatim quote from the retrieved context that supports this finding]"
    }},
    {{
      "law_or_precedent": "",
      "violation_or_principle": "[e.g., Culpable Homicide]",
      "description": "[Explain the context of this law's discussion]",
      "supporting_evidence_quote": "[A direct, verbatim quote...]"
    }}
  ]
}}
Begin.
"""

# --------------------------------------------------------------------------
# STAGE 1: LOAD MODELS
# --------------------------------------------------------------------------
# We load these once at the start.
try:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    # This is the "Bi-Encoder" for retrieval
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print(f"Loading reranker model: {RERANKER_MODEL_NAME}...")
    # This is the "Cross-Encoder" for reranking [1]
    reranker = CrossEncoder(RERANKER_MODEL_NAME)
    
    print("Models loaded successfully.")
    
    # Check if Ollama is available
    print(f"Checking Ollama connection with model: {OLLAMA_MODEL}...")
    ollama.list()
    print("Ollama is ready.")
    
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure 'sentence-transformers' and 'ollama' are installed.")
    print("Also ensure Ollama is running: ollama serve")
    exit()

# --------------------------------------------------------------------------
# STAGE 2: LOAD EXISTING CHROMADB AND BUILD BM25 INDEX
# --------------------------------------------------------------------------
def load_existing_database(chroma_client, collection_name):
    """
    Loads the existing ChromaDB collection and builds a BM25 index
    from the existing documents.
    """
    print("\n--- STAGE 2: LOADING EXISTING CHROMADB DATABASE ---")
    print(f"Loading from: {CHROMA_DB_PATH}")
    
    # Get the existing collection
    try:
        collection = chroma_client.get_collection(name=collection_name)
        print(f"Successfully loaded collection: {collection_name}")
        
        # Get all documents from the collection for BM25
        results = collection.get(include=['documents', 'metadatas'])
        
        doc_count = len(results['documents'])
        print(f"Found {doc_count} documents in the collection.")
        
        # Build BM25 document list
        bm25_docs = []
        for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
            bm25_docs.append(
                Document(page_content=doc_text, metadata=metadata or {})
            )
        
        print(f"Built BM25 index with {len(bm25_docs)} documents.")
        return collection, bm25_docs
        
    except Exception as e:
        print(f"Error loading collection: {e}")
        print(f"Available collections: {[col.name for col in chroma_client.list_collections()]}")
        raise

# --------------------------------------------------------------------------
# STAGE 3: SETUP HYBRID RETRIEVER (Section 3.3 of report)
# --------------------------------------------------------------------------
def setup_retrievers(chroma_collection, bm25_docs, chroma_client_obj, collection_name):
    """
    Initializes the dense, sparse, and ensemble retrievers.
    """
    print("\n--- STAGE 3: SETTING UP HYBRID RETRIEVERS ---")
    
    # 1. Dense Retriever (ChromaDB)
    # We need a LangChain-compatible embedding function wrapper
    class MyEmbeddingFunctionWrapper(chromadb.EmbeddingFunction):
        def __call__(self, input):
            return embedding_model.encode(input).tolist()

    # Initialize the LangChain Chroma retriever
    vector_retriever = Chroma(
        collection_name=collection_name,
        client=chroma_client_obj,
        embedding_function=MyEmbeddingFunctionWrapper()
    ).as_retriever(search_kwargs={'k': 5}) # Get top 5 semantic results

    print("Chroma vector retriever initialized.")

    # 2. Sparse Retriever (BM25) [7, 8]
    bm25_retriever = BM25Retriever.from_documents(bm25_docs)
    bm25_retriever.k = 5 # Get top 5 keyword results
    print("BM25 sparse retriever initialized.")

    # 3. Ensemble Retriever (RRF) [9, 5, 7]
    # This fuses the results from both retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5] # 50/50 split, tune as needed
    )
    print("EnsembleRetriever (Hybrid Search) initialized.")
    return ensemble_retriever

# --------------------------------------------------------------------------
# STAGE 4 & 5: RETRIEVE-THEN-RERANK (Section 3.4 of report)
# --------------------------------------------------------------------------
def retrieve_and_rerank(query, hybrid_retriever, reranker_model):
    """
    Executes the full retrieve-then-rerank pipeline.
    """
    print(f"\n--- STAGE 4: HYBRID RETRIEVAL (The 'Broad Net') ---")
    print(f"Query: {query}")
    
    # Stage 3: Hybrid Retrieval [5, 7]
    # This invokes both BM25 and Chroma and fuses the results with RRF
    hybrid_results = hybrid_retriever.invoke(query)
    print(f"Found {len(hybrid_results)} initial candidates via Hybrid Search (RRF).")
    
    # De-duplicate results (RRF can return the same doc from both retrievers)
    unique_chunks = {doc.page_content: doc for doc in hybrid_results}.values()
    print(f"De-duplicated to {len(list(unique_chunks))} unique chunks.")
    
    if not unique_chunks:
        return [], "No relevant context found."

    print(f"\n--- STAGE 5: RE-RANKING (The 'Fine-Toothed Comb') ---")
    
    # Stage 4: Reranking with Cross-Encoder [10, 11, 1]
    # Create (query, chunk_text) pairs for the reranker
    pairs = [[query, doc.page_content] for doc in unique_chunks]
    
    # Predict relevance scores
    scores = reranker_model.predict(pairs)
    
    # Combine scores with Document objects and sort
    scored_docs = list(zip(scores, unique_chunks))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Get the "golden" top 3 chunks
    final_top_k = 3
    final_docs = [doc for score, doc in scored_docs[:final_top_k]]
    
    print(f"Reranked results. Top {final_top_k} chunks:")
    for i, (score, doc) in enumerate(scored_docs[:final_top_k]):
        print(f"  Rank {i+1} (Score: {score:.4f}): {doc.page_content.splitlines()[0]}...")

    # Build the final context string for the LLM
    final_context_str = "\n\n---\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" 
        for doc in final_docs
    )
    
    return final_docs, final_context_str

# --------------------------------------------------------------------------
# STAGE 6: ANALYTIC GENERATION (Section 5.2 of report)
# --------------------------------------------------------------------------
def generate_analysis(query, context_str):
    """
    Calls Ollama's Llama 3.1 model for analysis.
    It formats the prompt and gets the response.
    """
    print(f"\n--- STAGE 6: ANALYTIC GENERATION (The 'Synthesizer') ---")
    
    # Format the prompt from Section 5.2
    final_prompt = VIOLATION_PROMPT_TEMPLATE.format(
        user_query=query, 
        retrieved_context=context_str
    )
    
    print("Sending prompt to Ollama (Llama 3.1)...")
    
    try:
        # Call Ollama with the prompt
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    'role': 'user',
                    'content': final_prompt
                }
            ]
        )
        
        # Extract the response text
        response_text = response['message']['content']
        
        print("\n--- RAW RESPONSE FROM OLLAMA ---")
        print(response_text)
        
        # Try to parse as JSON
        try:
            # Find JSON in the response (it might have extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                response_json = json.loads(json_str)
                
                print("\n--- PARSED JSON RESPONSE ---")
                print(json.dumps(response_json, indent=2))
                return response_json
            else:
                print("\nWarning: No JSON found in response. Returning raw text.")
                return {"raw_response": response_text}
                
        except json.JSONDecodeError as e:
            print(f"\nWarning: Could not parse JSON: {e}")
            print("Returning raw response.")
            return {"raw_response": response_text}
            
    except Exception as e:
        print(f"\nError calling Ollama: {e}")
        print("Please ensure Ollama is running: ollama serve")
        return {"error": str(e)}

# --------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# --------------------------------------------------------------------------
def main():
    # --- 1. Setup ChromaDB (Using existing persistent database) ---
    print(f"Connecting to existing ChromaDB at: {CHROMA_DB_PATH}")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # List available collections
    collections = chroma_client.list_collections()
    print(f"Available collections: {[col.name for col in collections]}")
    
    # You may need to change this to match your collection name
    collection_name = "legal_contextual_v1"  # UPDATE THIS to your actual collection name
    
    if not collections:
        print("ERROR: No collections found in the database!")
        print("Please verify the CHROMA_DB_PATH is correct.")
        return
    
    # If you're not sure of the collection name, use the first one
    if not any(col.name == collection_name for col in collections):
        print(f"Collection '{collection_name}' not found.")
        print(f"Using first available collection: {collections[0].name}")
        collection_name = collections[0].name
    
    # --- 2. Load Existing Database (replaces ingestion stage) ---
    collection, bm25_docs = load_existing_database(chroma_client, collection_name)
    
    # --- 3. Setup Retrievers (Stage 3) ---
    hybrid_retriever = setup_retrievers(collection, bm25_docs, chroma_client, collection_name)
    
    # --- 4. Run Query 1 (Specific Query) ---
    # This query is good because it requires BOTH keyword (BM25) and semantic (BGE)
    # "Section 438" -> BM25
    # "principles for bail" -> BGE
    print("\n" + "="*50)
    print("--- RUNNING FIRST EXAMPLE (SPECIFIC QUERY) ---")
    print("="*50)
    
    user_query_1 = "What are the principles for 'anticipatory bail' under Section 438?"
    
    final_docs_1, final_context_1 = retrieve_and_rerank(user_query_1, hybrid_retriever, reranker)
    
    if final_docs_1:
        final_answer_1 = generate_analysis(user_query_1, final_context_1)
    else:
        print("Sorry, no relevant information was found for your query.")
        
    
    # --- 5. Run Query 2 (Temporal Query) ---
    # This query demonstrates the temporal/evolution aspect, which the
    # "Contextual Retrieval" ingestion strategy helps solve.
    print("\n" + "="*50)
    print("--- RUNNING SECOND EXAMPLE (TEMPORAL QUERY) ---")
    print("="*50)
    
    user_query_2 = "How did the 2005 amendment affect the Sibbia case?"
    
    final_docs_2, final_context_2 = retrieve_and_rerank(user_query_2, hybrid_retriever, reranker)
    
    if final_docs_2:
        # We can reuse the same prompt, as it's designed for analysis
        final_answer_2 = generate_analysis(user_query_2, final_context_2)
    else:
        print("Sorry, no relevant information was found for your query.")
            
    print("\n--- SCRIPT COMPLETE ---")

if __name__ == "__main__":
    main()