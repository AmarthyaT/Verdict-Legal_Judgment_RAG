import os
import chromadb
import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import json

# --- CONFIGURATION ---
COLLECTION_NAME = "legal_docs_contextual"
CHROMA_PATH = r"C:/Users/tamar/Desktop/MUNN/chroma_db"
# IMPORTANT: This is the Embedding Model (all-MiniLM-L6-v2)
MODEL_NAME = "BAAI/bge-large-en" 
os.makedirs(CHROMA_PATH, exist_ok=True)
# Fallback message for when no documents are found
FALLBACK_CONTEXT = "🔍 No relevant documents found in the local database for your query."


# --- INITIALIZE MODELS AND CLIENTS (CACHED) ---

@st.cache_resource
def initialize_models():
    """Initializes LLM, Chroma client, and LangChain chain/Prompt."""
    
    # 1. Initialize LLM (Requires Ollama server running)
    try:
        # --- MODEL CONSTRAINT: STICKING TO 'llama3' as per user's constraint ---
        llm = OllamaLLM(model="llama3") 
    except Exception as e:
        st.error(f"❌ Failed to initialize LLM: {e}. Is Ollama running and the model pulled?")
        st.stop() 

    # 2. Initialize Chroma client
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    except Exception as e:
        st.error(f"❌ Failed to initialize Chroma client: {e}")
        st.stop()

    # 3. Load Embedding Model 
    try:
        embedding_model = SentenceTransformer(MODEL_NAME)
    except Exception as e:
        st.error(f"⚠️ Failed to load SentenceTransformer model '{MODEL_NAME}': {e}")
        st.stop()

    # 4. UNIFIED PROMPT TEMPLATE (Handles both RAG and Fallback logic)
    # The prompt still uses Llama model instruction to guide the response.
    template_unified = """
SYSTEM: You are an expert Legal Analyst for the Indian legal system, developed from a Llama 3 model fine-tuned on the Indian Legal Corpus.

---
[RETRIEVED CONTEXT]
{document}
---

USER QUERY:
{question}
    
---
[INSTRUCTIONS AND EXECUTION]

Analyze the status of the [RETRIEVED CONTEXT] and follow only ONE of the two paths below.

### PATH 1: CONTEXT IS FOUND (RAG ANALYSIS)
If the [RETRIEVED CONTEXT] contains relevant excerpts (i.e., does NOT contain the standard fallback message):
1.  **ROLE:** Act as a strict summarizer. Your analysis must be **precise, factual, and based *solely* on the provided context.** Do not include any external legal knowledge.
2.  **FORMAT:** Provide a full-scale summary and analysis in **structured text format** using the following three headings:
    * **1. Case Summary and Core Facts:** A concise, neutral summary of the facts and arguments discussed in the context.
    * **2. Ratio Decidendi (Core Legal Principle):** Identify the key legal principle applied, the court's reasoning, and provide the exact, verbatim quote from the context that supports it.
    * **3. Analysis of Laws and Precedents:** Identify all laws, statutes, articles, or precedents explicitly mentioned, explain their usage based *only* on the text, and provide supporting quotes.

### PATH 2: CONTEXT IS MISSING (KNOWLEDGE-BASED FALLBACK)
If the [RETRIEVED CONTEXT] is empty, irrelevant, or contains the standard fallback message:
1.  **ROLE:** Act as a comprehensive expert on Indian Law. **Use your existing knowledge base** to provide an accurate, detailed summary relevant to the USER QUERY. **You must think on your own.**
2.  **FORMAT:** **CRITICAL: Begin your response with a clear introductory sentence like 'Based on a general analysis of the USER QUERY and relevant Indian law, here is the legal overview:'** Then, provide a detailed analysis in **structured text format** using the following three headings. **DO NOT** mention the missing database context.
    * **1. General Legal Overview and Context (Dog Bites):** Summary of relevant legal principles concerning liability for pet attacks.
    * **2. Core Legal Principle and Reasoning:** Identify key legal principles (e.g., negligence, absolute liability) relevant to the query and explain their judicial basis.
    * **3. Relevant Laws, Statutes, and Precedents:** Identify all relevant Articles, Sections (e.g., IPC 289, local municipality acts), and major judgments, explaining their application.

**Final Guardrail:** **YOU MUST RETURN A TEXT RESPONSE.** Do not return an empty string or JSON. Adhere strictly to the chosen path's output format.
"""
    prompt_unified = ChatPromptTemplate.from_template(template_unified)
    chain = prompt_unified | llm

    # Return the embedding model explicitly for the query step
    return chroma_client, llm, chain, embedding_model


# --- CHROMA COLLECTION SETUP (GET ONLY, NO CREATE/DELETE) ---

@st.cache_resource
def get_or_create_collection(_client, name):
    """
    Ensures the Chroma collection exists and returns it.
    """
    try:
        collection = _client.get_collection(name=name)
        st.sidebar.success(f"✅ Loaded existing collection: '{name}'")
        return collection
    except Exception:
        error_message = f"❌ Collection '{name}' not found. Please run the ingestion script first to create the necessary database."
        st.error(error_message)
        st.stop( )


# --- INITIALIZE RESOURCES ---
chroma_client, llm, chain, embedding_model = initialize_models()
collection = get_or_create_collection(chroma_client, COLLECTION_NAME)


# --- UTILITY FUNCTION ---
def embed_query(query_text, model):
    """Embeds the user's query using sentence-transformers."""
    try:
        embedding = model.encode(query_text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception:
        return None


# --- MAIN RAG EXECUTION FUNCTION (Optimized Backend) ---

def generate_answer(question: str, collection, embedding_model, chain) -> str:
    """
    Executes the RAG pipeline with a unified chain for RAG or Fallback.
    """
    if not question:
        return "Please enter a question to search the documents."
    
    query_embedding = embed_query(question, embedding_model)
    if query_embedding is None:
        return "⚠️ Error: Could not embed the query. Check Streamlit log/sidebar."

    # 1. Vector Search
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3, # --- CHANGE 2: AGGRESSIVELY REDUCED n_results TO 3 FOR MAXIMUM SPEED ---
            include=['documents', 'metadatas'] 
        )
    except Exception as e:
        return f"⚠️ Error: Semantic query failed: {e}"
    
    documents = results.get("documents")
    metadatas = results.get("metadatas")

    # 2. Prepare Context (RAG or Fallback)
    combined_text = FALLBACK_CONTEXT # Default to fallback message
    is_fallback = True

    if documents and documents[0]:
        # --- PATH 1: Documents Found (RAG) ---
        relevant_chunks = documents[0]
        top_k_metadatas = metadatas[0]
        
        # Combine chunks and metadata for the LLM context
        combined_text = "\n\n---\n\n".join(relevant_chunks)

        source_info = "\n\n--- Source Documents Metadata ---\n"
        for i, meta in enumerate(top_k_metadatas):
            source_info += f"Chunk {i+1} Source: {meta.get('source', 'Unknown')}\n"
        
        combined_text = source_info + "\n--- Relevant Excerpts ---\n" + combined_text
        is_fallback = False # Mark as RAG Path
    else:
           # --- PATH 2: No Documents Found (Knowledge-Based Fallback) ---
           st.warning("🔍 No relevant documents found in the database. LLM is using its general knowledge.")


    # 3. Invoke LLM with Context (Unified Chain)
    try:
        # Note: If generation is still slow/empty, the 'llama3' model is likely too large for the host machine's resources.
        response = chain.invoke({
            "document": combined_text,
            "question": question
        })
        
        # 4. Cleanup and return response
        if not response.strip():
             return "⚠️ The analysis completed, but the LLM returned an empty response. This usually indicates an issue with the model or prompt adherence. Please try a different query or LLM model."
        
        return response
        
    except Exception as e:
        return f"⚠️ An error occurred while generating the response: {e}"


# --- STREAMLIT FRONTEND ---

def run_streamlit_app():
    
    def clear_all_resource_cache():
        # Clear cache is crucial when changing models/paths
        st.cache_resource.clear()
        # --- FIX: Changed st.experimental_rerun() to st.rerun() ---
        st.rerun() 

    st.set_page_config(
        page_title="⚖️ Legal RAG Chatbot", 
        layout="wide"
    )

    st.title("⚖️ Legal Judgment Analyst (RAG Chatbot)")
    st.markdown("Ask a question to search the **Supreme Court Judgments** database using Ollama and ChromaDB.")
    
    # Display diagnostics in the sidebar
    with st.sidebar:
        st.header("System Status")
        st.write(f"**Collection:** `{COLLECTION_NAME}`")
        try:
            total_items = collection.count()
            st.write(f"**Total Documents:** `{total_items}`")
        except Exception:
            st.write("**Total Documents:** Error reading count.")
        st.write(f"**LLM Model:** `{llm.model}` (via Ollama)")
        st.write(f"**Embedding Model:** `{MODEL_NAME}`")
        st.markdown("---")
        
        st.subheader("⚠️ Troubleshooting")
        st.info("If you see the '1024 got 384' dimension error, click below to force a cache reset.")
        if st.button("Force Clear Cache & Rerun", type="secondary", on_click=clear_all_resource_cache):
            pass 


# 1. User Input
    question = st.text_area(
        "Enter your legal question:",
        value="", # Blank to receive user input
        height=100
    )

    # 2. Search Button
    if st.button("Search Documents and Get Answer", type="primary"):
        if question:
            st.subheader("🤖 Generated Answer and Summary") 
            
            with st.spinner(f"🔍 Searching and generating answer with {llm.model} (RAG in progress)..."):
                answer =generate_answer(question, collection, embedding_model, chain)
            
            # 3. Display Result
            st.markdown(answer) 
            
            st.success("Analysis Complete!")
        else:
            st.warning("Please enter a question before searching.")


if __name__ == "__main__":
    run_streamlit_app()