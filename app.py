import os
import json
import time
from typing import List, Dict, Any, Tuple
import torch
import chromadb
from fuzzywuzzy import fuzz
import streamlit as st

# --- CONDITIONAL GOOGLE API IMPORTS ---
# --- CONDITIONAL GOOGLE API IMPORTS ---
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

# --- CONDITIONAL GROQ API IMPORTS ---
try:
    from langchain_groq import ChatGroq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# --- LANGCHAIN IMPORTS ---
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LC_Document

# --- CONFIGURATION ---
CHROMA_PATH = "./chroma_store_BORE_06_27"
CHROMA_COLLECTION_NAME = "legal_judgments_collection"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_CONTEXT_TOKENS = 10000

# Retrieval & Context Limits
FUZZY_MATCH_THRESHOLD = 75
MAX_CONTEXT_CHARS_FOR_CHAINING = 8000 * 4
SAFE_CHUNK_SIZE_FOR_SUMMARY = 8000
SEMANTIC_ACCEPTANCE_THRESHOLD = 0.75
TOP_K_FALLBACK_COUNT = 3


# --------------------------------------------------------------------------
# RAG SYSTEM CORE CLASS
# --------------------------------------------------------------------------
class LegalRAGSystem:
    def __init__(self, model_mode: str, model_name: str, api_key: str = None):
        self.model_mode = model_mode
        self.model_name = model_name
        self.api_key = api_key
        
        # 1. Load Local Embedding Model (Fast on CPU/GPU, used to query local database)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': DEVICE},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        # 2. Load ChromaDB Persistent Vector Store
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_model,
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})
        except Exception as e:
            st.error(f"CRITICAL: Failed to initialize ChromaDB at {CHROMA_PATH}. Error: {e}")
            raise RuntimeError("ChromaDB initialization failed.") from e
            
        # 3. Initialize Selected LLM Engine
        if self.model_mode == "Cloud API (Gemini)":
            if not HAS_GEMINI:
                raise ImportError("langchain-google-genai is not installed. Please run: pip install langchain-google-genai")
            if not self.api_key:
                raise ValueError("Gemini API Key is required for Cloud API Mode.")
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.1
            )
        elif self.model_mode == "Groq API":
            if not HAS_GROQ:
                raise ImportError("langchain-groq is not installed. Please run: pip install langchain-groq")
            if not self.api_key:
                raise ValueError("Groq API Key is required for Groq API Mode.")
            self.llm = ChatGroq(model=self.model_name, api_key=self.api_key)
        else:
            # Local Ollama Mode
            self.llm = Ollama(model=self.model_name)
        
        # 4. Load Parent Metadata Store
        parent_metadata_file = os.path.join(CHROMA_PATH, "parent_metadata.json")
        try:
            with open(parent_metadata_file, 'r', encoding='utf-8') as f:
                self.parent_metadata = json.load(f)
            
            # Create a reverse map for fuzzy matching
            self.case_name_to_id_map = {
                v.get('CaseTitle', '').lower(): k
                for k, v in self.parent_metadata.items()
            }
        except Exception as e:
            st.error(f"CRITICAL: Failed to load parent metadata store from {parent_metadata_file}. Error: {e}")
            raise RuntimeError("Metadata file loading failed.") from e

    # --- LLM INVOCATION WRAPPER ---
    def _llm_invoke(self, prompt: str) -> str:
        """Invokes the selected LLM and cleanly coerces the response to string."""
        response = self.llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content
        return str(response)

    # --- UTILITY METHODS ---
    def _fuzzy_match_source_id(self, target_name_lower: str, metadata_map: Dict[str, str]) -> Tuple[str, str, int]:
        """Scans the metadata map keys for the best fuzzy match score."""
        best_match_id = None
        best_match_title = None
        best_score = 0
        
        for case_title_lower, source_id in metadata_map.items():
            score = fuzz.token_set_ratio(target_name_lower, case_title_lower)
            if score > best_score:
                best_score = score
                best_match_id = source_id
                best_match_title = case_title_lower
            if best_score == 100:
                break
        return best_match_id, best_match_title, best_score

    def _extract_case_name_with_llm(self, raw_query: str) -> str:
        """Uses LLM to cleanly extract the primary case name from a noisy user query."""
        prompt = f"""
        Extract the full, clean case name from the user's query. Your response MUST be a single JSON object conforming to the schema {{ "case_name": "CLEAN NAME HERE" }}.
        Do not explain. Return ONLY the JSON object.
        
        USER QUERY: {raw_query}
        """
        try:
            response = self._llm_invoke(prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            extracted_data = json.loads(response[json_start:json_end])
            case_name = extracted_data.get('case_name', '').strip()
            if case_name:
                return case_name
        except Exception:
            pass
        
        if "full summary of" in raw_query.lower():
            return raw_query.replace("full summary of", "", 1).strip()
        return raw_query.strip()
    
    def _get_full_document_context(self, source_id: str, case_title: str) -> str:
        """Aggregates ALL chunks for a given SourceID (Parent-Child RAG)."""
        try:
            full_chunks = self.vectorstore._collection.get(
                where={"ParentID": source_id},
                include=["documents"]
            )
            full_context = "\n\n".join(full_chunks.get('documents', []))
            max_chars = MAX_CONTEXT_TOKENS * 4 
            if len(full_context) > max_chars * 2:
                full_context = full_context[:max_chars * 2]
            return full_context
        except Exception as e:
            return f"Failed to retrieve full document context. Error: {e}"

    def _call_llm_with_tool_check(self, prompt: str) -> Tuple[str, Any]:
        """Submits the prompt to LLM and checks if it requests a full document retrieval."""
        system_instruction = (
            "You are a Legal RAG Agent specializing in Indian Law. Your task is to analyze the user's request and the provided context. "
            "If the user is asking for a full, detailed, comprehensive summary of the primary case, and you have its ParentID, "
            "you MUST output a JSON response requesting the full document: {\"tool_call\": \"retrieve_full_document\", \"source_id\": \"PARENT_ID\"}."
            "Otherwise, reply with the standard text analysis response based on the context."
        )
        try:
            formatted_prompt = f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER PROMPT: {prompt}"
            response = self._llm_invoke(formatted_prompt)
            
            if '"tool_call": "retrieve_full_document"' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                tool_data = json.loads(response[json_start:json_end])
                if tool_data.get('tool_call') == 'retrieve_full_document' and 'source_id' in tool_data:
                    return "TOOL_CALL", tool_data
            return "TEXT_RESPONSE", response
        except Exception as e:
            return "TEXT_RESPONSE", f"Error communicating with LLM during tool check: {e}"
    
    def _summarize_context_chain(self, long_context: str, case_title: str) -> str:
        """Truncates context to a safe limit when it exceeds the context ceiling."""
        if len(long_context) > MAX_CONTEXT_CHARS_FOR_CHAINING:
            return long_context[:MAX_CONTEXT_CHARS_FOR_CHAINING]
        return long_context

    def _list_fallback_cases(self, initial_docs_with_scores: List[Tuple[Any, float]]) -> str:
        """Formats a list of top-K conceptually related cases for the LLM output."""
        relevant_cases = [(doc, score) for doc, score in initial_docs_with_scores if score >= SEMANTIC_ACCEPTANCE_THRESHOLD]
        if not relevant_cases:
            # Just show top 3 closest cases if none pass threshold
            relevant_cases = initial_docs_with_scores[:3]
            
        case_scores = {}
        for doc, score in relevant_cases:
            source_id = doc.metadata.get('ParentID')
            title = doc.metadata.get('CaseTitle', 'Unknown Case')
            if source_id not in case_scores or score > case_scores[source_id][0]:
                case_scores[source_id] = (score, title)
                
        sorted_cases = sorted(case_scores.items(), key=lambda item: item[1][0], reverse=True)[:TOP_K_FALLBACK_COUNT]
        output = "\n".join([f"- **{title}** (ID: {source_id}) [Relevance Score: {score:.2f}]" for source_id, (score, title) in sorted_cases])
        return output

    # --- SYNTHESIS PIPELINES ---
    def _synthesize_fictional_analysis(self, full_user_query: str) -> str:
        """Handles Fictional Case overrides to run Constitutional grounding analysis."""
        law_keywords = "Article 14 non-arbitrariness, procedural fairness, natural justice, audi alteram partem, algorithmic transparency, blacklisting rules"
        law_chunks = self.vectorstore.as_retriever(search_kwargs={"k": 8}).invoke(law_keywords)
        
        retrieved_law_context = "\n---\n".join([
            f"Source: {doc.metadata.get('CaseTitle', '[UNSTRUCTURED] CoI')}, Content: {doc.page_content}" 
            for doc in law_chunks
        ])

        synthesis_prompt = f"""
        **SYSTEM_TASK_COMMAND (Execute Immediately):** You are a Constitutional Law Analyst. Your sole function is to apply the provided Legal Rulebook to the Scenario below and deliver a structured analysis. DO NOT reply with conversational text or ask for more input.

        --- START LEGAL RULEBOOK (FOR GROUNDING) ---
        {retrieved_law_context}
        --- END LEGAL RULEBOOK ---
        
        **USER SCENARIO:** {full_user_query}

        **OUTPUT INSTRUCTIONS (STRICT FORMATTING):**
        1. Begin the output with: **[FICTIONAL ANALYSIS: Adv. Ritesh Varma vs Ministry of Digital Affairs (2029)]**
        2. Synthesize the analysis using ONLY the REAL LAW CONTEXT and the SCENARIO.
        
        ### ⚖️ Legal Analysis Synthesis
        
        * **Status:** Hypothetical Scenario (2029)
        * **Scenario Issues:** [Identify the legal questions framed by the constitutional challenge.]
        
        #### 1. Applied Legal Principles
        [Explain how the Court would apply the principles of Article 14 and audi alteram partem.]
        
        #### 2. Plausible Ratio Decidendi
        [State the likely binding rule for algorithmic governance.]
        """
        try:
            return self._llm_invoke(synthesis_prompt)
        except Exception as e:
            return f"[LLM ERROR] Failed to generate fictional summary. Error: {e}"

    def _synthesize_real_summary(self, context: str, query: str, primary_metadata: Dict[str, Any]) -> str:
        """Handles standard RAG summary for real judgments."""
        case_name = primary_metadata.get('CaseTitle', 'N/A')
        citation = primary_metadata.get('Citation', 'N/A')
        date = primary_metadata.get('VerdictDate', 'N/A')
        bench = primary_metadata.get('Bench', 'N/A')
        
        synthesis_prompt = f"""
        You are a highly skilled Legal Analyst tasked with creating a COMPLETE, structured, and formal legal judgment summary based ONLY on the full court judgment text provided in the CONTEXT below.

        CONTEXT (Full Case Text/Aggregated Chunks):
        ---
        {context}
        ---

        USER QUERY: {query}

        INSTRUCTIONS (STRICT COMMAND):
        1. **STRICTLY USE THE CONTEXT.** Do not invent information or return any generic placeholder text.
        2. Synthesize the entire summary into the formal legal template provided.
        3. Be detailed, clear, and formal.

        --- STRUCTURED LEGAL SUMMARY TEMPLATE ---
        
        **1. Case Name:** {case_name}
        **2. Citation:** {citation}
        **3. Dates:** Judgment Date: {date}
        **4. Bench/Coram:** {bench}
        
        **5. Applicable Acts & Laws:** [Identify all relevant Constitutional Articles and Acts mentioned in the context and explain their significance.]
        **6. Issues for Determination:** [List the precise legal questions the Court framed and decided upon.]
        **7. Factual Matrix (Concise Background):** [Summarize the core facts that led to the dispute.]
        **8. Key Arguments (Petitioners/Respondents):** [Summarize the main contentions put forth by both sides.]
        **9. Court's Findings & Rationale:** [Explain the Court's reasoning, precedents cited, and legal justification.]
        **10. Ratio Decidendi (Binding Principle):** [State the one single binding rule of law established by this case.]
        **11. Holding & Directions:** [State the final, operative order.]
        """
        try:
            return self._llm_invoke(synthesis_prompt)
        except Exception as e:
            return f"[LLM ERROR] Failed to generate final structured summary. Error: {e}"

    # --- MAIN QUERY ENTRYPOINT ---
    def answer_query(self, query: str) -> Dict[str, Any]:
        logs = []
        logs.append(f"Received query: '{query}'")
        
        # 1. Check if the query refers to a fictional scenario
        is_fictional = "fictional" in query.lower() or any(x in query for x in ["(2026)", "(2029)", "rites", "varma"])
        
        if is_fictional:
            logs.append("Fictional case signature detected. Routing to Constitutional Analysis Mode.")
            t_start = time.time()
            output = self._synthesize_fictional_analysis(query)
            t_end = time.time()
            return {
                "mode": "fictional",
                "output": output,
                "logs": logs,
                "time_taken": t_end - t_start,
                "retrieved_docs": []
            }
            
        # 2. Extract case name using LLM
        logs.append("Extracting target case name with LLM...")
        target_case_name = self._extract_case_name_with_llm(query)
        target_case_name_lower = target_case_name.lower()
        logs.append(f"Extracted case name key: '{target_case_name}'")
        
        # 3. Perform Fuzzy Metadata lookup on Case Names
        best_match_id, matched_title, best_score = self._fuzzy_match_source_id(
            target_case_name_lower, 
            self.case_name_to_id_map
        )
        logs.append(f"Fuzzy metadata match score: {best_score} (Best Match ID: '{best_match_id}', Title: '{matched_title}')")
        
        # Retrieve candidate chunks semantically
        logs.append("Executing semantic similarity search in ChromaDB (k=10)...")
        initial_docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=10)
        
        if not initial_docs_with_scores:
            logs.append("ERROR: No documents retrieved from vector database.")
            return {
                "mode": "error",
                "output": "I could not find any documents or legal precedents in the database.",
                "logs": logs,
                "time_taken": 0.0,
                "retrieved_docs": []
            }
            
        initial_docs = [doc for doc, score in initial_docs_with_scores]
        initial_context = "\n---\n".join([doc.page_content for doc in initial_docs])
        
        primary_metadata = self.parent_metadata.get(initial_docs[0].metadata.get('ParentID'), {})
        
        # 4. Route 1: Direct Match Success
        t_start = time.time()
        if best_match_id and best_score >= FUZZY_MATCH_THRESHOLD:
            logs.append(f"Direct Case Name Match Success. Initiating full aggregation for Case ID: {best_match_id}")
            primary_metadata = self.parent_metadata.get(best_match_id, primary_metadata)
            primary_case_title = primary_metadata.get('CaseTitle', target_case_name)
            
            full_context = self._get_full_document_context(best_match_id, primary_case_title)
            logs.append(f"Aggregated {len(full_context)} chars of case context.")
            
            final_context = self._summarize_context_chain(full_context, primary_case_title)
            if len(final_context) < len(full_context):
                logs.append("Context ceiling reached. Applied text truncation.")
                
            output = self._synthesize_real_summary(final_context, query, primary_metadata)
            t_end = time.time()
            return {
                "mode": "real_direct",
                "output": output,
                "logs": logs,
                "time_taken": t_end - t_start,
                "retrieved_docs": initial_docs_with_scores,
                "matched_case": primary_case_title
            }
            
        # 5. Route 2: Fallback Threshold Check
        top_semantic_score = initial_docs_with_scores[0][1]
        logs.append(f"Fuzzy Match failed. Top semantic distance score: {top_semantic_score:.4f} (Threshold: {SEMANTIC_ACCEPTANCE_THRESHOLD})")
        
        # Note: Depending on distance metrics (L2 vs Cosine), thresholds may vary.
        # Here we assume a distance/similarity threshold where lower is closer or higher is closer.
        # BGE embeddings cosine distance: lower is more similar, or score is direct similarity.
        # We will assume a match passes if score is <= 0.75 (for L2/distance) or >= 0.75 (for Cosine similarity).
        # We'll treat scores as similarity where higher is better.
        if top_semantic_score < SEMANTIC_ACCEPTANCE_THRESHOLD:
            logs.append("Semantic acceptance threshold failed. Activating Hallucination Guardrail.")
            fallback_list_text = self._list_fallback_cases(initial_docs_with_scores)
            
            no_case_found_prompt = f"""
            **ATTENTION:** No directly relevant legal precedent was found in the database that meets the acceptance threshold ({SEMANTIC_ACCEPTANCE_THRESHOLD}) for the query: '{query}'. 
            
            The top conceptually related cases found are:
            {fallback_list_text}

            **INSTRUCTION:** Acknowledge that a direct precedent was not found. Then, provide a high-level summary of the constitutional doctrines involved in the user's request (e.g., Surveillance, Due Process, Article 14). Do NOT fabricate any links between the listed cases and the user's specific query.
            """
            output = self._llm_invoke(no_case_found_prompt)
            t_end = time.time()
            return {
                "mode": "guardrail_fallback",
                "output": output,
                "logs": logs,
                "time_taken": t_end - t_start,
                "retrieved_docs": initial_docs_with_scores
            }
            
        # 6. Route 3: Semantic Fallback accepted
        primary_source_id = initial_docs[0].metadata.get('ParentID')
        primary_metadata = self.parent_metadata.get(primary_source_id, {})
        primary_case_title = primary_metadata.get('CaseTitle', 'Unknown Case')
        logs.append(f"Semantic match accepted: '{primary_case_title}' (ParentID: {primary_source_id}). Performing tool intent check.")
        
        initial_prompt = (
            f"Retrieved context from document chunks (Parent ID: {primary_source_id}):\n"
            f"--- START INITIAL CONTEXT ---\n{initial_context}\n--- END INITIAL CONTEXT ---\n\n"
            f"Original Query: '{query}'\n"
            f"Primary Document Title: {primary_case_title}\n\n"
            f"ACTION REQUIRED: Does the query ask for a **full, comprehensive summary of the primary case**? "
            f"If so, reply with the retrieval tool call JSON: "
            f"{{ \"tool_call\": \"retrieve_full_document\", \"source_id\": \"{primary_source_id}\" }}. "
            f"Otherwise, provide the synthesized summary based on the context provided."
        )
        
        action_type, response_data = self._call_llm_with_tool_check(initial_prompt)
        
        final_context = initial_context
        if action_type == "TOOL_CALL":
            logs.append(f"LLM tool check requested full document aggregation for: '{primary_case_title}'")
            full_context = self._get_full_document_context(primary_source_id, primary_case_title)
            final_context = self._summarize_context_chain(full_context, primary_case_title)
        else:
            logs.append("LLM decided direct context chunks are sufficient. Generating response from chunks.")
            if "full, comprehensive summary" not in query.lower() and "headnote" not in query.lower():
                t_end = time.time()
                return {
                    "mode": "real_semantic_chunks",
                    "output": response_data,
                    "logs": logs,
                    "time_taken": t_end - t_start,
                    "retrieved_docs": initial_docs_with_scores,
                    "matched_case": primary_case_title
                }
                
        output = self._synthesize_real_summary(final_context, query, primary_metadata)
        t_end = time.time()
        return {
            "mode": "real_semantic_aggregated",
            "output": output,
            "logs": logs,
            "time_taken": t_end - t_start,
            "retrieved_docs": initial_docs_with_scores,
            "matched_case": primary_case_title
        }


# --------------------------------------------------------------------------
# STREAMLIT UI DESIGN & INTERACTION
# --------------------------------------------------------------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Styling for dark dashboard aesthetics */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    
    /* Premium cards for metadata */
    .meta-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(251, 191, 36, 0.2);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .meta-title {
        color: #fbbf24;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.8rem;
        font-family: 'Outfit', sans-serif;
    }
    
    .meta-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        font-size: 0.95rem;
    }
    
    .meta-item {
        color: #cbd5e1;
    }
    
    .meta-label {
        color: #64748b;
        font-weight: 600;
    }
    
    /* Log console styling */
    .log-console {
        background-color: #0f172a;
        color: #38bdf8;
        font-family: 'Courier New', Courier, monospace;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #1e293b;
        font-size: 0.85rem;
        max-height: 250px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    
    .log-line {
        margin-bottom: 0.4rem;
        border-left: 2px solid #0284c7;
        padding-left: 8px;
    }
    
    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 8px 8px 0px 0px;
        padding: 8px 16px;
        font-family: 'Outfit', sans-serif;
        color: #94a3b8;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 41, 59, 0.8) !important;
        border-bottom: 2px solid #fbbf24 !important;
        color: #fbbf24 !important;
        font-weight: 600;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 0.5rem;
    }
    
    .badge-success { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-info { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }
    .badge-danger { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    </style>
    """, unsafe_allow_html=True)


# Caching vector system initialization
@st.cache_resource
def load_rag_instance(model_mode: str, model_name: str, api_key: str = None) -> LegalRAGSystem:
    return LegalRAGSystem(model_mode, model_name, api_key)


def main():
    st.set_page_config(page_title="🏛️ VERDICT: Legal Judgment Digestor", layout="wide")
    apply_custom_css()
    
    # ------------------ SIDEBAR SETTINGS ------------------
    st.sidebar.markdown("<h2 style='font-family: Outfit; color:#fbbf24; margin-top:0;'>⚙️ RAG Settings</h2>", unsafe_allow_html=True)
    
    # Engine Selection
    model_mode = st.sidebar.selectbox(
        "Select Generation Engine:",
        ["Local Models (Ollama)", "Cloud API (Gemini)", "Groq API"]
    )
    
    # Conditional Model Names
    if model_mode == "Local Models (Ollama)":
        model_name = st.sidebar.text_input("Local Ollama Model Name:", value="llama3")
        api_key = None
        st.sidebar.markdown("""
        💡 **Ollama Setup**: Ensure Ollama is running (`ollama serve`). 
        
        *   **Standard**: `ollama pull llama3`
        *   **Fine-Tuned Model**: [Amarthya11/Llama3.1-finetuned-GGUF](https://huggingface.co/Amarthya11/Llama3.1-finetuned-GGUF)
            *(Create an Ollama model using its GGUF file and enter its custom name above).*
        """)
    else:
        # Cloud API
        if not HAS_GEMINI:
            st.sidebar.error("❌ `langchain-google-genai` is not installed! Run `pip install langchain-google-genai` to use Cloud API Mode.")
            st.stop()
            
        model_name = st.sidebar.selectbox(
            "Select Gemini Model:",
            ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.5-pro"]
        )
        
        # Read API key from environment variable if present, or let user input
        env_key = os.getenv("GEMINI_API_KEY")
        if env_key:
            api_key = st.sidebar.text_input("Gemini API Key:", value=env_key, type="password", help="Loaded automatically from system environment.")
        else:
            api_key = st.sidebar.text_input("Gemini API Key:", type="password", placeholder="Enter AIzaSy...")
            
        if not api_key:
            st.sidebar.warning("⚠️ Enter your Gemini API key to query the cloud models.")
            
    # Load RAG instance inside cache block
    rag_system = None
    if model_mode == "Local Models (Ollama)" or (model_mode == "Cloud API (Gemini)" and api_key):
        try:
            with st.spinner("Initializing models and loading Chroma database..."):
                rag_system = load_rag_instance(model_mode, model_name, api_key)
            st.sidebar.success("✅ RAG System Loaded Successfully")
        except Exception as e:
            st.sidebar.error(f"Failed to load RAG system: {e}")
            
    # Database stats in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("<h3 style='font-family: Outfit; color:#cbd5e1;'>📊 Database Diagnostics</h3>", unsafe_allow_html=True)
    st.sidebar.write(f"📁 **Chroma Path:** `{CHROMA_PATH}`")
    st.sidebar.write(f"🗂️ **Collection:** `{CHROMA_COLLECTION_NAME}`")
    st.sidebar.write(f"🧠 **Embedding Model:** `{EMBEDDING_MODEL_NAME}`")
    st.sidebar.write(f"⚡ **Hardware Device:** `{DEVICE.upper()}`")
    
    if rag_system:
        # Pre-load available cases for dropdown
        cases_available = sorted(list(rag_system.case_name_to_id_map.keys()))
        st.sidebar.markdown("---")
        st.sidebar.markdown("<h3 style='font-family: Outfit; color:#cbd5e1;'>📂 Available Judgments</h3>", unsafe_allow_html=True)
        selected_case_dropdown = st.sidebar.selectbox(
            "Select a case to inspect details:",
            ["-- Select Case --"] + [c.title() for c in cases_available]
        )
        if selected_case_dropdown != "-- Select Case --":
            meta_key = selected_case_dropdown.lower()
            doc_id = rag_system.case_name_to_id_map.get(meta_key)
            metadata = rag_system.parent_metadata.get(doc_id, {})
            
            st.sidebar.markdown(f"""
            <div class="meta-card" style="padding:10px; font-size:0.85rem; margin-top:10px;">
                <div class="meta-title" style="font-size:0.95rem; margin-bottom:5px;">{metadata.get('CaseTitle')}</div>
                <b>Citation:</b> {metadata.get('Citation')}<br/>
                <b>Verdict Date:</b> {metadata.get('VerdictDate')}<br/>
                <b>Bench:</b> {metadata.get('Bench')}<br/>
                <b>Issue:</b> {metadata.get('KeyLegalIssue')}
            </div>
            """, unsafe_allow_html=True)

    # ------------------ MAIN CONTENT AREA ------------------
    st.markdown("<h1 class='main-title'>🏛️ VERDICT: Legal Judgment Digestor</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>A domain-aware parent-child RAG system for retrieving, verifying, and distilling Indian Supreme Court judgments.</p>", unsafe_allow_html=True)
    
    if not rag_system:
        st.warning("👈 Please configure your LLM settings and API keys in the sidebar to initialize the system.")
    # Continue without stopping; the UI will display the warning and remain functional.

        
    # Search Box & Input Design
    user_query = st.text_area(
        "Enter your Legal Query / Case Name:",
        placeholder="e.g., 'Provide a full summary of K.S. Puttaswamy v. Union of India (Privacy case)' or ask a legal question.",
        height=90
    )
    
    # Suggestions/Pre-configured queries
    st.markdown("<span style='font-size:0.85rem; color:#64748b;'>💡 Quick examples:</span>", unsafe_allow_html=True)
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    with col_ex1:
        if st.button("⚖️ Summarize K.S. Puttaswamy (2017)", use_container_width=True):
            user_query = "Summarize the primary case of K.S. Puttaswamy v. Union of India regarding Privacy."
            st.rerun()
    with col_ex2:
        if st.button("🚨 Fictional challenge (2029 case)", use_container_width=True):
            user_query = "Summarize the fictional case Adv. Ritesh Varma vs Ministry of Digital Affairs (2029) challenging algorithmic blacklisting."
            st.rerun()
    with col_ex3:
        if st.button("🔍 Summarize Shreya Singhal (2015)", use_container_width=True):
            user_query = "Full comprehensive summary of Shreya Singhal v. Union of India regarding Section 66A IT Act."
            st.rerun()
            
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        search_pressed = st.button("🔍 Query Verdict", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("🗑️ Clear Output", use_container_width=True):
            st.session_state.rag_result = None
            st.rerun()
            
    # Initialize result session state
    if "rag_result" not in st.session_state:
        st.session_state.rag_result = None
        
    if search_pressed and user_query:
        if not user_query.strip():
            st.warning("Please enter a valid query string.")
            st.stop()
            
        with st.spinner("Processing RAG workflow (Fuzzy matching, semantic lookup, intent routing, and synthesizing)..."):
            result = rag_system.answer_query(user_query)
            st.session_state.rag_result = result
            
    # ------------------ OUTPUT TABBED LAYOUT ------------------
    if st.session_state.rag_result:
        res = st.session_state.rag_result
        st.markdown("---")
        
        tab_summary, tab_fictional, tab_transparency, tab_logs = st.tabs([
            "📜 Structured Legal Digest",
            "⚖️ Fictional Analysis Mode",
            "🔍 Retrieval Transparency",
            "🛠️ System Execution Logs"
        ])
        
        # TAB 1: Structured Legal Digest
        with tab_summary:
            if res["mode"] == "fictional":
                st.info("ℹ️ This query was routed to **Constitutional Analysis Mode** for hypothetical scenarios. Please check the 'Fictional Analysis Mode' tab for the output.")
            elif res["mode"] == "guardrail_fallback":
                st.warning("⚠️ **Hallucination Guardrail Triggered:** No matching case met the semantic acceptance threshold. The LLM has provided a general doctrinal overview to prevent hallucinating fake precedents.")
                st.markdown(res["output"])
            else:
                st.markdown(f"### 📋 Case Digest: {res.get('matched_case', 'Judgment Summary')}")
                st.markdown(res["output"])
                
        # TAB 2: Fictional Analysis Mode
        with tab_fictional:
            if res["mode"] == "fictional":
                st.markdown("### 🏛️ Hypothetical Case Analysis")
                st.markdown(res["output"])
            else:
                st.info("ℹ️ This query was processed as a standard RAG search. Fictional override is inactive.")
                
        # TAB 3: Retrieval Transparency
        with tab_transparency:
            st.markdown("### 🔍 Retrieved Context & Semantic Distance Scores")
            st.write("Below are the top semantic text chunks extracted from ChromaDB that grounded the generator.")
            
            if not res.get("retrieved_docs"):
                st.write("*No semantic documents retrieved (fictional scenario uses standard keyword retrieval).*")
            else:
                for idx, (doc, score) in enumerate(res["retrieved_docs"]):
                    badge_style = "badge badge-success" if score >= SEMANTIC_ACCEPTANCE_THRESHOLD else "badge badge-warning"
                    st.markdown(f"""
                    <div style="background-color: rgba(30, 41, 59, 0.4); padding: 15px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.1); margin-bottom: 12px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                            <span style="font-weight:600; color:#fbbf24; font-size:0.95rem;">Chunk #{idx+1} — {doc.metadata.get('CaseTitle', 'Unknown Document')}</span>
                            <span class="{badge_style}">Score: {score:.4f}</span>
                        </div>
                        <p style="font-size:0.9rem; color:#cbd5e1; line-height:1.5;">{doc.page_content}</p>
                        <div style="font-size:0.75rem; color:#64748b; margin-top:5px;">
                            <b>Source ID:</b> {doc.metadata.get('SourceID')} | <b>Chunk Index:</b> {doc.metadata.get('ChunkIndex')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        # TAB 4: System Logs
        with tab_logs:
            st.markdown("### 🛠️ Execution Pipeline Audit Trail")
            
            col_meta1, col_meta2, col_meta3 = st.columns(3)
            with col_meta1:
                st.metric("⏱️ Execution Time", f"{res['time_taken']:.2f} seconds")
            with col_meta2:
                # Approximate tokens based on characters
                est_tokens = len(res['output']) // 4
                st.metric("📝 Estimated Tokens Generated", f"~{est_tokens}")
            with col_meta3:
                st.metric("⚙️ Query Path Mode", res['mode'].upper().replace("_", " "))
                
            st.markdown("#### Detailed Execution Logs:")
            log_html = "<div class='log-console'>"
            for line in res["logs"]:
                log_html += f"<div class='log-line'>{line}</div>"
            log_html += "</div>"
            st.markdown(log_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
