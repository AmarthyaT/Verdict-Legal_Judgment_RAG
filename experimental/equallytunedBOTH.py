import os
import json
from typing import List, Dict, Any, Tuple
import torch 
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# --- CRITICAL IMPORTS ---
import chromadb
from fuzzywuzzy import fuzz
# --------------------------

# --- CONFIGURATION (Final and Corrected Limits) ---
CHROMA_PATH = "./chroma_store_GONECASE2"
CHROMA_COLLECTION_NAME = "legal_judgments_collection"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
MAX_CONTEXT_TOKENS = 10000 

# Retrieval and Context Management
FUZZY_MATCH_THRESHOLD = 75
# CRITICAL FIX: Set the safe character limit based on 8000 tokens (8000 * 4 = 32000 chars)
MAX_CONTEXT_CHARS_FOR_CHAINING = 8000 * 4 
SAFE_CHUNK_SIZE_FOR_SUMMARY = 8000
SEMANTIC_ACCEPTANCE_THRESHOLD = 0.75 
TOP_K_FALLBACK_COUNT = 3

# Define schemas (rest remains the same)
CASE_NAME_SCHEMA = {"type": "object", "properties": {"case_name": {"type": "string", "description": "The cleaned, primary name of the case..."}}, "required": ["case_name"]}


class LegalRAGSystem:
    
    # --- UTILITY METHODS ---
    
    def _fuzzy_match_source_id(self, target_name_lower: str, metadata_map: Dict[str, str]) -> Tuple[str, str, int]:
        """Scans the entire metadata map keys for the best fuzzy match score."""
        best_match_id = None; best_score = 0
        for case_title_lower, source_id in metadata_map.items():
            score = fuzz.token_set_ratio(target_name_lower, case_title_lower)
            if score > best_score:
                best_score = score
                best_match_id = source_id
            if best_score == 100: break
        return best_match_id, None, best_score

    def _extract_case_name_with_llm(self, raw_query: str) -> str:
        """Uses LLM to cleanly extract the primary case name from a noisy user query."""
        prompt = f"""
        Extract the full, clean case name from the user's query. Your response MUST be a single JSON object conforming to the schema {{ "case_name": "CLEAN NAME HERE" }}.
        
        USER QUERY: {raw_query}
        """
        try:
            response = self.llm.invoke(prompt)
            json_start = response.find('{'); json_end = response.rfind('}') + 1
            extracted_data = json.loads(response[json_start:json_end])
            case_name = extracted_data.get('case_name', '').strip()
            if case_name: return case_name
        except Exception: pass
        if "full summary of" in raw_query.lower():
            return raw_query.replace("full summary of", "", 1).strip()
        return raw_query.strip()
    
    def _get_full_document_context(self, source_id: str, case_title: str) -> str:
        """Aggregates ALL chunks for a given SourceID (Parent-Child RAG)."""
        full_chunks = self.vectorstore._collection.get(where={"ParentID": source_id}, include=["documents"])
        full_context = "\n\n".join(full_chunks.get('documents', []))
        max_chars = MAX_CONTEXT_TOKENS * 4 
        if len(full_context) > max_chars * 2: full_context = full_context[:max_chars * 2]
        return full_context

    def _call_llama_with_tool_check(self, prompt: str) -> Tuple[str, Any]:
        """Submits the prompt to Llama 3 8B and checks if the output is a tool call."""
        system_instruction = ("You are a Legal RAG Agent specializing in Indian Law...")
        try:
            response = self.llm.invoke(f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER PROMPT: {prompt}")
            if '"tool_call": "retrieve_full_document"' in response:
                json_start = response.find('{'); json_end = response.rfind('}') + 1
                tool_data = json.loads(response[json_start:json_end])
                if tool_data.get('tool_call') == 'retrieve_full_document' and 'source_id' in tool_data: return "TOOL_CALL", tool_data
            return "TEXT_RESPONSE", response
        except Exception:
            return "TEXT_RESPONSE", "Error communicating with LLM during tool check. Check Ollama."
    
    # CRITICAL FIX: TEMPORARY TRUNCATION FUNCTION (Context Overload Fix)
    def _summarize_context_chain(self, long_context: str, case_title: str) -> str:
        """
        [TEMPORARY FIX] Truncates context to the safe limit when chaining is not feasible.
        """
        if len(long_context) > MAX_CONTEXT_CHARS_FOR_CHAINING:
            print(f"[Context Fix] Truncating large context ({len(long_context)} chars) to {MAX_CONTEXT_CHARS_FOR_CHAINING} chars.")
            return long_context[:MAX_CONTEXT_CHARS_FOR_CHAINING]
        return long_context

    def _list_fallback_cases(self, initial_docs_with_scores: List[Tuple[Any, float]]) -> str:
        """Formats a list of top-K conceptually related cases for the LLM output."""
        relevant_cases = [(doc, score) for doc, score in initial_docs_with_scores if score >= SEMANTIC_ACCEPTANCE_THRESHOLD]
        if not relevant_cases: return "NO_RELEVANT_CASES_FOUND"
        case_scores = {}; 
        for doc, score in relevant_cases:
            source_id = doc.metadata.get('ParentID'); title = doc.metadata.get('CaseTitle', 'Unknown Case')
            if source_id not in case_scores or score > case_scores[source_id][0]: case_scores[source_id] = (score, title)
        sorted_cases = sorted(case_scores.items(), key=lambda item: item[1][0], reverse=True)[:TOP_K_FALLBACK_COUNT]
        output = "\n".join([f"- {title} (ID: {source_id}) [Score: {score:.2f}]" for source_id, (score, title) in sorted_cases])
        return output

    # --- DISENTANGLED GENERATION FUNCTIONS ---

    def _synthesize_fictional_analysis(self, query: str) -> str:
        """Handles the Fictional Case Override (Analysis Mode)."""
        law_keywords = "Article 14 non-arbitrariness, procedural fairness, natural justice, audi alteram partem"
        law_chunks = self.vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(law_keywords)
        retrieved_law_context = "\n---\n".join([f"Source: {doc.metadata.get('CaseTitle', 'COI/Act')}, Chunk Index {doc.metadata.get('ChunkIndex')}\nContent: {doc.page_content}" for doc in law_chunks])

        synthesis_prompt = f"""You are a Constitutional Law Analyst. Your task is to analyze the fictional scenario provided by the user... [Aggressive Fictional Analysis Prompt structure] ..."""
        try:
            return self.llm.invoke(synthesis_prompt)
        except Exception as e:
            return f"[LLM ERROR] Failed to generate fictional summary. Error: {e}"


    def _synthesize_real_summary(self, context: str, query: str, primary_metadata: Dict[str, Any]) -> str:
        """Handles REAL Case Summary: Extracts and summarizes the provided context."""
        case_name = primary_metadata.get('CaseTitle', 'N/A'); citation = primary_metadata.get('Citation', 'N/A')
        date = primary_metadata.get('VerdictDate', 'N/A'); bench = primary_metadata.get('Bench', 'N/A')
        
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
            return self.llm.invoke(synthesis_prompt)
        except Exception as e:
            return f"[LLM ERROR] Failed to generate final structured summary. The Ollama connection may be unstable. Error: {e}"
            
            
    # --- INITIALIZATION ---
    def __init__(self):
        print(f"Loading consistent embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, 
            model_kwargs={'device': DEVICE}, 
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32} 
        )
        
        print("Loading ChromaDB persistent client...")
        try:
            self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=CHROMA_COLLECTION_NAME,
                embedding_function=self.embedding_model,
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10}) 
        except Exception as e:
            print(f"CRITICAL: Failed to initialize ChromaDB. Error: {e}"); exit()
            
        print(f"Connecting to Ollama using model: {LLM_MODEL}")
        self.llm = Ollama(model=LLM_MODEL)
        
        parent_metadata_file = os.path.join(CHROMA_PATH, "parent_metadata.json")
        try:
            with open(parent_metadata_file, 'r', encoding='utf-8') as f:
                raw_metadata = json.load(f)
            self.parent_metadata = raw_metadata
            
            self.case_name_to_id_map = {
                v.get('CaseTitle', '').lower(): k
                for k, v in self.parent_metadata.items()
            }
            print(f"Loaded {len(self.parent_metadata)} Parent Document records.")
        except Exception as e:
            print(f"CRITICAL: Failed to load parent metadata store. Error: {e}"); exit()


    def answer_query(self, query: str) -> str:
        
        if "fictional" in query.lower() or "(2026)" in query or "(2029)" in query:
            return self._synthesize_fictional_analysis(query) 

        # 1. LLM-ASSISTED NAME EXTRACTION 
        target_case_name = self._extract_case_name_with_llm(query)
        target_case_name_lower = target_case_name.lower()
        
        # 2. FUZZY METADATA LOOKUP 
        best_match_id, _, best_score = self._fuzzy_match_source_id(
            target_case_name_lower, 
            self.case_name_to_id_map
        )
        
        # --- Contexts initialized for scope ---
        initial_docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=10)
        initial_docs = [doc for doc, score in initial_docs_with_scores]
        if not initial_docs: return "I could not find any documents or legal precedents relevant to your query in the database."
        initial_context = "\n---\n".join([doc.page_content for doc in initial_docs])
        final_context = initial_context
        
        primary_metadata = self.parent_metadata.get(initial_docs[0].metadata.get('ParentID'), {})
        primary_case_title = primary_metadata.get('CaseTitle', 'Unknown')
        
        
        if best_match_id and best_score >= FUZZY_MATCH_THRESHOLD:
            # --- ROUTE 1: DIRECT MATCH SUCCESS (Fixed Path) ---
            primary_source_id = best_match_id
            primary_metadata = self.parent_metadata.get(primary_source_id, primary_metadata)
            primary_case_title = primary_metadata.get('CaseTitle', target_case_name)
            
            print(f"[Direct Match SUCCESS via FUZZY MATCH] Found ID {primary_source_id} (Score: {best_score}).")

            # 3. DIRECT AGGREGATION 
            full_document_context = self._get_full_document_context(primary_source_id, primary_case_title)

            # --- CONTEXT MANAGEMENT (TRUNCATION FIX) ---
            final_context_for_synthesis = full_document_context
            
            if len(full_document_context) > MAX_CONTEXT_CHARS_FOR_CHAINING:
                print(f"[Context Fix] Truncating large context ({len(full_document_context)} chars) to {MAX_CONTEXT_CHARS_FOR_CHAINING} chars.")
                final_context_for_synthesis = self._summarize_context_chain(full_document_context, primary_case_title)
            
            final_context = final_context_for_synthesis
            print(f"[Success] Context prepared for final synthesis.")

            return self._synthesize_real_summary(final_context, query, primary_metadata)

        # --- ROUTE 2: FALLBACK & HALLUCINATION GUARDRAIL ---
        
        print(f"[Direct Match FAILURE] Best fuzzy score ({best_score}) was below {FUZZY_MATCH_THRESHOLD}. Falling back to semantic search.")
        
        # 4. THRESHOLD REJECTION LOGIC
        top_semantic_score = initial_docs_with_scores[0][1] if initial_docs_with_scores else 0

        if top_semantic_score < SEMANTIC_ACCEPTANCE_THRESHOLD:
            # 4. THRESHOLD REJECTION (The anti-hallucination fix)
            fallback_list_text = self._list_fallback_cases(initial_docs_with_scores)
            
            no_case_found_prompt = f"""
            **ATTENTION:** No directly relevant legal precedent was found in the database that meets the acceptance threshold ({SEMANTIC_ACCEPTANCE_THRESHOLD:.2f}) for the query: '{query}'. 
            
            The top conceptually related cases found are:
            {fallback_list_text}

            **INSTRUCTION:** Acknowledge that a direct precedent was not found. Then, provide a high-level summary of the constitutional doctrines involved in the user's request (e.g., Surveillance, Due Process, Article 14). Do NOT fabricate any links between the listed cases and the user's specific query.
            """
            
            return self.llm.invoke(no_case_found_prompt)


        # 5. SEMANTIC ACCEPTED (Proceed with standard aggregation if LLM requests it)
        primary_source_id = initial_docs_with_scores[0][0].metadata.get('ParentID')
        primary_metadata = self.parent_metadata.get(primary_source_id, {})
        primary_case_title = primary_metadata.get('CaseTitle', 'Unknown')
        
        print(f"[Semantic Fallback ACCEPTED] Using top-ranked result: {primary_case_title} (Score: {top_semantic_score:.2f})")

        # Use tool check path to decide if full aggregation is needed
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
        
        action_type, response_data = self._call_llama_with_tool_check(initial_prompt)
        
        if action_type == "TOOL_CALL":
            try:
                target_id = response_data['source_id']; target_title = self.parent_metadata.get(target_id, {}).get('CaseTitle', 'Unknown Case')
                full_document_context = self._get_full_document_context(target_id, target_title)
                
                # Check for chaining again
                if len(full_document_context) > MAX_CONTEXT_CHARS_FOR_CHAINING:
                     final_context = self._summarize_context_chain(full_document_context, target_title)
                else:
                     final_context = full_document_context
                
            except Exception as e:
                print(f"[Fallback] Error executing full retrieval tool call: {e}. Proceeding with initial context.")
                final_context = initial_context

        elif action_type == "TEXT_RESPONSE":
            if "full, comprehensive summary" not in query.lower() and "headnote" not in query.lower():
                return response_data

        # 6. FINAL GENERATION PASS
        return self._synthesize_real_summary(final_context, query, primary_metadata)


if __name__ == "__main__":
    
    print("--- WARNING: Ensure all prerequisites are installed. ---")
    
    rag_system = LegalRAGSystem()
    print("\n--- Legal RAG System Ready ---")
    
    while True:
        user_query = input("\nEnter your legal query (or type 'quit'): ")
        if user_query.lower() == 'quit':
            break

        query_case = user_query
        
        print(f"\nProcessing Query: {query_case}")
        
        try:
            response_case = rag_system.answer_query(query_case)
            print("\n[LLM RESPONSE] *************************************")
            print(response_case)
            print("*************************************************")
        except Exception as e:
            print(f"\n[FATAL ERROR] An unexpected error occurred during the RAG process: {e}")
            break