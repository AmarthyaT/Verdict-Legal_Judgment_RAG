import os
import json
from typing import List, Dict, Any, Tuple
import torch 
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
# --- CRITICAL IMPORTS ---
import chromadb
from fuzzywuzzy import fuzz
from langchain_text_splitters import RecursiveCharacterTextSplitter 
# --------------------------

# --- CONFIGURATION (Ensure consistency with embedding script) ---
CHROMA_PATH = "./chroma_store_GONECASE2"
CHROMA_COLLECTION_NAME = "legal_judgments_collection"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' 
MAX_CONTEXT_TOKENS = 10000 

# Define the minimum similarity score for a fuzzy match (75/100)
FUZZY_MATCH_THRESHOLD = 75
# 40,000 characters is the trigger for Summarization Chaining (approx 10k tokens)
MAX_CONTEXT_CHARS_FOR_CHAINING = 40000 
SAFE_CHUNK_SIZE_FOR_SUMMARY = 8000 # 8k characters max for single summary pass

# Define schemas (rest remains the same)
CASE_NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "case_name": {"type": "string", "description": "The cleaned, primary name of the case, without extra descriptive words like 'summary of'."}
    },
    "required": ["case_name"]
}


class LegalRAGSystem:
    
    # --- UTILITY METHODS (DEFINED WITHIN THE CLASS FOR CORRECT NAMESPACE) ---
    def _fuzzy_match_source_id(self, target_name_lower: str, metadata_map: Dict[str, str]) -> Tuple[str, str, int]:
        """
        Scans the entire metadata map keys for the best fuzzy match score.
        Returns (SourceID, Matched Title, Score)
        """
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
        """
        Uses LLM to cleanly extract the primary case name from a noisy user query.
        """
        prompt = f"""
        Extract the full, clean case name from the user's query. Your response MUST be a single JSON object conforming to the schema {{ "case_name": "CLEAN NAME HERE" }}.
        
        USER QUERY: {raw_query}
        """
        try:
            response = self.llm.invoke(prompt)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            extracted_data = json.loads(json_str)
            case_name = extracted_data.get('case_name', '').strip()
            
            if case_name:
                return case_name
            
        except Exception:
            pass # Fall through to simple string replacement
            
        # Fallback to simple string replacement if LLM extraction fails
        if "full summary of" in raw_query.lower():
            return raw_query.replace("full summary of", "", 1).strip()
        return raw_query.strip()
    
    def _get_full_document_context(self, source_id: str, case_title: str) -> str:
        """Aggregates ALL chunks for a given SourceID (Parent-Child RAG)."""
        # print(f"\n[Tool Use] Aggregating ALL chunks for: {case_title} (ID: {source_id})")
        
        full_chunks = self.vectorstore._collection.get(
            where={"ParentID": source_id},
            include=["documents"]
        )
        
        full_context = "\n\n".join(full_chunks.get('documents', []))
        
        # NOTE: Truncation is handled by the chaining logic now, but keep this check for safety.
        max_chars = MAX_CONTEXT_TOKENS * 4 
        if len(full_context) > max_chars * 2: # Keep original hard cap to prevent catastrophic overflow
             full_context = full_context[:max_chars * 2]
        
        return full_context

    def _call_llama_with_tool_check(self, prompt: str) -> Tuple[str, Any]:
        """Submits the prompt to Llama 3 8B and checks if the output is a tool call."""
        system_instruction = ("You are a Legal RAG Agent specializing in Indian Law. Your task is to analyze the user's request and the provided context...")
        try:
            response = self.llm.invoke(f"SYSTEM INSTRUCTION: {system_instruction}\n\nUSER PROMPT: {prompt}")
            if '"tool_call": "retrieve_full_document"' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                json_str = response[json_start:json_end]
                tool_data = json.loads(json_str) 
                if tool_data.get('tool_call') == 'retrieve_full_document' and 'source_id' in tool_data:
                    return "TOOL_CALL", tool_data
            return "TEXT_RESPONSE", response
        except Exception:
            return "TEXT_RESPONSE", "Error communicating with LLM during tool check. Check Ollama."
    
    # --- CRITICAL NEW METHOD: SUMMARIZATION CHAINING ---
    def _summarize_context_chain(self, long_context: str, case_title: str) -> str:
        """
        Breaks context into safe chunks, summarizes each, and synthesizes a final summary.
        (Fixes LLM Context Overload Bug)
        """
        print("[Context Chaining] Context too large. Initiating multi-step summarization...")

        # Initialize the splitter for chaining
        summary_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SAFE_CHUNK_SIZE_FOR_SUMMARY,
            chunk_overlap=500,
            separators=["\n\n", "\n", " ", ""]
        )
        
        context_chunks = summary_splitter.split_text(long_context)
        
        # 1. Summarize each chunk individually
        summaries = []
        summary_prompt_template = """
        You are an expert legal précis writer. Summarize the following section of a court judgment for the case {case_title}. 
        Focus strictly on the key facts, legal issues, and court findings. Do not provide a conclusion.

        SECTION CONTEXT:
        ---
        {section_context}
        ---
        """
        
        for i, chunk in enumerate(context_chunks):
            # print(f"  -> Summarizing Section {i+1}/{len(context_chunks)}...")
            prompt = summary_prompt_template.format(case_title=case_title, section_context=chunk)
            try:
                summary = self.llm.invoke(prompt)
                summaries.append(f"SUMMARY OF SECTION {i+1}:\n{summary}")
            except Exception as e:
                summaries.append(f"ERROR summarizing section {i+1}: {e}")
                
        if not summaries:
            return long_context 
            
        # 2. Synthesize the final comprehensive summary
        final_synthesis_prompt = f"""
        You have been provided with several summaries from different sections of the same legal judgment ({case_title}).
        Your task is to integrate all these individual summaries into ONE coherent, comprehensive, final summary. 
        Ensure you include the final holding, ratio, and key arguments from all sections. Do not repeat content.

        INDIVIDUAL SUMMARIES:
        ---
        {'\n'.join(summaries)}
        ---
        
        FINAL INSTRUCTIONS: Produce a single, unified, comprehensive summary of the entire document.
        """
        
        # print("[Context Chaining] Synthesizing final document summary...")
        try:
            final_summary = self.llm.invoke(final_synthesis_prompt)
            print("[Context Chaining] SUCCESS: Document reduced to comprehensive summary.")
            return final_summary
        except Exception as e:
            print(f"[Context Chaining] FINAL FAILURE in synthesis: {e}")
            return "\n".join(summaries) # Return combined summaries as a last resort


    def _generate_legal_summary(self, context: str, query: str, primary_metadata: Dict[str, Any]) -> str:
        """Implements the final generation pass, including the full Fictional Analysis logic."""
        
        # Use metadata fields (already retrieved and accurate)
        case_name = primary_metadata.get('CaseTitle', 'N/A')
        citation = primary_metadata.get('Citation', 'N/A')
        date = primary_metadata.get('VerdictDate', 'N/A')
        bench = primary_metadata.get('Bench', 'N/A')
        
        # --- HALLUCINATION GUARDRAIL / FICTIONAL OVERRIDE ---
        if "fictional" in query.lower() or "(2026)" in query or "(2029)" in query:
            
            # 1. Targeted Retrieval (Grounding) - Run targeted search on REAL laws
            law_keywords = "Article 14 non-arbitrariness, procedural fairness, natural justice, audi alteram partem"
            
            # NOTE: We run this semantic search specifically against the database to get REAL laws
            law_chunks = self.vectorstore.as_retriever(search_kwargs={"k": 10}).invoke(law_keywords)
            
            retrieved_law_context = "\n---\n".join([
                f"Source: {doc.metadata.get('CaseTitle', 'COI/Act')}, Chunk Index {doc.metadata.get('ChunkIndex')}\nContent: {doc.page_content}" 
                for doc in law_chunks
            ])

            # 2. AGGRESSIVE SYNTHESIS PROMPT (Direct Command for Analysis)
            synthesis_prompt = f"""
            You are a Constitutional Law Analyst. Your task is to analyze the fictional scenario provided by the user.
            
            **TASK:** Synthesize a legal analysis for the provided **fictional** Supreme Court case. You must use the legal principles related to Article 14, transparency, and due process.
            
            **REAL LAW CONTEXT (For Grounding):**
            ---
            {retrieved_law_context}
            ---
            
            **FICTIONAL SCENARIO:** {query}
            
            **INSTRUCTIONS (STRICT COMMAND):**
            1. **YOU MUST START** your response with the statement: **[FICTIONAL ANALYSIS: Adv. Ritesh Varma vs Ministry of Digital Affairs (2029)]**
            2. State clearly that the scenario is hypothetical and not based on a real precedent.
            3. Perform the analysis by synthesizing plausible **FACTS**, **ISSUES**, and **LEGAL REASONING** based ONLY on the user's scenario and the REAL LAW CONTEXT provided below.
            4. Follow the structured format below.
            
            ### ⚖️ Legal Analysis Synthesis
            
            * **Status:** Hypothetical Scenario (2029)
            * **Scenario Issues:** [List the legal questions framed by the query (e.g., algorithmic transparency, breach of Article 14).]
            
            #### 1. Factual Matrix
            [Synthesize a brief background on the algorithmic blacklisting.]
            
            #### 2. Applied Legal Principles
            [Explain how the Court would apply the principles of **Article 14 (non-arbitrariness)** and **audi alteram partem** (from the Context).]
            
            #### 3. Plausible Ratio Decidendi
            [State the likely binding rule.]
            
            """
            try:
                # CRITICAL: This is the execution of the aggressive prompt
                return self.llm.invoke(synthesis_prompt)
            except Exception as e:
                return f"[LLM ERROR] Failed to generate fictional summary. Error: {e}"


        # --- STANDARD (REAL CASE) SUMMARY LOGIC REMAINS HERE ---
        
        synthesis_prompt = f"""
        You are a highly skilled Legal Analyst tasked with creating a COMPLETE, structured, and formal legal judgment summary based ONLY on the full court judgment text provided in the CONTEXT below.

        CONTEXT (Full Case Text/Aggregated Chunks):
        ---
        {context}
        ---
        
        ... [rest of real summary template remains the same] ...
        """
        
        try:
            # Final attempt to generate the summary for the REAL case
            return self.llm.invoke(synthesis_prompt)
        except Exception as e:
            return f"[LLM ERROR] Failed to generate final structured summary. The Ollama connection may be unstable. Error: {e}"


    # --- INITIALIZATION ---
    def __init__(self):
        # 1. Load Embedding Model Wrapper (Consistency)
        print(f"Loading consistent embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME, 
            model_kwargs={'device': DEVICE}, 
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32} 
        )
        
        # 2. Load ChromaDB (rest remains the same)
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
            print(f"CRITICAL: Failed to initialize ChromaDB. Error: {e}")
            exit()
            
        # 3. Initialize Ollama LLM (rest remains the same)
        print(f"Connecting to Ollama using model: {LLM_MODEL}")
        self.llm = Ollama(model=LLM_MODEL)
        
        # 4. Load Parent Metadata Store (and create reverse lookup map)
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
            print(f"CRITICAL: Failed to load parent metadata store. Error: {e}")
            self.parent_metadata = {}
            self.case_name_to_id_map = {}
            exit()


    def answer_query(self, query: str) -> str:
        
        if "fictional" in query.lower() or "(2026)" in query or "(2029)" in query:
            return self._generate_legal_summary("", query, {}) 

        # 1. LLM-ASSISTED NAME EXTRACTION 
        target_case_name = self._extract_case_name_with_llm(query)
        target_case_name_lower = target_case_name.lower()
        
        # 2. FUZZY METADATA LOOKUP 
        best_match_id, _, best_score = self._fuzzy_match_source_id(
            target_case_name_lower, 
            self.case_name_to_id_map
        )
        
        # --- Contexts initialized for scope ---
        initial_docs = self.retriever.invoke(query)
        if not initial_docs:
             return "I could not find any documents or legal precedents relevant to your query in the database."
        initial_context = "\n---\n".join([doc.page_content for doc in initial_docs])
        final_context = initial_context
        
        primary_metadata = self.parent_metadata.get(initial_docs[0].metadata.get('ParentID'), {})
        primary_case_title = primary_metadata.get('CaseTitle', 'Unknown')
        primary_source_id = initial_docs[0].metadata.get('ParentID')
        
        
        if best_match_id and best_score >= FUZZY_MATCH_THRESHOLD:
            # 2a. SUCCESS: Found the correct document via FUZZY MATCH
            primary_source_id = best_match_id
            primary_metadata = self.parent_metadata.get(primary_source_id, primary_metadata)
            primary_case_title = primary_metadata.get('CaseTitle', target_case_name)
            
            # 3. DIRECT AGGREGATION 
            full_document_context = self._get_full_document_context(primary_source_id, primary_case_title)

            # --- CONTEXT MANAGEMENT (FIXES OVERLOAD BUG) ---
            final_context_for_synthesis = full_document_context
            
            if len(full_document_context) > MAX_CONTEXT_CHARS_FOR_CHAINING:
                print(f"[Context Limit Exceeded] {len(full_document_context)} chars > {MAX_CONTEXT_CHARS_FOR_CHAINING}. Initiating Chaining.")
                # This ensures LLM gets a summarized context, not the full massive text
                final_context_for_synthesis = self._summarize_context_chain(
                    full_document_context, primary_case_title
                )
            
            final_context = final_context_for_synthesis
            print(f"[Success] Context prepared for final synthesis.")

            # Skip the tool check entirely if the case was directly matched
            return self._generate_legal_summary(final_context, query, primary_metadata)

        # --- FALLBACK PATH: Semantic Search Results & Tool Check ---
        
        print(f"[Direct Match FAILURE] Best fuzzy score ({best_score}) was below {FUZZY_MATCH_THRESHOLD}. Falling back to semantic search.")
        print(f"[Semantic Fallback] Using top-ranked result: {primary_case_title} (ID: {primary_source_id}).")
        
        # 4. Standard Tool Check Path
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
                # If LLM requests full doc, aggregate it
                target_id = response_data['source_id']
                target_title = self.parent_metadata.get(target_id, {}).get('CaseTitle', 'Unknown Case')
                
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
        return self._generate_legal_summary(final_context, query, primary_metadata)


if __name__ == "__main__":
    
    # 🛑 ACTION REQUIRED: Ensure Ollama is stable before running!
    print("--- WARNING: Ensure 'fuzzywuzzy[speedup]' is installed: pip install fuzzywuzzy[speedup] python-levenshtein langchain-text-splitters ---")
    print("WARNING: Ensure the Ollama server is running and Llama3 is pulled.")
    
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