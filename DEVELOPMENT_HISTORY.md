# Verdict Development History & File Map

This document serves as a guide for third-party reviewers to understand the history of this project, its code iterations, and what each file in the `/experimental/` directory represents.

---

## 🗺️ Codebase Sitemap

```
├── data/
│   ├── chroma_store/             # Active vector database (ChromaDB)
│   │   ├── chroma.sqlite3        # SQLite persistent storage for child chunks
│   │   └── parent_metadata.json  # Catalog linking SourceIDs to case headnotes
│   └── structured_judgments.json # Structured case metadata maps
├── docs/
│   ├── presentation/
│   │   └── verdict_Amarthya_1AY23CD066.pptx # Final college project presentation
│   └── ILC_MODEL_EVALUATION_SUITE.md        # Evaluation benchmarking scenarios
├── experimental/                 # Archives & historical iterations (See details below)
│   ├── app_old.py                # Old Streamlit draft (pre-aesthetic update)
│   ├── Streammain.py             # Streamlit prototype with Ollama local model
│   ├── FInalMain.py              # CLI prototype of Streammain
│   ├── FinalTeaquedFRICTIONAL.py # CLI prototype focusing on Fictional Scenario overrides
│   ├── equallytunedBOTH.py       # CLI prototype balancing Real and Fictional cases
│   ├── embed.py                  # First layout-naive embedding script (Gemini-based)
│   ├── embed1.py                 # Second embedding draft with PyMuPDF
│   ├── vector.py                 # Core vector math experiments
│   ├── viz.py & viz_diag.py      # Vector space visualizations
│   ├── checkdb.py & checkbddb.py # Database debug and connection checking tools
│   ├── CHECdb.py                 # Diagnostic script inspecting Chroma chunk counts
│   ├── embedcheckerDB.py         # Verification script querying Chroma database indices
│   ├── query_documents.py        # Early prototype querying direct local GGUF models
│   ├── rag_test_with_newDB.py    # Streamlit prototype testing MiniLM vs BGE models
│   ├── main.py & main2.py        # Core generation testbeds with local LLMs
│   ├── main2_bm25test.py         # Sparse retrieval testbed using BM25
│   ├── myffug.py                 # Simple query helper sandbox script
│   ├── unique_case_names.txt     # List of extracted case keys
│   ├── renaming_log.txt          # Logging file for document cleaning scripts
│   ├── extraction_errors.txt     # Parsing diagnostic logs
│   ├── rag.txt                   # Sandbox prompt templates notes
│   ├── chroma_store1/            # Archive: Older test vector database
│   └── chroma_store_BORE_06_27/  # Archive: Older test vector database
├── app.py                        # PRODUCTION ENTRYPOINT: Unified, styled Streamlit UI
├── persistent_embedder.py        # Core parent-child document chunker and DB generator
├── MakeQA.py                     # Secure synthetic dataset generator using RAGAS
├── requirements.txt              # Standard python dependencies
└── .gitignore                    # Local environment and private asset guards
```

---

## ⏳ Project Evolution & Lessons Learned

### Phase 1: Local GGUF and Vector Basics (`query_documents.py`, `main.py`)
*   **The Approach**: The earliest version of the project attempted to query a local GGUF model (`unsloth.Q4_K_M.gguf`) directly using `llama-cpp-python` and generate embeddings via `nomic-embed-text-v1.5` APIs.
*   **The Problems**:
    *   Direct GGUF model handling in python was unstable, memory-heavy, and had CPU/GPU allocation issues on local machines.
    *   Naive chunking severed key structural segments of legal documents, leading to fragmented context.
*   **Archived Files**:
    *   [query_documents.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/query_documents.py): Early CLI pipeline using llama-cpp-python.
    *   [main.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/main.py) / [main2.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/main2.py): Testing local text splitters and templates.

### Phase 2: Embedding Refinements and Ingestion (`embed.py`, `embed1.py`, `vector.py`)
*   **The Approach**: Initial ingestion experiments used `fitz` (PyMuPDF) and naive section headings splits, aiming to build a vector db using Google Gemini API embeddings.
*   **The Lessons**:
    *   Relying on APIs to embed hundreds of pages of judgments was expensive and prone to rate limits.
    *   Switching to a local embedding standard (`BAAI/bge-large-en` or `BAAI/bge-base-en-v1.5`) proved to be fast, free, and highly accurate.
*   **Archived Files**:
    *   [embed.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/embed.py) / [embed1.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/embed1.py): Layout-naive ingestion scripts.
    *   [vector.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/vector.py): Basic similarity distance math.
    *   [viz.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/viz.py) / [viz_diag.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/viz_diag.py): Vector spacing plotting scripts.

### Phase 3: Transition to Ollama & Streamlit Prototypes (`rag_test_with_newDB.py`)
*   **The Approach**: Moving away from direct GGUF files, the project adopted **Ollama** as the local model provider, standardizing on the `llama3` 8B model. This allowed standard LangChain abstractions.
*   **The Lessons**:
    *   Standard retrieval (top 5 chunks) failed for comprehensive summaries. If a user asked for a "full summary," the LLM needed to read the *entire* case.
    *   Led to the development of the **Parent-Child RAG** strategy, implemented in `persistent_embedder.py`. Chunking split the main body, but retained a Parent ID mapping to retrieve all pages when needed.
*   **Archived Files**:
    *   [rag_test_with_newDB.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/rag_test_with_newDB.py): Streamlit draft integrating SentenceTransformers and path fallbacks.

### Phase 4: Intent Routing & Fictional Scenario Tuning (`FinalTeaquedFRICTIONAL.py`, `equallytunedBOTH.py`, `Streammain.py`)
*   **The Approach**: In testing (refer to the `ILC_MODEL_EVALUATION_SUITE.md`), standard RAG models failed when asked about fictional scenarios. They would either confidently hallucinate that the fake case exists or return an error saying nothing is in the database.
*   **The Fixes**:
    *   **LLM Case Name Extraction**: Extracts the candidate case name.
    *   **Fuzzy Lookup**: Matches names against the parent database metadata. If a case match score is < 75%, it alerts the user.
    *   **Fictional Scenario Routing**: Detects fictional signatures (like "Ritesh Varma 2029") and switches the prompt to a Constitutional Analysis mode, pulling real principles (like Article 14 or audi alteram partem) to analyze the hypothetical facts.
*   **Archived Files**:
    *   [FinalTeaquedFRICTIONAL.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/FinalTeaquedFRICTIONAL.py): Tuned prompt specifically for fictional synthesis routing.
    *   [equallytunedBOTH.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/equallytunedBOTH.py): CLI balancing both paths.
    *   [Streammain.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/Streammain.py): Streamlit prototype incorporating these methods.
    *   [app_old.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/app_old.py): Previous front-page draft.

---

## 🧠 The Fine-Tuned Model: Indian Legal Llama 3.1

To achieve optimal abstractive capabilities, a specialized model was fine-tuned and published on Hugging Face:
*   **Model Repository**: [Amarthya11/Llama3.1-finetuned-GGUF](https://huggingface.co/Amarthya11/Llama3.1-finetuned-GGUF)
*   **Fine-Tuning Methodology**: Fine-tuned from `unsloth/meta-llama-3.1-8b-unsloth-bnb-4bit` using **Unsloth & QLoRA** (4-bit quantization).
*   **Dataset Significance**: Trained on the **Indian Legal Corpus (ILC)**. Navigating complex legal proceedings requires abstractive reasoning. Traditional summarization methods focus on text-extractive summaries (simple compression). The ILC corpus features human-written abstractive summaries, allowing the model to understand the core intent, procedural facts, and the binding *ratio decidendi* of judgments.
*   **Format**: Distributed in GGUF format (`unsloth.Q4_K_M.gguf`, ~4.92 GB) for lightweight local CPU/GPU execution.

---

## 🛠️ Database Debugging History (`checkdb.py`, `CHECdb.py`, etc.)

During structural migration, ChromaDB collections often mismatch or get corrupted due to relative paths. The following utilities were built to debug index errors:
*   [checkdb.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/checkdb.py) / [checkbddb.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/checkbddb.py): Small scripts verifying Chroma DB is readable.
*   [CHECdb.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/CHECdb.py): Counts persistent embedded chunks in the database.
*   [embedcheckerDB.py](file:///c:/Users/tamar/Desktop/programming/github_repos/rag_proto/experimental/embedcheckerDB.py): Tests distance metrics and returns matching parent documents.
