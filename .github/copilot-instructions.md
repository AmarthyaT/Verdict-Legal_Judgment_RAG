# Legal RAG System - Project Guidelines

## Project Overview

**rag_proto** is a Retrieval-Augmented Generation system for querying Supreme Court judgments. Users ask questions about legal cases, and the system retrieves semantically similar judgment excerpts, then synthesizes structured summaries using a local LLM.

**Key distinction**: Multiple "Final" versions exist (FInalMain.py, FinalTeaquedFRICTIONAL.py, equallytunedBOTH.py) representing experimental retrieval strategies. This is active prototyping, not production code.

## Code Style

- **Python 3.12** with venv312 environment
- **PEP 8** compliance preferred (no strict linting enforced)
- **Type hints** not consistently used; inference OK via agent tools
- **Error handling**: Sparse; focus on adding try-except for external integrations (Ollama, HuggingFace)
- **Key libraries**: LangChain (core, community, text-splitters), ChromaDB, Streamlit, PyMuPDF

## Architecture

### Core Components

| Component | Responsibility |
|-----------|-----------------|
| **Query Engine** (`FInalMain.py` / `FinalTeaquedFRICTIONAL.py`) | Case name extraction, metadata fuzzy-matching, vector retrieval, context aggregation, LLM synthesis |
| **Embedding Pipeline** (`persistent_embedder.py`) | PDF chunking (1024 chars, 100 overlap), metadata extraction, parent-child hierarchy creation, vector embedding |
| **Vector Store** (`chroma_store_GONECASE2/`) | ChromaDB with SQLite backend, stores embeddings + parent metadata |
| **LLM Integration** | Ollama (llama3.1-finetuned GGUF Q8_0 quantized) for JSON parsing and synthesis |
| **UIs** | `app.py` (Streamlit web + auth), `main.py` (CLI with diagnostics), `Streammain.py` (advanced Streamlit) |
| **Metadata** | `structured_judgments.json` (case records), `config.yaml` (Streamlit credentials) |

### Data Flow

```
User Query → Case Name Extraction (LLM) → Fuzzy Metadata Match 
  → Vector Search → Parent Doc Aggregation → Context Chaining (if >8K chars) 
  → LLM Synthesis → Return Structured Summary
```

**Key insight**: Parent-child hierarchy in ChromaDB allows partial document retrieval (chunks) with full context reconstruction.

## Build and Test

### Environment Setup
```powershell
# Activate Python 3.12 environment (already configured in venv312/)
source venv312/Scripts/Activate.ps1

# Install dependencies (if running fresh)
pip install -r requirements.txt
# Minimal: chromadb, langchain-ollama, langchain-core
```

### Add Documents (Embedding Pipeline)
```powershell
# Place PDF files in a folder (e.g., C:\Users\tamar\Desktop\judg\pdfs)
# Ensure case record exists in structured_judgments.json

# Run embedding
python persistent_embedder.py
# Produces: chroma_store_GONECASE2/ + parent_metadata.json

# Validate
python CHECdb.py  # Expected: ~33K embeddings for full dataset
```

### Query (Choose One Interface)
```powershell
# CLI (recommended for testing/diagnostics)
python main.py
# → Interactive prompt; type case name or question
# → Includes diagnose_collection() and diagnose_store_files() for DB inspection

# Streamlit Web UI (end-user facing)
streamlit run app.py
# → Login as "legal_analyst" (see config.yaml)
# → Open http://localhost:8501

# Alternative Streamlit (advanced context chaining)
streamlit run Streammain.py
```

### Diagnostics
```powershell
# Inspect vector store status
python main.py  # Check diagnose_collection() and diagnose_store_files() output

# Check metadata alignment
python checkdb.py  # Validate structured_judgments.json integrity

# View embeddings (optional)
python viz.py  # Renumics Spotlight visualization
```

## Conventions

### Document Management
- **Primary data**: `chroma_store_GONECASE2/` (current production set)
- **Legacy stores**: `chroma_store1/`, `chroma_store_BORE_06_27/` (kept for A/B testing)
- **Metadata source**: Always validate case names against `structured_judgments.json` first; fallback to filename parsing only if not found

### Query Strategy Selection
| Query Type | Entry Point | Strategy |
|------------|-------------|----------|
| Basic search | `FInalMain.py` | Fuzzy case name match + vector retrieval |
| Large documents (>10K chars) | `FinalTeaquedFRICTIONAL.py` | Context chaining to split long context |
| Robust fallback | `equallytunedBOTH.py` | Dual synthesis (real + fallback) if LLM fails |
| Diagnostics | `main.py` | CLI with database inspection |

### File Naming Patterns
- `*Final*.py` → Attempts at complete query logic (ignore typos like `FInalMain`)
- `embed*.py` → Evolution of embedding (embed.py primary, embed1.py legacy with Google Gemini)
- `main*.py` → Baseline CLI variants (main.py primary, main2.py tests BM25)
- `check*.py` → Validation utilities (CHECdb, checkdb)

### Environment & Secrets
- `.env` → Unused; could store Ollama URL or HuggingFace tokens if needed
- `a.env` → Deprecated (Google API key, removed from codebase)
- `config.yaml` → Streamlit auth only; credentials are bcrypt-hashed

## Key Pitfalls & Workarounds

### Critical
1. **Ollama must be running**
   - No error handling for missing server; code will hang or crash
   - Workaround: Always verify `ollama serve` is active in a separate terminal before running queries

2. **GPU auto-detection**
   - `torch.cuda.is_available()` silently falls back to CPU
   - Embedding 33K documents on CPU takes hours
   - Workaround: Force GPU if available, or use pre-computed embeddings

3. **Fuzzy matching threshold = 75**
   - Too high = misses typos in case names; too low = false positives
   - Tuned empirically for legal case names
   - Workaround: If you change it, re-validate against structured_judgments.json

### Non-Critical
4. **Context window limits**
   - MAX_CONTEXT_TOKENS=10K set high; LLM fails on 50K+ char docs without chaining
   - Use `FinalTeaquedFRICTIONAL.py` for large documents

5. **HuggingFace model inconsistency**
   - `BAAI/bge-base-en-v1.5` (small) vs `bge-large-en` (large) not standardized
   - Workaround: Use bge-large-en for all new embeddings unless memory-constrained

6. **Multiple Chroma stores**
   - Easy to pick wrong path in config; no enum to prevent desync
   - Workaround: Document which store is "active" in a .txt file, check before querying

7. **Metadata JSON fragility**
   - Cases not in `structured_judgments.json` get `[UNSTRUCTURED]` prefix
   - Filename parsing fallback is unreliable (no standardized naming)
   - Workaround: Keep JSON normalized; add new cases before embedding

## Next Steps

1. **Consolidate UIs**: Choose one primary entry point (app.py recommended)
2. **Add error handling**: Ollama connection, HF model fallback, Chroma schema validation
3. **Config management**: Use enum/constants for Chroma store paths instead of hardcoded strings
4. **Document decision tree**: Create a flow chart for which query file to use when
5. **Consider production hardenings**: Caching, query result scoring, audit logging
