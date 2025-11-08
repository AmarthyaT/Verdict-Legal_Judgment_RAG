import os
import sys
import textwrap
import chromadb
from nomic import embed
from llama_cpp import Llama

# --- CONFIGURATION ---
COLLECTION_NAME = "legal-pdf-embeddings"
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_store")

# Path to your GGUF model file. You can set it via the environment variable MODEL_PATH
# or place the file in the same folder and use the filename below as default.
DEFAULT_MODEL_FILENAME = "unsloth.Q4_K_M.gguf"
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(__file__), DEFAULT_MODEL_FILENAME))

# LLM generation params
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))

# --- PROMPT TEMPLATE ---
TEMPLATE = textwrap.dedent("""
You are an expert legal analyst. You will be provided with excerpts from a legal judgment retrieved based on a user's question.
Your task is to synthesize the information from these excerpts to answer the user's question and provide a structured summary based on the available text.
You are an expert in legal documents. You will be provided with excerpts from a legal judgment. Also include any relevant case law or statutes. 
You are supposed to summarise the document in points, covering 1/3 to 1/2 of the original text, with the following sections:

1. Title of the document
2. Parties involved
3. Effective date
4. Key terms and conditions
5. Governing law
6. Key issues
7. Obligations and responsibilities
8. Facts and evidences presented before the court
9. Any other important information
10. Key arguments presented by each party
11. Ratio decidendi
12. Reasons for the decision
13. Any dissenting opinions
14. Implications of the case
15. Precedents cited
16. Legal principles established
17. Final judgment or outcome of the case
18. Subsequent developments or amendments

(MAKE SURE TO NEVER MISS ARGUMENTS PRESENTED BY EACH PARTY AND THE REASONS FOR THE DECISION BY THE COURT that should be the majority of the summary)

---
**Provided Excerpts:**
{document}
---
**User's Question:**
{question}
---
**Your Answer and Summary:**
""")


def embed_query(query_text: str):
    """Embeds the user's query using Nomic."""
    try:
        response = embed.text(
            texts=[query_text],
            model="nomic-embed-text-v1.5",
            task_type="search_query",
        )
        return response["embeddings"][0]
    except Exception as e:
        print(f"⚠️ Query embedding failed: {e}")
        return None


def extract_text_from_llama_response(resp: dict) -> str:
    """Try several common keys returned by llama-cpp-python to extract generated text."""
    if not resp:
        return ""
    # Some versions return choices -> text
    choices = resp.get("choices")
    if choices and isinstance(choices, list):
        first = choices[0]
        # new versions: 'message' -> {'role':..., 'content': '...'}
        if isinstance(first, dict):
            if "text" in first and first["text"]:
                return first["text"]
            if "message" in first and isinstance(first["message"], dict):
                return first["message"].get("content", "")
    # older/simple shape
    if "text" in resp and isinstance(resp["text"], str):
        return resp["text"]
    # fallback
    return str(resp)


def main():
    # --- INITIALIZE MODELS AND DATABASE ---
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception as e:
        print(f"❌ Failed to initialize chroma database: {e}")
        sys.exit(1)

    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Set MODEL_PATH env var to your GGUF file.")

        print(f"Loading local GGUF model from: {MODEL_PATH}")
        llm = Llama(model_path=MODEL_PATH)
    except Exception as e:
        print(f"❌ Failed to initialize Llama model: {e}")
        sys.exit(1)

    # --- MAIN LOOP ---
    while True:
        question = input("Ask a question about your documents (or enter 'q' to quit): ")
        if question.lower() == 'q':
            print("Exiting...")
            break

        print("\nEmbedding your query...")
        query_embedding = embed_query(question)
        if query_embedding is None:
            print("⚠️ Could not embed query. Please try again.")
            continue

        print("🔍 Searching for relevant document chunks...")
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=15,
            )
        except Exception as e:
            print(f"⚠️ Chroma query failed: {e}")
            continue

        documents = results.get("documents")
        if not documents or not documents[0]:
            print("⚠️ No relevant documents were found to answer your question.")
            continue

        relevant_chunks = documents[0]
        combined_text = "\n\n---\n\n".join(relevant_chunks)

        print(f"✅ Found {len(relevant_chunks)} relevant chunks. Generating response...")

        # fill template
        prompt_text = TEMPLATE.format(document=combined_text, question=question)

        try:
            resp = llm(
                prompt_text,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            output_text = extract_text_from_llama_response(resp)

            print("\n" + "="*50 + " RESPONSE " + "="*50)
            print(output_text)
            print("="*110 + "\n")
        except Exception as e:
            print(f"⚠️ An error occurred while generating the response: {e}")


if __name__ == "__main__":
    main()
