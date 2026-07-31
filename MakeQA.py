import os
import pandas as pd
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Gemini Integration
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# RAGAS Imports
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# --- ⚙️ Configuration Section ---
PDF_FOLDER_PATH = "C:/Users/tamar/Desktop/judgments"   # Folder containing your PDFs
OUTPUT_CSV_FILE = "ragas_synthetic_testset_gemini.csv" # Output CSV file
TEST_SET_SIZE = 1000                                   # Number of Q&A pairs to generate

# Load API key securely from environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it before running the script.")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
# --- End Configuration Section ---


def generate_ragas_testset():
    """
    Loads PDFs, chunks them, and generates a synthetic RAGAS evaluation dataset
    using the Gemini API.
    """
    if not os.path.isdir(PDF_FOLDER_PATH):
        print(f"ERROR: PDF folder not found at '{PDF_FOLDER_PATH}'. Please check the path.")
        return

    print("--- 1. Loading and Chunking Documents ---")

    # Load all PDF files from the directory
    loader = DirectoryLoader(
    PDF_FOLDER_PATH,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

    documents = loader.load()

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=256,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Total documents loaded: {len(documents)}")
    print(f"Total chunks created: {len(chunks)}")
    print("-" * 40)

    print("--- 2. Initializing Gemini Models for RAGAS ---")

    # Initialize Gemini models and wrap them for RAGAS
    generator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
    )
    critic_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    )
    embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    )

    generator = TestsetGenerator(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings,
    )
    print("Gemini models initialized.")
    print("-" * 40)

    print(f"--- 3. Generating {TEST_SET_SIZE} Synthetic Q&A Pairs ---")

    query_distribution = default_query_distribution(generator.llm, generator.knowledge_graph)

    testset = generator.generate_with_langchain_docs(
        chunks,
        testset_size=TEST_SET_SIZE,
        query_distribution=query_distribution,
        raise_exceptions=False,
    )

    print("-" * 40)
    print("--- 4. Finalizing and Saving Dataset ---")

    df = testset.to_pandas()
    df.to_csv(OUTPUT_CSV_FILE, index=False)

    print(f"✅ Successfully generated {len(df)} Q&A pairs.")
    print(f"Dataset saved to: {OUTPUT_CSV_FILE}")


if __name__ == "__main__":
    generate_ragas_testset()