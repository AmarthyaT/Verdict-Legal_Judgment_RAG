import chromadb
import os

# --- CONFIGURATION ---
# Make the CHROMA_PATH relative to this script and absolute so running from other CWD works
CHROMA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "chroma_store1"))
# The name of the collection you created.
COLLECTION_NAME = "supreme_court_judgments"

def _first_of(metadata, keys, default="Unknown"):
    for k in keys:
        if k in metadata and metadata[k]:
            return metadata[k]
    return default

def get_unique_case_names():
    """
    Connects to a ChromaDB database, retrieves all records, and extracts
    a unique list of case names.
    """
    # Ensure the persistent store directory exists (create if needed)
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # 1. Connect to the ChromaDB persistent client
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
    except Exception as e:
        print(f"Error: Failed to initialize ChromaDB client at '{CHROMA_PATH}': {e}")
        return

    # 2. Get or create the collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"Connected to collection '{COLLECTION_NAME}'.")
    except Exception:
        try:
            collection = client.create_collection(name=COLLECTION_NAME)
            print(f"Created new collection: '{COLLECTION_NAME}'.")
        except Exception as e:
            print(f"Error: Failed to get or create collection '{COLLECTION_NAME}': {e}")
            return

    # 3. Get the total number of items to fetch all of them
    try:
        total_docs = collection.count()
    except Exception as e:
        print(f"Error: Could not get collection count: {e}")
        return

    if total_docs == 0:
        print("The collection is empty. No case names to display.")
        return

    print(f"Found {total_docs} total document chunks in the database. Fetching metadata...")

    # 4. Retrieve all metadata from the collection
    try:
        retrieved_data = collection.get(
            limit=total_docs,
            include=['metadatas']
        )
    except Exception as e:
        print(f"Error fetching metadata from collection: {e}")
        return

    # 5. Extract and store unique case names
    unique_case_names = set()
    petitioner_keys = ['pet', 'petitioner', 'Petitioner', 'plaintiff']
    respondent_keys = ['res', 'respondent', 'Respondent', 'defendant']

    for metadata in retrieved_data.get('metadatas', []):
        if not metadata:
            continue
        # metadata might be a dict; if it's a list or other structure, skip gracefully
        if not isinstance(metadata, dict):
            continue

        petitioner = _first_of(metadata, petitioner_keys, default="Unknown Petitioner")
        respondent = _first_of(metadata, respondent_keys, default="Unknown Respondent")

        # Construct a standardized case name
        case_name = f"{petitioner} vs. {respondent}"
        unique_case_names.add(case_name)

    # 6. Print the results and write them to a text file
    total_unique = len(unique_case_names)
    print(f"\n--- Found {total_unique} Unique Case Names ---")
    for i, name in enumerate(sorted(unique_case_names), 1):
        print(f"{i}. {name}")

    # Write unique case names to a text file next to this script
    output_path = os.path.join(os.path.dirname(__file__), "unique_case_names.txt")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for name in sorted(unique_case_names):
                f.write(name + "\n")
        print(f"\nSaved unique case names to: {output_path}")
    except Exception as e:
        print(f"Error writing to file '{output_path}': {e}")

if __name__ == "__main__":
    get_unique_case_names()