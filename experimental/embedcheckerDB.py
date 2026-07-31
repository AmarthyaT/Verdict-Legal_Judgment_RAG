import random
from typing import List, Dict, Any

# --- SIMULATED DATA (Replace with your actual MockVectorDB instance if running live) ---

class SimulatedVectorDB:
    """
    A class that simulates the state of the MockVectorDB after a successful 
    embedding run. It contains placeholder data for verification checks.
    """
    def __init__(self):
        # Simulate storing 10 Parent Documents (Headnotes)
        self.parent_vectors: Dict[str, Any] = self._create_parent_data(10) 
        # Simulate storing 250 Child Chunks (average 25 per document)
        self.child_vectors: Dict[str, Any] = self._create_child_data(250)

    def _create_parent_data(self, count: int) -> Dict[str, Any]:
        data = {}
        for i in range(count):
            doc_id = f"DOCID_{100+i}"
            data[doc_id] = {
                'vector': [random.uniform(-0.1, 0.1) for _ in range(768)], # 768 dim vector
                'metadata': {
                    "SourceID": doc_id,
                    "CaseTitle": f"Case of the State v. Respondent {i}",
                    "VerdictDate": "2024-01-01",
                    "Metadata_Keys": ["SourceID", "CaseTitle", "VerdictDate", "Bench"]
                }
            }
        return data

    def _create_child_data(self, count: int) -> Dict[str, Any]:
        data = {}
        # Use existing parent IDs for linking
        parent_ids = list(self.parent_vectors.keys()) 
        for i in range(count):
            parent_id = random.choice(parent_ids)
            chunk_id = f"{parent_id}_C{i}"
            data[chunk_id] = {
                'vector': [random.uniform(-0.1, 0.1) for _ in range(768)],
                'metadata': {
                    "ParentID": parent_id,
                    "TextExcerpt": f"The court observed in paragraph {i} that..."
                }
            }
        return data

    # ----------------- VERIFICATION METHODS -----------------

    def check_embedding_status(self) -> Dict[str, int]:
        """
        [VERIFICATION 1] Returns the total count of documents and chunks.
        This is the quickest way to see if the embedding job was complete.
        """
        return {
            "total_parent_documents": len(self.parent_vectors),
            "total_child_chunks": len(self.child_vectors)
        }

    def inspect_sample_embeddings(self, count: int = 1) -> List[Dict[str, Any]]:
        """
        [VERIFICATION 2] Inspects a sample of the stored data for integrity (Vector Dim and Metadata).
        """
        samples = []
        
        # 1. Inspect a sample Parent Document (Headnote)
        parent_keys = list(self.parent_vectors.keys())
        if parent_keys:
            doc_id = random.choice(parent_keys)
            data = self.parent_vectors[doc_id]
            samples.append({
                "Type": "Parent (Headnote)",
                "ID": doc_id,
                "Title": data['metadata'].get('CaseTitle', 'N/A'),
                "Vector_Dim": len(data['vector']) if data['vector'] else 0,
                "Metadata_Keys": list(data['metadata'].keys())
            })
            
        # 2. Inspect a sample Child Chunk (Paragraph)
        child_keys = list(self.child_vectors.keys())
        if child_keys:
            chunk_id = random.choice(child_keys)
            data = self.child_vectors[chunk_id]
            samples.append({
                "Type": "Child (Chunk)",
                "ID": chunk_id,
                "Parent_ID": data['metadata'].get('ParentID', 'N/A'),
                "Vector_Dim": len(data['vector']) if data['vector'] else 0,
                "Text_Excerpt": data['metadata'].get('TextExcerpt', 'N/A')
            })
        
        return samples

# --------------------------------------------------------------------------
# EXECUTION BLOCK
# --------------------------------------------------------------------------

if __name__ == '__main__':
    # Initialize the simulated database (pretends the embedding job already ran)
    embedded_db = SimulatedVectorDB()

    # --- 1. CHECK TOTAL COUNTS ---
    print("\n--- EMBEDDING STATUS CHECK (Verification 1: Total Counts) ---")
    status = embedded_db.check_embedding_status()
    print(f"Total Documents Successfully Embedded (Parents): {status['total_parent_documents']}")
    print(f"Total Paragraph Chunks Stored (Children): {status['total_child_chunks']}")
    
    # --- 2. QUALITY CONTROL INSPECTION ---
    print("\n--- SAMPLE QUALITY CONTROL (Verification 2: Data Integrity) ---")
    
    # Run the inspection on 1 random Parent and 1 random Child
    sample_data = embedded_db.inspect_sample_embeddings(count=1) 
    
    if sample_data:
        print("Inspection of a random Parent/Child sample:")
        for item in sample_data:
            print("-" * 30)
            print(f"Type: {item['Type']}")
            print(f"ID: {item['ID']}")
            print(f"Vector Dimension: {item['Vector_Dim']} (Should be 768 for BGE-Base)")
            
            if item['Type'] == 'Parent (Headnote)':
                print(f"Title: {item['Title']}")
                print(f"Metadata Keys: {item['Metadata_Keys']}")
            else:
                print(f"Linked Parent: {item['Parent_ID']}")
                print(f"Text Snippet: {item['Text_Excerpt'][:50]}...")
    else:
        print("Verification failed: No documents were found in the database.")