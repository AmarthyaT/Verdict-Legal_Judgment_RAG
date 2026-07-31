import os
import re
from pypdf import PdfReader
import logging
from typing import Optional
import json
import sys

# --- Configuration ---
# Set the path to the folder containing your 1000+ PDF files.
# PLEASE UPDATE THIS PATH with the actual location of your judgment files.
# Example: r"C:\Users\tamar\Desktop\judgments - Copy"
PDF_FOLDER_PATH = r"C:/Users/tamar/Desktop/judgments - Copy"

# --- Logging Setup ---
# Log to a file for error tracking.
logging.basicConfig(filename='extraction_errors.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w') # 'w' to overwrite on each run

print(f"Starting text extraction from PDF files in: {PDF_FOLDER_PATH}")
print(f"Results will be saved to: extracted_first_pages.json")
print(f"Errors (e.g., corrupt or scanned PDFs) will be logged to: extraction_errors.txt")
# --------------------

def extract_first_page_content(pdf_path: str) -> Optional[str]:
    """
  Extracts the raw text content from the first page of a single PDF file.
    """
    try:
        reader = PdfReader(pdf_path)
        if len(reader.pages) == 0:
            logging.error(f"PDF has no pages.", exc_info=False)
            return None
        
        # Extract text from the first page only (index 0)
        first_page_text = reader.pages[0].extract_text()
        
        if not first_page_text or first_page_text.strip() == "":
            logging.error(f"Failed to extract any meaningful text from the first page. Likely a scanned image or corrupt file.", exc_info=False)
            return "--- TEXT EXTRACTION FAILED: LIKELY SCANNED OR NON-TEXT PDF ---"

        return first_page_text.strip()

    except Exception as e:
        # Log generic extraction errors
        logging.error(f"Unexpected error during PDF processing: {e}", exc_info=False)
        return f"--- UNEXPECTED ERROR: {str(e)} ---"

def batch_extract_content(folder_path: str):
    """
    Iterates through all PDFs and extracts the content of the first page 
    #into a JSON structure.""""""
    """
    if not os.path.isdir(folder_path):
        print(f"Error: Folder '{folder_path}' not found. Please check PDF_FOLDER_PATH configuration.")
        sys.exit(1)

    extracted_data = {}
    file_count = 0
    successful_extractions = 0
    
    # Iterate over files in the source folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_count += 1
            full_path = os.path.join(folder_path, filename)
            
            # Extract the raw content
            content = extract_first_page_content(full_path)
            
            if content and not content.startswith("---"):
                successful_extractions += 1
                
            # Store the data using the original filename as the key
            extracted_data[filename] = content
            
            if file_count % 100 == 0:
                print(f"Processed {file_count} files...")

    # Write the results to a JSON file
    output_filename = "extracted_first_pages.json"
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, ensure_ascii=False, indent=4)
        print(f"\n--- Extraction Complete ---")
        print(f"Total PDFs found: {file_count}")
        print(f"Files with extractable text: {successful_extractions}")
        print(f"Review the file '{output_filename}' to analyze the content and find a new extraction pattern.")
    except Exception as e:
        print(f"CRITICAL ERROR: Could not write JSON file: {e}")
        
if __name__ == "__main__":
    batch_extract_content(PDF_FOLDER_PATH)