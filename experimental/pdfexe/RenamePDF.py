import os
import re
from typing import Optional, Dict, Any
import json
import logging
import sys

# --- Configuration ---
# 1. Path to the input JSON file created in the previous step (containing the raw text).
# Assumes it's in the same directory as this script.
INPUT_JSON_FILE = 'extracted_first_pages.json'

# 2. Name of the output file that will contain the clean, structured data.
OUTPUT_JSON_FILE = 'structured_judgments.json'

# --- Logging Setup ---
LOG_FILE = 'processing_errors.txt'
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='w') 

print(f"Starting structured data extraction from: {INPUT_JSON_FILE}")
print(f"Clean results will be saved to: {OUTPUT_JSON_FILE}")
print(f"Errors will be logged to: {LOG_FILE}")
# --------------------

def clean_name_for_output(name: str) -> str:
    """Cleans up names by removing excessive spacing and illegal characters."""
    # Remove sequences of whitespace, hyphens, and underscores with a single space
    name = re.sub(r'[\s\-_]+', ' ', name).strip()
    return name

def extract_case_details(pdf_text: str) -> Optional[Dict[str, str]]:
    """
    Extracts Petitioner, Respondent, and Year using highly targeted regex 
    anchored by specific keywords found in the SC judgment format.
    """
    
    # 1. Clean up excessive whitespace/newlines in the text to make regex easier
    cleaned_text = re.sub(r'[\n\r\t]+', ' ', pdf_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Define a search area near the beginning where the titles are
    search_area = cleaned_text[:3500] 

    petitioner = None
    respondent = None
    year = None
    
    # --- STEP 1: Extract Petitioner and Respondent ---
    # Pattern: PETITIONER: (Capture Group 1: Petitioner Name) Vs./RESPONDENT: (Capture Group 2: Respondent Name)
    # The (?:Vs\.|RESPONDENT:) handles the messy text between the two names.
    case_pattern = re.compile(
        r'PETITIONER:\s*(.*?)\s*(?:Vs\.|RESPONDENT:)\s*RESPONDENT:\s*(.*?)\s*DATE OF JUDGMENT', 
        re.IGNORECASE | re.DOTALL
    )
    
    match = case_pattern.search(search_area)
    
    if match:
        # Group 1 is the Petitioner's name (before "Vs." or "RESPONDENT:")
        # We strip internal legal phrases like [IN T.C.]
        petitioner = match.group(1).split('[IN T.C.')[0].strip() 
        
        # Group 2 is the Respondent's raw name (after "RESPONDENT:")
        respondent_raw = match.group(2).strip()
        
        # We stop the respondent capture at the next major field anchor (DATE OF JUDGMENT or BENCH)
        respondent = respondent_raw.split('DATE OF JUDGMENT')[0].split('BENCH')[0].strip()
    else:
        return None # Failed to find the core P/R structure

    # --- STEP 2: Extract Year ---
    # Pattern: DATE OF JUDGMENT followed by anything, then a 4-digit year.
    year_pattern = re.compile(r'DATE OF JUDGMENT.*?(\d{4})', re.IGNORECASE)
    year_match = year_pattern.search(search_area)
    
    if year_match:
        year = year_match.group(1).strip()
    else:
        return None # Failed to find the year

    # Final construction of clean data dictionary
    if petitioner and respondent and year:
        # If the respondent field is empty in the PDF, use a placeholder
        if not respondent:
            respondent = "State/Union" 
            
        return {
            "Petitioner": clean_name_for_output(petitioner),
            "Respondent": clean_name_for_output(respondent),
            "Year": year
        }
    
    return None

def process_json_data():
    """Reads the raw JSON, processes the text, and writes clean data to a new JSON file."""
    try:
        with open(INPUT_JSON_FILE, 'r', encoding='utf-8') as f:
            raw_data: Dict[str, str] = json.load(f)
    except FileNotFoundError:
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' not found. Did you run the previous extraction step?")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"CRITICAL ERROR: Input file '{INPUT_JSON_FILE}' is corrupted or not valid JSON.")
        sys.exit(1)
        
    structured_data: Dict[str, Any] = {}
    total_count = 0
    success_count = 0

    for filename, raw_content in raw_data.items():
        total_count += 1
        
        # Skip entries that failed text extraction in the previous step
        if raw_content.startswith("---"):
            logging.error(f"Skipped '{filename}': Text content was marked as failed/error in input JSON.")
            structured_data[filename] = {"Status": "Extraction Failed in Original Step"}
            continue
            
        # Apply the targeted extraction logic
        details = extract_case_details(raw_content)
        
        if details:
            structured_data[filename] = details
            success_count += 1
        else:
            logging.warning(f"Failed to match pattern for '{filename}'. Manual extraction required.")
            structured_data[filename] = {"Status": "Pattern Match Failed"}

        if total_count % 100 == 0:
            print(f"Processed {total_count} records...")

    # Write the results to the output JSON file
    try:
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(structured_data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not write output JSON file: {e}")
        
    print(f"\n--- Processing Summary ---")
    print(f"Total records processed: {total_count}")
    print(f"Successfully structured: {success_count}")
    print(f"Failed pattern matches: {total_count - success_count}")
    print(f"Review '{OUTPUT_JSON_FILE}' for the clean data and '{LOG_FILE}' for failed items.")


if __name__ == "__main__":
    process_json_data() 
    
    
    
    
    
    
    
    
    
    
