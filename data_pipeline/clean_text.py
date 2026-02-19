# data_pipeline/clean_text.py

import re                    # Regular expressions for pattern matching and text manipulation
from pathlib import Path     # For cross-platform file path handling
from tqdm import tqdm        # For displaying progress bar during batch processing

# Directory containing raw text extracted from PDFs (from pdf_to_text.py)
raw_dir = Path("../data/raw_text")

# Directory where cleaned/processed text files will be saved
clean_dir = Path("../data/cleaned_text")
clean_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

def clean_text(text):
    """
    Cleans and preprocesses raw text extracted from academic papers.
    Removes non-essential content that would add noise during LLM training.
    
    Args:
        text: str - Raw text extracted from PDF
    
    Returns:
        str - Cleaned text suitable for language model training
    
    Cleaning Steps:
        1. Remove references/bibliography section
        2. Remove figure captions
        3. Remove equation-heavy lines
        4. Normalize whitespace
    """
    
    # STEP 1: Remove everything after "References" section
    # Academic papers typically have a "References" heading followed by bibliography
    # re.split() splits text at the word "References" (case-sensitive)
    # [0] takes only the content BEFORE "References"
    # This removes: bibliography, citations, appendices that come after references
    # \b ensures we match whole word "References" not "references" within another word
    text = re.split(r"\bReferences\b", text)[0]
    
    # STEP 2: Remove figure captions
    # Pattern matches: "Figure" followed by whitespace and one or more digits, then anything
    # Examples removed:
    #   "Figure 1: Crystal structure of the material"
    #   "Figure 23 shows the temperature dependence"
    # .* matches everything on that line after "Figure N"
    # These add noise and don't contribute meaningful text for training
    text = re.sub(r"Figure\s+\d+.*", "", text)
    
    # STEP 3: Remove equation-heavy lines
    # Pattern matches: 2 or more consecutive mathematical operators
    # [=+\-*/^] matches any of these characters: = + - * / ^
    # {2,} means "2 or more in a row"
    # Removes lines like: "E = mc^2", "a + b = c", "x^2 + y^2 = r^2"
    # Mathematical equations don't help language understanding and add noise
    # Note: This is aggressive - removes any line with consecutive operators
    text = re.sub(r"[=+\-*/^]{2,}", "", text)
    
    # STEP 4: Normalize whitespace
    # \s+ matches one or more whitespace characters (spaces, tabs, newlines)
    # Replaces with single space " "
    # This:
    #   - Removes multiple consecutive spaces
    #   - Removes tabs
    #   - Converts multiple newlines to single space
    #   - Removes line breaks within paragraphs
    # Result: entire paper becomes one continuous string with single spaces
    text = re.sub(r"\s+", " ", text)
    
    # Remove leading/trailing whitespace and return
    return text.strip()

def main():
    """
    Main function that orchestrates the text cleaning pipeline.
    Processes all raw text files, cleans them, and filters by length.
    """
    
    # Get list of all text files in raw_dir
    # tqdm shows progress bar: [=====>  ] 2345/9000
    for file in tqdm(list(raw_dir.glob("*.txt"))):
        
        # Read the raw text file
        # errors="ignore" skips any characters that can't be decoded as UTF-8
        # This handles corrupted characters from PDF extraction
        with open(file, "r", errors="ignore") as f:
            text = f.read()
        
        # Apply all cleaning steps
        cleaned = clean_text(text)
        
        # QUALITY FILTER: Discard papers that are too short after cleaning
        # len(cleaned.split()) counts the number of words
        # 3000 words is roughly 6-8 pages of text
        # Papers shorter than this are likely:
        #   - Abstracts only (references section was most of the content)
        #   - Posters or extended abstracts
        #   - Failed PDF extractions
        #   - Papers where cleaning removed too much content
        # These provide insufficient training data, so we skip them
        if len(cleaned.split()) < 3000:
            continue  # Don't save this file, move to next
        
        # Save cleaned text to the clean_dir
        # Uses same filename as original (e.g., "2015.12345.txt")
        # "w" mode overwrites if file exists
        with open(clean_dir / file.name, "w") as f:
            f.write(cleaned)

# Standard Python idiom - only run main() if script is executed directly
if __name__ == "__main__":
    main()
