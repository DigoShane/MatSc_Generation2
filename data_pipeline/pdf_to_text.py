# data_pipeline/pdf_to_text.py

import subprocess       # For running external command-line programs (pdftotext)
from pathlib import Path     # For cross-platform file path handling
from tqdm import tqdm        # For displaying progress bar during batch processing

# Directory containing the downloaded PDF files (from download_pdfs.py)
pdf_dir = Path("../data/raw_pdfs")

# Directory where extracted text files will be saved
text_dir = Path("../data/raw_text")
text_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

def convert_pdf(pdf_path, txt_path):
    """
    Converts a single PDF file to text using the pdftotext command-line tool.
    
    Args:
        pdf_path: Path object pointing to the source PDF file
        txt_path: Path object where the extracted text should be saved
    
    Returns:
        None - creates a text file on disk
    
    Technical Details:
        Uses the pdftotext utility (part of poppler-utils package)
        The -layout flag preserves the original page layout, including:
        - Line breaks
        - Column formatting
        - Spacing and indentation
        This is important for academic papers to maintain structure
    """
    subprocess.run([
        "pdftotext",           # Command-line tool name
        "-layout",             # Preserve original page layout/formatting
        str(pdf_path),         # Input PDF file (converted to string path)
        str(txt_path)          # Output text file (converted to string path)
    ])
    # Note: No error checking - if pdftotext fails, the script continues
    # Failed conversions will result in empty or missing text files

def main():
    """
    Main function that orchestrates batch PDF-to-text conversion.
    Processes all PDFs in the raw_pdfs directory and extracts their text.
    """
    # Get list of all PDF files in the pdf_dir
    # .glob("*.pdf") finds all files ending in .pdf
    # list() converts the generator to a list (needed for tqdm to show total count)
    # tqdm() wraps the list to show progress bar: [=====>  ] 2345/9000
    for pdf_file in tqdm(list(pdf_dir.glob("*.pdf"))):
        
        # Construct output text filename
        # pdf_file.stem gets the filename without extension
        # e.g., "2015.12345.pdf" -> "2015.12345"
        # Then append ".txt" to create "2015.12345.txt"
        txt_file = text_dir / (pdf_file.stem + ".txt")
        
        # Skip if text file already exists (resume capability)
        # Allows you to interrupt and restart without re-processing
        if txt_file.exists():
            continue
        
        # Convert the PDF to text
        convert_pdf(pdf_file, txt_file)
        
        # Note: No rate limiting or delay between conversions
        # This is a local operation, so no need to be polite to servers

# Standard Python idiom - only run main() if script is executed directly
if __name__ == "__main__":
    main()
