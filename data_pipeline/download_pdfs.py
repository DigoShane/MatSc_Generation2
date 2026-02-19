# data_pipeline/download_pdfs.py

import json          # For reading the metadata JSON file
import requests      # For downloading PDF files via HTTP
import time          # For adding delays between downloads
from pathlib import Path   # For cross-platform file path handling
from tqdm import tqdm      # For displaying a progress bar during downloads

# Path to the metadata file created by scrape_metadata.py
metadata_path = Path("../data/metadata/metadata.json")

# Directory where downloaded PDFs will be saved
pdf_dir = Path("../data/raw_pdfs")
pdf_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

def download_pdf(url, save_path):
    """
    Downloads a single PDF file from a URL and saves it to disk.
    
    Args:
        url: The PDF download URL (from arXiv)
        save_path: Path object where the PDF should be saved
    
    Returns:
        None - writes file to disk or prints error message
    """
    try:
        # Make HTTP GET request to download the PDF
        # timeout=30 prevents hanging on slow/dead connections
        response = requests.get(url, timeout=30)
        
        # Check if download was successful (HTTP 200 OK)
        if response.status_code == 200:
            # Write binary content to file
            # "wb" mode = write binary (PDFs are binary files, not text)
            with open(save_path, "wb") as f:
                f.write(response.content)
    
    # Catch any errors (network issues, timeouts, etc.)
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    """
    Main function that orchestrates the PDF download process.
    Reads metadata, downloads each PDF, and implements rate limiting.
    """
    # Load the metadata JSON file containing all paper information
    with open(metadata_path) as f:
        entries = json.load(f)
    
    # Iterate through each paper with a progress bar
    # tqdm wraps the iterable and shows progress: [====> ] 45%
    for entry in tqdm(entries):
        # Extract paper ID from the full arXiv URL
        # e.g., "http://arxiv.org/abs/2015.12345" -> "2015.12345"
        paper_id = entry["id"].split("/")[-1]
        
        # Construct the filename for this PDF
        # e.g., "2015.12345.pdf"
        save_path = pdf_dir / f"{paper_id}.pdf"
        
        # Skip if PDF already exists (resume capability)
        # Useful if script crashes or is interrupted - won't re-download
        if save_path.exists():
            continue
        
        # Download the PDF from arXiv
        download_pdf(entry["pdf_url"], save_path)
        
        # Sleep 2 seconds between downloads to be respectful to arXiv servers
        # Prevents overwhelming the server with rapid requests
        time.sleep(2)

# Standard Python idiom - only run main() if script is executed directly
if __name__ == "__main__":
    main()
