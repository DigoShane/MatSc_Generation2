# data_pipeline/scrape_metadata.py

import requests      # For making HTTP requests to the arXiv API
import feedparser    # For parsing the Atom/RSS feed returned by arXiv
import json          # For saving metadata to JSON file
import time          # For adding delays between API requests
from pathlib import Path  # For cross-platform file path handling

# arXiv API base URL for querying papers
BASE_URL = "http://export.arxiv.org/api/query?"

# Materials science related categories to search
# cond-mat.mtrl-sci = Condensed Matter - Materials Science
# cond-mat.supr-con = Condensed Matter - Superconductivity  
# physics.app-ph = Applied Physics
CATEGORIES = [
    "cond-mat.mtrl-sci",
    "cond-mat.supr-con",
    "physics.app-ph"
]

# Only fetch papers submitted from 2015 onwards
START_YEAR = 2015

# Maximum number of papers to fetch per category
MAX_RESULTS = 3000

# Number of papers to fetch per API request (arXiv recommends <= 100)
BATCH_SIZE = 100

# Create directory to store the metadata JSON file
metadata_dir = Path("../data/metadata")
metadata_dir.mkdir(parents=True, exist_ok=True)  # Create if doesn't exist

def build_query(category):
    """
    Constructs an arXiv API query string for a specific category.
    
    Args:
        category: arXiv category code (e.g., "cond-mat.mtrl-sci")
    
    Returns:
        Query string filtering by category and date range
        Format: cat:CATEGORY+AND+submittedDate:[START+TO+END]
    """
    return f"cat:{category}+AND+submittedDate:[{START_YEAR}01010000+TO+202412312359]"

def fetch_batch(query, start, max_results):
    """
    Fetches a batch of papers from arXiv API.
    
    Args:
        query: The search query string
        start: Starting index for pagination (0-indexed)
        max_results: Number of results to fetch in this batch
    
    Returns:
        List of feed entries (papers) from the API response
    """
    # Construct full API URL with query parameters
    url = (
        BASE_URL
        + f"search_query={query}"
        + f"&start={start}"              # Pagination offset
        + f"&max_results={max_results}"  # Batch size
    )
    
    # Make HTTP GET request to arXiv API
    response = requests.get(url)
    
    # Parse the Atom feed XML response into Python objects
    feed = feedparser.parse(response.text)
    
    return feed.entries

def main():
    """
    Main function that orchestrates the metadata scraping process.
    Iterates through categories, fetches papers in batches, and saves to JSON.
    """
    all_entries = []  # Accumulator for all paper metadata
    
    # Loop through each materials science category
    for cat in CATEGORIES:
        print(f"Fetching category: {cat}")
        query = build_query(cat)
        
        # Fetch papers in batches (pagination)
        # range(0, 3000, 100) -> [0, 100, 200, ..., 2900]
        for start in range(0, MAX_RESULTS, BATCH_SIZE):
            entries = fetch_batch(query, start, BATCH_SIZE)
            
            # If no entries returned, we've exhausted this category
            if not entries:
                break
            
            # Extract relevant metadata from each paper
            for entry in entries:
                all_entries.append({
                    "id": entry.id,                 # arXiv ID (e.g., "2015.12345")
                    "title": entry.title,           # Paper title
                    "published": entry.published,   # Publication date
                    # Convert abstract URL to PDF URL
                    # e.g., "arxiv.org/abs/2015.12345" -> "arxiv.org/pdf/2015.12345.pdf"
                    "pdf_url": entry.link.replace("abs", "pdf") + ".pdf"
                })
            
            # Sleep 3 seconds between requests to be respectful to arXiv servers
            # arXiv recommends at least 3 seconds between requests
            time.sleep(3)
    
    # Save all collected metadata to JSON file
    with open(metadata_dir / "metadata.json", "w") as f:
        json.dump(all_entries, f, indent=2)  # indent=2 for readable formatting
    
    print(f"Saved {len(all_entries)} entries.")

# Standard Python idiom - only run main() if script is executed directly
# (not imported as a module)
if __name__ == "__main__":
    main()
