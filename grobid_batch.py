# grobid_batch.py

import requests
from pathlib import Path
from tqdm import tqdm

PDF_DIR = Path("data/raw_pdfs")
OUT_DIR = Path("data_v3/grobid_xml")
OUT_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8070/api/processFulltextDocument"

for pdf in tqdm(list(PDF_DIR.glob("*.pdf"))):
    out_file = OUT_DIR / (pdf.stem + ".xml")

    if out_file.exists():
        continue  # Skip already processed

    try:
        with open(pdf, "rb") as f:
            response = requests.post(
                URL,
                files={"input": f},
                data={"consolidateHeader": "1"}
            )

        if response.status_code == 200:
            out_file.write_text(response.text, encoding="utf-8")
        else:
            print(f"Failed: {pdf.name}")

    except Exception as e:
        print(f"Error processing {pdf.name}: {e}")
