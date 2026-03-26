from pathlib import Path
from lxml import etree
import unicodedata
import re
from tqdm import tqdm

XML_DIR = Path("data_v3/grobid_xml")
OUT_DIR = Path("data_v3/cleaned_text")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ns = {"tei": "http://www.tei-c.org/ns/1.0"}

xml_files = list(XML_DIR.glob("*.xml"))
print(f"Found {len(xml_files)} XML files")

for xml_file in tqdm(xml_files):
    try:
        tree = etree.parse(str(xml_file))
        texts = []

        # Abstract
        abstract = tree.xpath("//tei:abstract//tei:p", namespaces=ns)
        for p in abstract:
            if p.text:
                texts.append(p.text)

        # Body paragraphs
        body = tree.xpath("//tei:body//tei:p", namespaces=ns)
        for p in body:
            if p.text:
                texts.append(p.text)

        full_text = "\n".join(texts)

        # Unicode normalization
        full_text = unicodedata.normalize("NFKC", full_text)

        # Remove excessive whitespace
        full_text = re.sub(r"\s+", " ", full_text)

        # Basic quality filter
        if len(full_text) > 2000:
            out_file = OUT_DIR / (xml_file.stem + ".txt")
            out_file.write_text(full_text, encoding="utf-8")

    except Exception as e:
        print("Parse error:", xml_file.name)
