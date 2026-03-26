#!/bin/bash

PDF_DIR="data/raw_pdfs"
OUT_DIR="data_v2/raw_text"

mkdir -p "$OUT_DIR"

for pdf in "$PDF_DIR"/*.pdf; do
    base=$(basename "$pdf" .pdf)
    pdftotext -layout -enc UTF-8 -eol unix "$pdf" "$OUT_DIR/$base.txt"
done
