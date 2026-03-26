#!/bin/bash

PDF_DIR="data/raw_pdfs"
OUT_DIR="data_v2/raw_text"

mkdir -p "$OUT_DIR"

for pdf in "$PDF_DIR"/*.pdf; do
    base=$(basename "$pdf" .pdf)
    mutool draw -F txt -o "$OUT_DIR/$base.txt" "$pdf" 2>/dev/null
done
