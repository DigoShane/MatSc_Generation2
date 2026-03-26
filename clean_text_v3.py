import os
import re
import unicodedata
from pathlib import Path

RAW_DIR = "data_v2/raw_text"
CLEAN_DIR = "data_v2/cleaned_text"

Path(CLEAN_DIR).mkdir(parents=True, exist_ok=True)

# -----------------------------
# Utility Functions
# -----------------------------

def remove_control_characters(text):
    # Remove ASCII control chars except newline
    return ''.join(
        c for c in text
        if (c == '\n') or (32 <= ord(c) <= 126) or (ord(c) >= 160)
    )

def truncate_sections(text):
    section_patterns = [
        r'\nReferences\b',
        r'\nREFERENCES\b',
        r'\nBibliography\b',
        r'\nAcknowledgements\b',
        r'\nAcknowledgments\b'
    ]
    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            return text[:match.start()]
    return text

def remove_citations(text):
    text = re.sub(r'\[[0-9,\-\s]+\]', '', text)
    text = re.sub(r'\(Ref\.\s*[0-9]+\)', '', text)
    text = re.sub(r'Ref\.\s*[0-9]+', '', text)
    return text

def is_caption_line(line):
    caption_patterns = [
        r'^\s*FIG\.?',
        r'^\s*Figure',
        r'^\s*TABLE',
        r'^\s*Table'
    ]
    return any(re.search(p, line) for p in caption_patterns)

def is_display_equation_line(line):
    stripped = line.strip()
    if len(stripped) < 5:
        return False

    math_symbols = set("=+-*/^_{}\\()[]<>|")
    symbol_count = sum(1 for c in stripped if c in math_symbols)
    density = symbol_count / max(len(stripped), 1)

    digit_ratio = sum(c.isdigit() for c in stripped) / max(len(stripped), 1)

    if density > 0.45:
        return True

    if symbol_count > 8:
        return True

    if digit_ratio > 0.5:
        return True

    return False

def clean_text(text):
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters
    text = remove_control_characters(text)

    # Truncate at references / acknowledgements
    text = truncate_sections(text)

    # Remove citations
    text = remove_citations(text)

    cleaned_lines = []
    removed_lines = 0
    total_lines = 0

    for line in text.split("\n"):
        total_lines += 1

        if is_caption_line(line):
            removed_lines += 1
            continue

        if is_display_equation_line(line):
            removed_lines += 1
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# -----------------------------
# Run Cleaning
# -----------------------------

for filename in os.listdir(RAW_DIR):
    input_path = os.path.join(RAW_DIR, filename)

    try:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()
    except:
        continue

    cleaned = clean_text(raw)

    if cleaned and len(cleaned) > 1000:
        output_path = os.path.join(CLEAN_DIR, filename)
        with open(output_path, "w", encoding="utf-8") as out:
            out.write(cleaned)

print("Cleaning complete.")
