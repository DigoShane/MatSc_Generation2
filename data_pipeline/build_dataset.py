# data_pipeline/build_dataset.py (assumed filename)

import numpy as np           # For efficient numerical array operations
from pathlib import Path     # For cross-platform file path handling
from tokenizers import Tokenizer  # To load the custom tokenizer
from tqdm import tqdm        # For progress bars during processing

# Dynamically find the project root directory
# __file__ = this script
# .parent.parent = go up two levels to project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory containing cleaned text files (output from clean_text.py)
clean_dir = BASE_DIR / "data/cleaned_text"

# Path to the trained tokenizer (output from train_tokenizer.py)
tokenizer_path = BASE_DIR / "tokenizer/tokenizer.json"

# Directory where the final binary dataset files will be saved
output_dir = BASE_DIR / "data"
output_dir.mkdir(exist_ok=True)  # Create if doesn't exist (though it should already)

# ============================================================================
# LOAD THE TOKENIZER
# ============================================================================

print("Loading tokenizer...")
# Load the custom tokenizer we trained on materials science text
# This tokenizer knows our 32,000 token vocabulary
tokenizer = Tokenizer.from_file(str(tokenizer_path))

# ============================================================================
# FIND ALL TEXT FILES TO PROCESS
# ============================================================================

# Get a list of all cleaned text files
# These are the papers that passed the 3000+ word filter from clean_text.py
files = list(clean_dir.glob("*.txt"))

print(f"Found {len(files)} cleaned files.")
# Expected: ~6,000-7,000 files (papers that met quality criteria)

# ============================================================================
# TOKENIZE ALL TEXT FILES
# ============================================================================

# Initialize an empty list to accumulate ALL tokens from ALL papers
# This will become one giant sequence of tokens
all_tokens = []

# Process each file with a progress bar. Thats the job of tqdm.
for file in tqdm(files):
    # Open and read the entire file
    # encoding="utf-8" handles Unicode characters
    # errors="ignore" skips characters that can't be decoded
    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()  # Read all text and remove leading/trailing whitespace
        
        # Skip empty files (shouldn't happen after clean_text.py, but safety check)
        if len(text) == 0:
            continue
        
        # TOKENIZE: Convert text string to token IDs
        # Example: "Superconductivity in YBCO" → [1234, 5678, 90, 1122]
        encoding = tokenizer.encode(text)
        
        # encoding.ids is a list of integer token IDs
        # Extend adds all elements from this list to all_tokens
        # This concatenates all papers into one continuous token stream
        all_tokens.extend(encoding.ids)

# At this point, all_tokens is a Python list containing millions of token IDs
# Example: [1234, 5678, 90, 1122, 3344, ..., 9876] (millions of integers)

print("Total tokens collected:", len(all_tokens))
# Expected: ~50-60 million tokens (based on earlier estimate_tokens.py)

# ============================================================================
# CONVERT TO NUMPY ARRAY
# ============================================================================

# Convert Python list to numpy array for efficient storage and processing
# dtype=np.uint16 means:
#   - unsigned integer (no negative numbers)
#   - 16 bits = can represent 0 to 65,535
#   - Our vocab_size is 32,000, so uint16 is perfect (saves memory vs uint32)
#   - Each token ID takes only 2 bytes instead of 4 or 8
all_tokens = np.array(all_tokens, dtype=np.uint16)

# Memory savings example:
# 60 million tokens:
#   - uint16: 60M × 2 bytes = 120 MB
#   - uint32: 60M × 4 bytes = 240 MB
#   - uint64: 60M × 8 bytes = 480 MB
# Using uint16 saves 50-75% memory!

# ============================================================================
# SPLIT INTO TRAIN AND VALIDATION SETS
# ============================================================================

# Calculate the split point for 90% train, 10% validation
# Example: If we have 60 million tokens:
#   split_idx = 0.9 × 60,000,000 = 54,000,000
split_idx = int(0.9 * len(all_tokens))

# Split the array into training and validation portions
# Train: first 90% of tokens (indices 0 to split_idx-1)
train_tokens = all_tokens[:split_idx]

# Validation: last 10% of tokens (indices split_idx to end)
val_tokens = all_tokens[split_idx:]

# Print the sizes
print("Train tokens:", len(train_tokens))
# Example: "Train tokens: 54000000" (54M tokens)

print("Val tokens:", len(val_tokens))
# Example: "Val tokens: 6000000" (6M tokens)

# ============================================================================
# SAVE AS BINARY FILES
# ============================================================================

# Save training tokens to binary file
# .tofile() writes the numpy array as raw binary data
# No formatting, no headers - just the raw uint16 values
# This is the most memory-efficient storage format
# File size: len(train_tokens) × 2 bytes
train_tokens.tofile(output_dir / "train.bin")
# Creates: Materials-LLM-Pretraining/data/train.bin (~108 MB for 54M tokens)

# Save validation tokens to binary file
val_tokens.tofile(output_dir / "val.bin")
# Creates: Materials-LLM-Pretraining/data/val.bin (~12 MB for 6M tokens)

print("Dataset build complete.")

# ============================================================================
# WHAT WE HAVE NOW
# ============================================================================
#
# Two binary files ready for training:
# 
# train.bin:
#   - Contains 90% of all tokens (e.g., 54M tokens)
#   - Used to train the model (update weights)
#   - Raw binary format (uint16)
#   - Size: ~108 MB
#
# val.bin:
#   - Contains 10% of all tokens (e.g., 6M tokens)
#   - Used to evaluate model during training (measure loss)
#   - Raw binary format (uint16)
#   - Size: ~12 MB
#
# These files are ready to be loaded by the training script!
#
# ============================================================================
