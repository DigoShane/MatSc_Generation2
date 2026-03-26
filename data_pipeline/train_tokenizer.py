# data_pipeline/train_tokenizer.py

# Import tokenizer components from the Hugging Face tokenizers library
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path       # For cross-platform file path handling
from tqdm import tqdm          # For progress bars (imported but not used in this code)

# Dynamically determine the project's base directory
# __file__ is the path to this script file -> since we run: python3 train_tokenizer.py, __file__ = "data_pipeline\train_tokenizer.py"
# .resolve() converts to absolute path -> If you ran the script from /home/user/projects/, this converts: Path("data_pipeline/train_tokenizer.py")
# .parent gets the parent directory (data_pipeline/)
# .parent again gets the project root (MatSc_Gen)
BASE_DIR = Path(__file__).resolve().parent.parent
#BASE_DIR -> Contains path to MatSc_Gen folder.

# Directory containing cleaned text files (input data for training tokenizer)
clean_dir = BASE_DIR / "data_v2/cleaned_text"

# Directory where the trained tokenizer will be saved
output_dir = BASE_DIR / "tokenizer"
output_dir.mkdir(exist_ok=True)  # Create directory if it doesn't exist

# ============================================================================
# INITIALIZE THE TOKENIZER
# ============================================================================

# Create a tokenizer using the BPE (Byte-Pair Encoding) algorithm
# It learns subword units by iteratively merging frequent character pairs
# model.BPE() pick Byte pair encoding as the algorithm.
tokenizer = Tokenizer(models.BPE())

# Set the pre-tokenizer to ByteLevel
# Pre-tokenizer runs BEFORE the main BPE algorithm
# ByteLevel means:
#   - Text is first converted to bytes (handles any Unicode character)
#   - Each byte becomes a token initially
#   - Then BPE merges frequent byte sequences into subwords
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

# ============================================================================
# CONFIGURE THE TRAINER
# ============================================================================

# Create a trainer object that specifies HOW to train the BPE tokenizer
trainer = trainers.BpeTrainer(
    # vocab_size: size of final vocabulary (number of unique tokens in the final vocabulary)
    # Larger vocab = more precise tokenization, but more memory usage
    vocab_size=50000,
    # min_frequency: Minimum times a subword must appear to be included
    min_frequency=2,
    # special_tokens: Reserved tokens with special meanings
    # These are added to the vocabulary and NOT learned from data
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=[
        "<s>",      # Start of sequence/sentence (marks beginning of text)
        "<pad>",    # Padding token (used to make sequences same length in batches)
        "</s>",     # End of sequence/sentence (marks end of text)
        "<unk>",    # Unknown token (for characters/words not in vocabulary)
        "<mask>"    # Mask token (used for masked language modeling like BERT)
    ],
)

# ============================================================================
# PREPARE DATA AND TRAIN
# ============================================================================

# Collect all text file paths from the cleaned_text directory
# Convert Path objects to strings (required by tokenizer.train())
# This creates a list like: ["/path/to/file1.txt", "/path/to/file2.txt", ...]
files = [str(f) for f in clean_dir.glob("*.txt")]

# Print how many files will be used for training
print(f"Training tokenizer on {len(files)} files...")
# Example output: "Training tokenizer on 6847 files..."

# TRAIN THE TOKENIZER
# This is where the actual learning happens:
# 1. Reads all the text files
# 2. Starts with individual bytes as initial tokens
# 3. Iteratively merges the most frequent byte pairs
# 4. Repeats until vocabulary reaches 32,000 tokens
# 5. Learns which subwords are most common in materials science text
# This process can take several minutes to hours depending on data size
tokenizer.train(files, trainer)

# ============================================================================
# SAVE THE TRAINED TOKENIZER
# ============================================================================

# Save the trained tokenizer to a JSON file
# The file contains:
#   - Vocabulary (all 32,000 tokens)
#   - Merge rules (which byte pairs to combine)
#   - Special token mappings
# This file can be loaded later to tokenize new text
tokenizer.save(str(output_dir / "tokenizer.json"))

print("Tokenizer training complete.")
# The tokenizer is now ready to use for encoding text into tokens
