# data_pipeline/estimate_tokens.py

from pathlib import Path  # For cross-platform file path handling

# Directory containing cleaned text files (output from clean_text.py)
clean_dir = Path("../data/cleaned_text")

# Initialize counters to accumulate statistics across all files
total_words = 0   # Running total of words across all papers
file_count = 0    # Number of text files processed

# Iterate through all text files in the cleaned_text directory
# .glob("*.txt") returns all files matching the pattern 
for file in clean_dir.glob("*.txt"):
    
    # Read the entire contents of the file
    # Default mode is "r" (read text mode)
    # Assumes UTF-8 encoding
    with open(file) as f:
        # .read() read entire file into memory as a string
        # .split() splits on whitespace (spaces, tabs, newlines)
        # Returns a list of words
        # Example: "hello world test" -> ["hello", "world", "test"]
        words = f.read().split()
        
        # Add the word count from this file to the running total
        total_words += len(words)
        
        # Increment the file counter
        file_count += 1

# Print summary statistics
print(f"Files: {file_count}")
# Example output: "Files: 6847"

print(f"Total words: {total_words}")
# Example output: "Total words: 45000000"

# Estimate token count using the 1.3x multiplier
# WHY 1.3x? Tokenizers (like GPT's BPE) split words into subword units
# Common conversions:
#   - "running" might become ["run", "ning"] = 2 tokens
#   - "superconductivity" might become ["super", "conduct", "ivity"] = 3 tokens
#   - "the" stays as ["the"] = 1 token
# On average, English text has ~1.3 tokens per word
# This is an approximation - actual ratio depends on:
#   - Vocabulary of the tokenizer
#   - Technical terminology (more subwords)
#   - Language complexity
print(f"Approx tokens: {int(total_words * 1.3)}")
# Example output: "Approx tokens: 58500000"
# int() converts float to integer (removes decimals)
