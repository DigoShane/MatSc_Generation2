"""
generate.py

Inference script for trained materials GPT model.

Loads:
- Custom tokenizer
- Trained model checkpoint

Generates:
- Scientific continuation text
"""

import torch
from pathlib import Path
from transformers import GPT2LMHeadModel
from tokenizers import Tokenizer

# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent # root directory.
MODEL_DIR = BASE_DIR / "checkpoints/step_200000"  # path to trained model check point. 200000 should be the final check point.
TOKENIZER_PATH = BASE_DIR / "tokenizer/tokenizer.json" # path to tokenizer.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device selection.

# -----------------------------
# Load Tokenizer
# -----------------------------

print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

# -----------------------------
# Load Model
# -----------------------------

print("Loading model...")
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)
model.to(device) # move model to GPU ??
model.eval() # eval() mode is the opposite of train() mode

# -----------------------------
# Prompt
# -----------------------------

prompt = """
The coexistence of superconductivity and magnetism can be understood through
"""

# Encode prompt
encoding = tokenizer.encode(prompt) # Convert text string to token IDs using our custom tokenizer, eg:- "The flexoelectric" -> [1234, 5678, 9012, ...]
input_ids = torch.tensor([encoding.ids], dtype=torch.long).to(device) # [encoding.ids] wraps in list -> [[1234, 5678, 9012, ...]]. Creates batch dimension (batch_size=1). Shape: [1, prompt_length] e.g., [1, 15] for 15-token prompt.

# -----------------------------
# Generate
# -----------------------------
# Disable gradient computation (we're not training, just generating)
# This saves memory and speeds up generation
with torch.no_grad():
    output = model.generate( # This is what does the autoregressive generation of GPTs.
        input_ids, # Starting tokens (the prompt)
        max_length=400,  # max_length: Maximum total length (prompt + generated)
        temperature=0.7,     # randomness control
        top_p=0.9,#Samples till tokens whose cumulative prob is >0.9. Ex: Token prob: [0.5, 0.3, 0.1, 0.05, 0.03, 0.02], Cumulative: [0.5, 0.8, 0.9, 0.95, 0.98, 1.0]. top_p=0.9 -> sample from first 3 tokens.
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,# do_sample: Whether to sample or use greedy decoding. True: Sample from probability distribution (with temperature/top_p).
        pad_token_id=tokenizer.token_to_id("<pad>") or 0 # In most Transformer models, sentences in a batch usually have different lengths. To make them the same length (so they can be stacked into a nice rectangular tensor for batch processing on GPU), we pad the shorter sequences with a special token — usually at the end. Example: 
                                                         #Sequence 1: "The cat sat" -> tokens [464, 4937, 6253]. 
                                                         #Sequence 2: "The cat sat on the mat" -> tokens [464, 4937, 6253, 319, 262, 11331]
                                                         #After padding to length 6: Seq1: [464, 4937, 6253, <pad>, <pad>, <pad>], Seq2: [464, 4937, 6253,  319,  262, 11331]..
                                                         #The model needs to ignore those <pad> positions during: 
                                                         #Loss computation (don't penalize predictions on padding)
                                                         #Attention (don't let padding tokens influence others)
                                                         # This is controlled by two things:
                                                         # 1. pad_token_id -> The actual integer ID of the padding token in the vocabulary.
                                                         # 2. attention_mask -> A binary mask (1 = real token, 0 = padding) that tells attention to ignore padding positions.
    )
# output shape: [batch_size, generated_length]
# Example: [1, 200] for our single prompt generating 200 tokens


#-------------------------------------------------------------
# Decode output
#-------------------------------------------------------------
# Extract the generated token IDs from the output tensor
# output[0] gets the first (and only) sequence from the batch
# .tolist() converts PyTorch tensor to Python list
# Result: [1234, 5678, 9012, ..., 7890] (200 token IDs)
generated_ids = output[0].tolist()
# Decode token IDs back to text using the tokenizer
# This reverses the encoding process
# [1234, 5678, 9012, ...] → "The flexoelectric effect in two-dimensional..."
generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
generated_text = generated_text.replace("Ġ", " ")

# The decoded text includes:
#   - The original prompt (first ~15 tokens)
#   - The newly generated continuation (remaining ~185 tokens)
#------------------------------------------------------------
# Display generated text
#------------------------------------------------------------
print("\n--- GENERATED TEXT ---\n")
print(generated_text)
print("\n----------------------\n")

