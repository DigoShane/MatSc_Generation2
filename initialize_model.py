# initialize_model.py

import torch
from transformers import GPT2Config, GPT2LMHeadModel
from tokenizers import Tokenizer
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Load tokenizer to get vocab size dynamically
tokenizer = Tokenizer.from_file(str(BASE_DIR / "tokenizer/tokenizer.json"))
vocab_size = tokenizer.get_vocab_size()

print("Tokenizer vocab size:", vocab_size)

# -------------------------------------------------------------------
# GPT Configuration
# -------------------------------------------------------------------
config = GPT2Config(
    vocab_size=vocab_size,     # MUST match tokenizer (50k)
    n_positions=1024,
    n_ctx=1024,
    n_embd=512,
    n_layer=8,
    n_head=8,
)

# -------------------------------------------------------------------
# Initialize model
# -------------------------------------------------------------------
model = GPT2LMHeadModel(config)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params/1e6:.2f}M")

# -------------------------------------------------------------------
# Save model
# -------------------------------------------------------------------
model_dir = BASE_DIR / "model"
model_dir.mkdir(exist_ok=True)

model.save_pretrained(model_dir)

print("Model initialized and saved.")
