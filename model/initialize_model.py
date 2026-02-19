# model/initialize_model.py

import torch  # PyTorch - deep learning framework
from transformers import GPT2Config, GPT2LMHeadModel  # Hugging Face transformers library
from pathlib import Path  # For cross-platform file path handling
from tokenizers import Tokenizer  # For loading the custom tokenizer we trained

# BASE_DIR = MatSc_Generation
BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================================
# LOAD THE CUSTOM TOKENIZER
# ============================================================================

# Path to the tokenizer we trained earlier (from train_tokenizer.py)
tokenizer_path = BASE_DIR / "tokenizer/tokenizer.json"

# Load the trained tokenizer from the JSON file
tokenizer = Tokenizer.from_file(str(tokenizer_path))

# Get the vocabulary size (should be 32,000 from our training)
# This tells us how many unique tokens the tokenizer knows
vocab_size = tokenizer.get_vocab_size()

print(f"Tokenizer vocab size: {vocab_size}")
# Expected output: "Tokenizer vocab size: 32000"

# ============================================================================
# DEFINE MODEL ARCHITECTURE (GPT-2 Configuration)
# ============================================================================

# GPT2Config defines the architecture/shape of the neural network
# Think of this as the "blueprint" for the model
config = GPT2Config(
    # vocab_size: Must match tokenizer vocabulary size
    # This determines the size of the embedding layer
    # The model needs one embedding vector for each possible token
    vocab_size=vocab_size,  # 32,000 (from our custom tokenizer)
    
    # n_positions: Maximum sequence length the model can handle
    # Also called "context window" or "max sequence length"
    # 512 tokens ≈ 400 words ≈ 1-2 paragraphs
    # GPT-2 original used 1024, GPT-3 used up to 2048, GPT-4 uses 8192+
    # We use 512 to keep memory usage reasonable during training
    n_positions=512,
    
    # n_ctx: Context size (same as n_positions in GPT-2)
    # This is the attention context window
    # Must be same as n_positions for GPT-2 architecture
    n_ctx=512,
    
    # n_embd: Embedding dimension. 
    # This is the dimension of the space of vector representation of each token (word/subword)
    # Comparison: GPT-2 medium=1024, GPT-2 large=1280, GPT-2 XL=1600
    n_embd=768,
    
    # n_layer: Number of transformer layers (depth of the network)
    # More layers = more sophisticated understanding, but slower training
    # 12 is GPT-2 "small" size
    n_layer=12,
    
    # n_head: Number of attention heads per layer
    # Attention heads allow the model to focus on different aspects simultaneously
    # For example: one head focuses on syntax, another on semantics
    # Must evenly divide n_embd (768 / 12 = 64 dimensions per head)
    # 12 is standard for GPT-2 small
    n_head=12,
    
    # resid_pdrop: Residual connection dropout probability
    # Dropout randomly sets some activations to zero during training
    # This prevents overfitting (model memorizing training data)
    # 0.1 = 10% of residual connections are randomly dropped
    # Acts as regularization
    resid_pdrop=0.1,
    
    # embd_pdrop: Embedding dropout probability
    # Dropout applied to the token embeddings
    # 0.1 = 10% of embedding values randomly zeroed during training
    # Helps model be robust to missing information
    embd_pdrop=0.1,
    
    # attn_pdrop: Attention dropout probability
    # Dropout applied to attention weights
    # 0.1 = 10% of attention connections randomly dropped
    # Prevents model from over-relying on specific token relationships
    attn_pdrop=0.1,
)

# ============================================================================
# INITIALIZE MODEL FROM SCRATCH
# ============================================================================

# Create a GPT-2 model with a language modeling head
# "LMHead" = Language Modeling Head (predicts next token)
# This initializes all weights RANDOMLY (not pre-trained)
# The model knows NOTHING yet - it's like a newborn brain
model = GPT2LMHeadModel(config)

# At this point:
# - Model has the correct architecture (12 layers, 768 dimensions, etc.)
# - All weights are random (Xavier/Kaiming initialization)
# - Model cannot generate coherent text yet (needs training)

# ============================================================================
# MOVE MODEL TO GPU (if available)
# ============================================================================

# Check if CUDA (NVIDIA GPU) is available
# Training on GPU is 10-100x faster than CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the device (GPU or CPU)
# This transfers all model parameters to GPU memory (if available)
model.to(device)

# ============================================================================
# CALCULATE AND DISPLAY MODEL SIZE
# ============================================================================

# Count total number of trainable parameters
# p.numel() returns number of elements in each parameter tensor
# sum() adds up parameters from all layers
# This tells us how "big" the model is
total_params = sum(p.numel() for p in model.parameters())

# Convert to millions and print
# 1e6 = 1,000,000
# .2f = format with 2 decimal places
print(f"Total parameters: {total_params / 1e6:.2f}M")

# ============================================================================
# PARAMETER COUNT BREAKDOWN (for understanding)
# ============================================================================
# Where do the ~124M parameters come from?
#
# 1. Token Embeddings: vocab_size × n_embd = 32,000 × 768 ≈ 24.6M
# 2. Position Embeddings: n_positions × n_embd = 512 × 768 ≈ 0.4M
# 3. Transformer Layers (×12 layers):
#    - Attention: 4 × (n_embd × n_embd) per layer ≈ 2.4M per layer
#    - Feed-Forward: 2 × (n_embd × 4×n_embd) per layer ≈ 4.7M per layer
#    - Total per layer: ~7.1M
#    - 12 layers: 7.1M × 12 ≈ 85M
# 4. Layer Norms and biases: ~0.5M
# 5. LM Head (output layer): vocab_size × n_embd = 32,000 × 768 ≈ 24.6M
#
# Total: 24.6 + 0.4 + 85 + 0.5 + 24.6 ≈ 135M
# (Actual may vary slightly due to weight sharing and implementation details)

# ============================================================================
# SAVE MODEL CONFIGURATION AND WEIGHTS
# ============================================================================

# Save the configuration (architecture blueprint) as config.json
# This file stores all the hyperparameters we defined above
# Can be loaded later to recreate the exact same architecture
config.save_pretrained(BASE_DIR / "model")
# Creates: Materials-LLM-Pretraining/model/config.json

# Save the model weights (parameters) 
# This saves the UNTRAINED model (random weights)
# Files created:
# - pytorch_model.bin (the actual weight values)
# - config.json (already saved above, gets overwritten with same content)
model.save_pretrained(BASE_DIR / "model")
# Creates: Materials-LLM-Pretraining/model/pytorch_model.bin

print("Model initialized and saved.")

# ============================================================================
# WHAT WE HAVE NOW
# ============================================================================
# 
# At this point, we have:
# 1. A custom tokenizer trained on materials science papers (32k vocab)
# 2. A GPT-2 architecture model configured for our needs
# 3. Random initialization (model doesn't know anything yet)
# 4. Model files saved to disk
#
# NEXT STEPS:
# 1. Create a training script to train this model on our cleaned text
# 2. The model will learn patterns from materials science papers
# 3. After training, it can generate scientific text and understand domain concepts
#
# ============================================================================

#=================================================================================
# Test forward pass
input_ids = torch.randint(0, vocab_size, (1, 512)).to(device)
outputs = model(input_ids)
print("Forward pass successful.")
#=================================================================================

