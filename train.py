"""
train.py

Pretraining script for 110M GPT model on materials corpus.

Features:
- Memory-mapped dataset (fast, low RAM)
- Mixed precision training (fp16)
- Gradient accumulation
- Cosine learning rate schedule with warmup
- Validation perplexity tracking
- Periodic checkpoint saving

GPU: RTX 3090 (24GB)
"""

import math       # For calculating perplexity (exp function)
import time       # For tracking training time
import torch      # PyTorch deep learning framework
import numpy as np  # For loading binary data files
from pathlib import Path  # For file path handling
from torch.utils.data import Dataset, DataLoader  # For batch loading
from transformers import GPT2LMHeadModel, get_cosine_schedule_with_warmup

# ============================================================================
# Configuration - All Training Hyperparameters
# ============================================================================

# Dynamically find project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"              # Where train.bin and val.bin are
MODEL_DIR = BASE_DIR / "model"            # Where initialized model is saved
CHECKPOINT_DIR = BASE_DIR / "checkpoints" # Where training checkpoints will be saved
CHECKPOINT_DIR.mkdir(exist_ok=True)       # Create checkpoints directory

# ---- Model Architecture Parameters ----
# block_size: Maximum sequence length (context window)
# This must match n_positions from model initialization
# 512 tokens ≈ 400 words
# The model sees 512 tokens at a time to predict the next token
block_size = 512

# ---- Batch Size Parameters ----
# batch_size: Number of sequences processed simultaneously on GPU
# 16 sequences × 512 tokens = 8,192 tokens per forward pass
# Larger batch = more stable gradients but more GPU memory
batch_size = 16

# grad_accum_steps: Gradient accumulation steps
# Simulates larger batch size without using more GPU memory
# Effective batch size = batch_size × grad_accum_steps = 16 × 2 = 32
# How it works:
#   - Run forward pass on batch 1 (16 sequences)
#   - Run forward pass on batch 2 (16 sequences)
#   - Average gradients from both batches
#   - THEN update weights
# This is crucial when GPU memory is limited
grad_accum_steps = 2

# ---- Training Duration Parameters ----
# max_steps: Total number of optimization steps (weight updates)
# 200,000 steps × 32 effective batch = 6.4M sequences seen
# With 54M training tokens and 512 block size:
#   - ~105,000 possible sequences
#   - Model will see each sequence ~60 times (60 epochs)
# Training time estimate: ~2-3 days on RTX 3090
max_steps = 200000

# eval_interval: How often to run validation
# Every 1000 steps = ~every 10-15 minutes
# Validation helps track if model is overfitting
eval_interval = 1000

# save_interval: How often to save checkpoints
# Every 5000 steps = ~every hour
# Checkpoints allow resuming if training crashes
save_interval = 5000

# ---- Learning Rate Schedule Parameters ----
# warmup_steps: Gradual learning rate increase at start
# Learning rate increases linearly from 0 to learning_rate over 2000 steps
# Why? Large learning rates at start can destabilize training
# After warmup, learning rate follows cosine decay schedule
warmup_steps = 2000

# learning_rate: Maximum learning rate (after warmup)
# 3e-4 = 0.0003 is a standard choice for transformer training
# GPT-2 and GPT-3 used similar values
# Too high: training unstable, loss explodes
# Too low: training too slow, gets stuck in local minima
learning_rate = 3e-4

# weight_decay: L2 regularization strength
# Penalizes large weights to prevent overfitting
# 0.1 is standard for transformer models
# Applied to all parameters except biases and layer norms
weight_decay = 0.1

# ---- Hardware Configuration ----
# Use GPU if available, otherwise CPU
# Training on CPU would take weeks/months - GPU is essential
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Dataset Definition - Memory-Mapped Data Loading
# ============================================================================

class GPTDataset(Dataset):
    """
    Memory-mapped dataset for efficient data loading.
    
    Key Innovation: Does NOT load entire dataset into RAM
    
    How normal loading works (BAD for large datasets):
        data = np.load("train.bin")  # Loads all 108 MB into RAM
        Problem: Wastes RAM, slow startup
    
    How memory mapping works (GOOD):
        data = np.memmap("train.bin")  # Maps file to virtual memory
        OS loads chunks on-demand when accessed
        Only active chunks are in RAM
        Benefit: Fast startup, minimal RAM usage
    """

    def __init__(self, path, block_size):
        # Create memory-mapped array pointing to the binary file
        # dtype=np.uint16: Each token is 2 bytes (matches how we saved it)
        # mode="r": Read-only (we don't modify the data)
        # This line is INSTANT - doesn't load data into RAM
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        
        # Store the context window size
        self.block_size = block_size

    def __len__(self):
        # Return number of possible sequences
        # We subtract block_size because we need block_size + 1 tokens
        # (block_size inputs + 1 target)
        # Example: 54M tokens - 512 = ~53.9M possible sequences
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Get one training example.
        
        Returns:
            x: Input sequence (512 tokens)
            y: Target sequence (512 tokens, shifted by 1)
        
        Example:
            Data: [1234, 5678, 90, 1122, 3344, 5566, ...]
            If idx=0:
                x = [1234, 5678, 90, 1122, 3344, ...] (512 tokens)
                y = [5678, 90, 1122, 3344, 5566, ...] (next 512 tokens)
            
            The model learns: given x[i], predict y[i]
                - Given token 1234, predict 5678
                - Given token 5678, predict 90
                - Given token 90, predict 1122
                - ... and so on
        """
        
        # Extract input tokens: positions idx to idx+block_size-1
        # .astype(np.int64) converts uint16 to int64 (required by PyTorch)
        # torch.from_numpy() converts numpy array to PyTorch tensor
        x = torch.from_numpy(
            self.data[idx : idx + self.block_size].astype(np.int64)
        )

        # Extract target tokens: positions idx+1 to idx+block_size
        # This is the "next token" for each position in x
        # Shifted by 1 position relative to x
        y = torch.from_numpy(
            self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64)
        )

        return x, y

# ============================================================================
# Load Datasets - Create Training and Validation Data Loaders
# ============================================================================

print("Loading datasets...")

# Create dataset objects (memory-mapped, doesn't load data yet)
train_dataset = GPTDataset(DATA_DIR / "train.bin", block_size)
val_dataset = GPTDataset(DATA_DIR / "val.bin", block_size)

# Create DataLoader for training data
# DataLoader handles:
#   - Batching (grouping sequences together)
#   - Shuffling (randomizing order each epoch)
#   - Parallel loading (loading next batch while GPU processes current batch)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,        # 16 sequences per batch
    shuffle=True,                 # Randomize order (important for training)
    pin_memory=True,              # Keep data in pinned memory for faster GPU transfer
    drop_last=True                # Drop incomplete last batch (keeps batch size consistent)
)

# Create DataLoader for validation data
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,        # Same batch size as training
    shuffle=False,                # No shuffling (order doesn't matter for validation)
    pin_memory=True,              # Faster GPU transfer
    drop_last=True                # Drop incomplete batch
)

# ============================================================================
# Load Model - Load the Initialized (Untrained) Model
# ============================================================================

print("Loading model...")

# Load the model we initialized in initialize_model.py
# This model has:
#   - Correct architecture (12 layers, 768 dimensions, etc.)
#   - Random weights (not trained yet)
#   - ~124M parameters
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

# Move model to GPU (all parameters transferred to GPU memory)
model.to(device)

# Set model to training mode
# This enables:
#   - Dropout (randomly drops connections for regularization)
#   - Batch norm updates (if any)
# Opposite is model.eval() for inference
model.train()

# ============================================================================
# Optimizer - Algorithm to Update Model Weights
# ============================================================================

# AdamW optimizer (Adam with Weight Decay)
# Adam: Adaptive learning rates per parameter
# Weight Decay: L2 regularization to prevent overfitting
optimizer = torch.optim.AdamW(
    model.parameters(),           # What to optimize (all model weights)
    lr=learning_rate,             # Learning rate (0.0003)
    betas=(0.9, 0.95),           # Momentum parameters (standard for transformers)
                                  # beta1=0.9: exponential decay for 1st moment (mean)
                                  # beta2=0.95: exponential decay for 2nd moment (variance)
    weight_decay=weight_decay     # L2 penalty (0.1)
)

# How AdamW works (simplified):
# For each parameter:
#   1. Compute gradient (direction to improve)
#   2. Use momentum (smooth out gradient noise)
#   3. Adapt learning rate per parameter (parameters that change a lot get smaller LR)
#   4. Apply weight decay (shrink weights slightly)
#   5. Update parameter: param = param - lr × adjusted_gradient

# ============================================================================
# Learning Rate Scheduler - Dynamic Learning Rate Adjustment
# ============================================================================

# Cosine schedule with warmup
# Creates a learning rate schedule that:
#   1. Warms up: 0 → learning_rate over 2000 steps (linear increase)
#   2. Cosine decay: learning_rate → ~0 over remaining steps (smooth decrease)
#
# Why warmup?
#   - Large LR at start can cause loss to explode
#   - Gradual increase stabilizes early training
#
# Why cosine decay?
#   - Learning rate decreases over time
#   - Allows model to "settle" into good minima
#   - Smooth decay is better than sudden drops
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,    # Warmup for first 2000 steps
    num_training_steps=max_steps      # Total training is 200,000 steps
)

# Learning rate over training:
# Steps 0-2000:      LR increases 0 → 0.0003 (warmup)
# Steps 2000-200000: LR decreases 0.0003 → ~0.00001 (cosine decay)

# ============================================================================
# Mixed Precision Training - Use FP16 for Speed and Memory Savings
# ============================================================================

# GradScaler enables mixed precision training
# Mixed precision means:
#   - Forward pass: FP16 (16-bit floats, 2 bytes per number)
#   - Backward pass: FP16
#   - Optimizer step: FP32 (32-bit floats, 4 bytes, more precise)
#
# Benefits:
#   - 2x faster training (GPU tensor cores optimized for FP16)
#   - 2x less GPU memory (can fit larger batches)
#   - Negligible accuracy loss
#
# The scaler prevents underflow:
#   - FP16 has limited range
#   - Very small gradients can become zero (underflow)
#   - Scaler multiplies gradients by large number (e.g., 2^16)
#   - After gradient step, divides by same number
scaler = torch.cuda.amp.GradScaler()

# ============================================================================
# Training Loop - The Main Training Process
# ============================================================================

print("Starting training...")

# Track current optimization step (how many weight updates)
global_step = 0

# Create iterator for training data
# This allows us to manually control when we fetch next batch
train_iter = iter(train_loader)

# Track training start time
start_time = time.time()

# Main training loop: run until we hit max_steps (200,000)
while global_step < max_steps:

    # Ensure model is in training mode (enable dropout, etc.)
    model.train()
    
    # Zero out gradients from previous step
    # Gradients accumulate by default in PyTorch
    # We need to clear them before computing new gradients
    optimizer.zero_grad()

    # Track accumulated loss across gradient accumulation steps
    total_loss = 0

    # ---- Gradient Accumulation Loop ----
    # Run forward/backward pass multiple times before updating weights
    # This simulates larger batch size without more GPU memory
    for _ in range(grad_accum_steps):  # 2 iterations

        # Get next batch of data
        try:
            x, y = next(train_iter)
        except StopIteration:
            # If we've gone through all data, start over (new epoch)
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # Move data to GPU
        # non_blocking=True: Don't wait for transfer to complete (parallel)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Dimensions:
        # x: [batch_size, block_size] = [16, 512]
        # y: [batch_size, block_size] = [16, 512]

        # ---- Forward Pass in Mixed Precision ----
        # autocast() enables FP16 computation
        with torch.cuda.amp.autocast():
            # Run model forward pass
            # input_ids: the input tokens [16, 512]
            # labels: the target tokens [16, 512]
            # The model computes:
            #   1. Predictions for next token at each position
            #   2. Cross-entropy loss comparing predictions to labels
            outputs = model(input_ids=x, labels=y)
            
            # Extract loss (average across batch and sequence length)
            loss = outputs.loss
            
            # Normalize loss by gradient accumulation steps
            # Why? We're accumulating gradients from multiple batches
            # Without division, gradients would be 2x too large
            # This makes grad_accum equivalent to larger batch size
            loss = loss / grad_accum_steps

        # ---- Backward Pass ----
        # Compute gradients (how to change weights to reduce loss)
        # scaler.scale() multiplies loss by large number to prevent underflow
        # .backward() computes ∂loss/∂weights for all parameters
        # Gradients are ACCUMULATED (added to existing gradients)
        scaler.scale(loss).backward()
        
        # Track total loss for logging
        total_loss += loss.item()

    # ---- Optimizer Step ----
    # After accumulating gradients from 2 batches, update weights
    
    # scaler.step() does:
    #   1. Unscale gradients (divide by scaler value)
    #   2. Check for inf/nan (skip update if found)
    #   3. Apply optimizer (update weights)
    scaler.step(optimizer)
    
    # Update scaler state (adjust scaling factor if needed)
    scaler.update()
    
    # Update learning rate according to schedule
    scheduler.step()

    # Increment global step counter
    global_step += 1

    # ========================================================================
    # Logging - Print Training Progress
    # ========================================================================

    # Every 100 steps, print progress
    if global_step % 100 == 0:
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        # Print current step, loss, and time
        print(f"Step {global_step} | Loss {total_loss:.4f} | Time {elapsed/60:.2f} min")
        # Example output: "Step 100 | Loss 8.2341 | Time 2.45 min"

    # ========================================================================
    # Validation - Evaluate Model on Held-Out Data
    # ========================================================================

    # Every 1000 steps, run validation
    if global_step % eval_interval == 0:
        # Switch to evaluation mode (disable dropout, batch norm updates)
        model.eval()
        
        # Track validation loss
        val_loss = 0
        val_batches = 0

        # No gradient computation during validation (saves memory and time)
        with torch.no_grad():
            # Iterate through validation batches
            for vx, vy in val_loader:
                # Move validation data to GPU
                vx = vx.to(device)
                vy = vy.to(device)

                # Forward pass in FP16 (faster)
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=vx, labels=vy)
                    # Accumulate validation loss
                    val_loss += outputs.loss.item()
                    val_batches += 1

                # Limit validation time (only use 100 batches)
                # Full validation would take too long
                # 100 batches is enough to get reliable estimate
                if val_batches >= 100:
                    break

        # Calculate average validation loss
        avg_val_loss = val_loss / val_batches
        
        # Calculate perplexity (more interpretable metric)
        # Perplexity = e^(loss)
        # Lower perplexity = better model
        # Perplexity ~1: model is certain about predictions
        # Perplexity ~32000: model is guessing randomly (as bad as random)
        # Good models: perplexity 10-50 after training
        perplexity = math.exp(avg_val_loss)

        print(f"Validation | Loss {avg_val_loss:.4f} | PPL {perplexity:.2f}")
        # Example output: "Validation | Loss 3.4521 | PPL 31.56"

        # Switch back to training mode
        model.train()

    # ========================================================================
    # Checkpoint Saving - Save Model Progress
    # ========================================================================

    # Every 5000 steps, save a checkpoint
    if global_step % save_interval == 0:
        # Create checkpoint directory (e.g., "checkpoints/step_5000")
        save_path = CHECKPOINT_DIR / f"step_{global_step}"
        
        # Save model weights and configuration
        # Creates two files:
        #   - pytorch_model.bin (weights)
        #   - config.json (architecture)
        model.save_pretrained(save_path)
        
        print(f"Checkpoint saved at step {global_step}")
        
        # Why save checkpoints?
        #   1. Resume training if it crashes
        #   2. Compare different training stages
        #   3. Use best checkpoint (not necessarily final one)

print("Training complete.")

# ============================================================================
# Training Complete Summary
# ============================================================================
#
# After 200,000 steps (2-3 days on RTX 3090):
#
# What happened:
#   - Model saw ~6.4M sequences (60 epochs through training data)
#   - Weights updated 200,000 times
#   - Learning rate gradually decreased from 0.0003 to ~0
#   - Model learned patterns in materials science text
#
# What you have:
#   - 40 checkpoints (saved every 5000 steps)
#   - Training logs (loss decreasing over time)
#   - Validation perplexity curve (should decrease then plateau)
#
# Next steps:
#   1. Evaluate best checkpoint on validation set
#   2. Generate sample text to qualitatively assess quality
#   3. Fine-tune on specific downstream tasks if needed
#
# Expected final perplexity: 15-30 (depends on data quality and training)
#
# ============================================================================
