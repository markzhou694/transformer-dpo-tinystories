import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from .models import TransformerModel


def compute_next_token_loss(logits, tokens):
  #Calculates the standard language modeling loss (Cross-Entropy Loss).
#It compares the model's predicted logits for token $t+1$ with the ground-truth token $t+1$.
#This effectively shifts the tokens to align predictions with targets.
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    preds = logits[:-1, :, :]
    gold = tokens[1:, :]
    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


def train_one_model(model, loader, epochs, model_name, device, lr=1e-3,
                    log_steps=100, sample_interval=30, max_steps_per_epoch=None,
                    enc=None, monosemantic_info=None, prompt="Once upon a"):
  #Manages the main training loop for a single model.
#It handles batch iteration, forward pass, loss calculation, backpropagation, and optimizer steps.
#Includes periodic logging of the average training loss.
    optimizer = optim.Adam(model.parameters(), lr=lr, fused=isinstance(model, TransformerModel))
    start_time = time.time()
    next_sample_time = start_time
    global_step = 0
    avg_losses_list = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0
        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1
            batch_tokens = batch_tokens.to(device)
            logits = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1
            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                avg_losses_list.append(avg_part_loss)
                print(f"[{model_name}] Epoch {epoch}/{epochs}, Step {batch_idx}/{len(loader)} "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0
            current_time = time.time()
            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                break
        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] End of Epoch {epoch} Avg Loss: {avg_loss:.4f}")
    return avg_losses_list


def evaluate_model(model, loader, device):
  #Performs a single pass over the test dataset.
#Calculates the model's average loss on unseen data.
#This provides a measure of generalization ability.
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_tokens in loader:
            batch_tokens = batch_tokens.to(device)
            logits = model(batch_tokens)
            loss = compute_next_token_loss(logits, batch_tokens)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss
