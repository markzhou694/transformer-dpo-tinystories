import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .data import seq_collate_fn


def get_batch_logps(logits, labels, average_log_prob=False):
    """
    Computes the log probabilities of the given labels under the given logits.

    logits: (seq_len, batch, vocab_size) - expected from your model
    labels: (seq_len, batch)
    """
    # Align logits and labels:
    # The model predicts the NEXT token, so logits[t] corresponds to labels[t+1]
    # We shift the input so we are calculating the probability of the actual sequence occurring.

    # Slice to remove the last logit (no next token) and the first label (no previous context)
    shift_logits = logits[:-1, :, :]
    shift_labels = labels[1:, :]

    # Standard Cross Entropy logic but keeping the raw log probs
    loss_fct = nn.CrossEntropyLoss(reduction='none')

    # PyTorch CrossEntropy expects (N, C) or (N, C, d1...)
    # We flatten to (seq_len*batch, vocab)
    shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    shift_labels = shift_labels.reshape(-1)

    # CrossEntropy is -log(p), so we take negative to get log(p)
    token_logps = -loss_fct(shift_logits, shift_labels)

    # Reshape back to (seq_len-1, batch)
    token_logps = token_logps.view(labels.shape[0]-1, labels.shape[1])

    # Sum over the sequence dimension to get log_prob of the whole sentence
    seq_logps = token_logps.sum(dim=0)

    if average_log_prob:
        seq_logps = seq_logps / token_logps.shape[0]

    return seq_logps


def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    """
    The core DPO loss formula:
    L_DPO = -E[log sigmoid(beta * (log(r_theta) - log(r_ref)))]
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = pi_logratios - ref_logratios

    # F.logsigmoid is numerically more stable than log(sigmoid(x))
    losses = -F.logsigmoid(beta * logits)

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


class PreferenceDataset(torch.utils.data.Dataset):
    """
    Creates pairs of (Chosen, Rejected) sequences.
    For this custom implementation, we will simulate this by taking:
    Chosen = A real sequence from the training data
    Rejected = A random corrupted sequence (or a different random sequence)
    """
    def __init__(self, original_seqs):
        self.seqs = original_seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        # 1. Get a "Chosen" sequence (real data)
        chosen_seq = torch.tensor(self.seqs[idx], dtype=torch.long)

        # 2. Get a "Rejected" sequence (randomly picked different story)
        # In a real scenario, this would be a hallucinated or toxic output
        rand_idx = random.randint(0, len(self.seqs)-1)
        if rand_idx == idx: rand_idx = (idx + 1) % len(self.seqs)
        rejected_seq = torch.tensor(self.seqs[rand_idx], dtype=torch.long)

        # Determine minimum length to ensure we can stack them if needed,
        # or just return them as tuple. Your collate function handles padding.
        return chosen_seq, rejected_seq


def dpo_collate_fn(batch):
    """
    Custom collate for DPO. Returns (padded_chosen, padded_rejected)
    """
    chosen_list = [item[0] for item in batch]
    rejected_list = [item[1] for item in batch]

    # Reuse your existing seq_collate_fn logic
    return seq_collate_fn(chosen_list), seq_collate_fn(rejected_list)


def train_dpo(model, ref_model, loader, epochs, device, lr=1e-5, beta=0.1, log_steps=10):
    """
    Training loop specifically for DPO.
    """
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    model.train()
    ref_model.eval() # Reference model must be frozen

    print("Starting DPO Alignment...")

    for epoch in range(epochs):
        total_loss = 0
        steps = 0

        for batch_idx, (chosen_tokens, rejected_tokens) in enumerate(loader):
            chosen_tokens = chosen_tokens.to(device)
            rejected_tokens = rejected_tokens.to(device)

            # 1. Forward pass Policy Model
            policy_chosen_logits = model(chosen_tokens)
            policy_rejected_logits = model(rejected_tokens)

            policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_tokens)
            policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_tokens)

            # 2. Forward pass Reference Model (No Grad)
            with torch.no_grad():
                ref_chosen_logits = ref_model(chosen_tokens)
                ref_rejected_logits = ref_model(rejected_tokens)

                ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_tokens)
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_tokens)

            # 3. Compute DPO Loss
            loss, reward_chosen, reward_rejected = dpo_loss(
                policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps,
                beta=beta
            )

            # 4. Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1

            if batch_idx % log_steps == 0:
                print(f"[DPO] Epoch {epoch+1} Step {batch_idx} Loss: {loss.item():.4f} "
                      f"Reward Diff: {(reward_chosen - reward_rejected).item():.4f}")

        print(f"[DPO] End of Epoch {epoch+1} Avg Loss: {total_loss/steps:.4f}")
