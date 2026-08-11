"""Direct Preference Optimization utilities.

This module supports both versions of the project:

1. The original DataLoader workflow that randomly pairs TinyStories samples.
2. The later workflow that uses OpenAI to create prompt/chosen/rejected records.

The OpenAI client is created only when preference generation is requested.
The API key must be supplied through the OPENAI_API_KEY environment variable.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from .data import seq_collate_fn


# ---------------------------------------------------------------------------
# Shared DPO mathematics
# ---------------------------------------------------------------------------

def get_batch_logps(logits, labels, average_log_prob=False, attention_mask=None):
    """Return each sequence's autoregressive log probability.

    ``logits[t]`` predicts ``labels[t + 1]``, so the tensors are shifted by one
    position before the token log probabilities are gathered.
    """
    shifted_logits = logits[:-1]
    shifted_labels = labels[1:]

    token_logps = F.log_softmax(shifted_logits, dim=-1)
    token_logps = token_logps.gather(
        dim=-1, index=shifted_labels.unsqueeze(-1)
    ).squeeze(-1)

    if attention_mask is not None:
        valid_tokens = attention_mask[1:].to(token_logps.dtype)
        token_logps = token_logps * valid_tokens
        token_count = valid_tokens.sum(dim=0).clamp_min(1)
    else:
        token_count = torch.full(
            (labels.shape[1],),
            token_logps.shape[0],
            dtype=token_logps.dtype,
            device=token_logps.device,
        )

    sequence_logps = token_logps.sum(dim=0)
    if average_log_prob:
        sequence_logps = sequence_logps / token_count
    return sequence_logps


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    ref_chosen_logps,
    ref_rejected_logps,
    beta=0.1,
):
    """Compute the standard DPO loss and detached diagnostic rewards."""
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = ref_chosen_logps - ref_rejected_logps
    preference_logit = policy_log_ratio - reference_log_ratio

    losses = -F.logsigmoid(beta * preference_logit)
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()

    return losses.mean(), chosen_rewards.mean(), rejected_rewards.mean()


# ---------------------------------------------------------------------------
# Legacy TinyStories pair workflow
# ---------------------------------------------------------------------------

class PreferenceDataset(torch.utils.data.Dataset):
    """Legacy synthetic pairs used by the earlier notebook.

    The current item is treated as chosen and a different random item is
    treated as rejected. This remains available so the old ``main.py`` still
    runs, although same-prompt OpenAI preference records are more meaningful.
    """

    def __init__(self, original_seqs):
        if len(original_seqs) < 2:
            raise ValueError("PreferenceDataset needs at least two sequences")
        self.seqs = original_seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        chosen_seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        rejected_idx = random.randrange(len(self.seqs) - 1)
        if rejected_idx >= idx:
            rejected_idx += 1
        rejected_seq = torch.tensor(self.seqs[rejected_idx], dtype=torch.long)
        return chosen_seq, rejected_seq


def dpo_collate_fn(batch):
    """Pad the chosen and rejected sides of a legacy batch separately."""
    chosen_list = [item[0] for item in batch]
    rejected_list = [item[1] for item in batch]
    return seq_collate_fn(chosen_list), seq_collate_fn(rejected_list)


def _train_dpo_loader(
    policy_model,
    ref_model,
    loader,
    epochs,
    device,
    lr,
    beta,
    log_steps,
):
    optimizer = optim.AdamW(policy_model.parameters(), lr=lr)
    policy_model.train()
    ref_model.eval()
    losses = []

    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    for epoch in range(epochs):
        total_loss = 0.0
        steps = 0

        for batch_idx, (chosen_tokens, rejected_tokens) in enumerate(loader):
            chosen_tokens = chosen_tokens.to(device)
            rejected_tokens = rejected_tokens.to(device)

            policy_chosen = get_batch_logps(
                policy_model(chosen_tokens), chosen_tokens
            )
            policy_rejected = get_batch_logps(
                policy_model(rejected_tokens), rejected_tokens
            )

            with torch.no_grad():
                ref_chosen = get_batch_logps(
                    ref_model(chosen_tokens), chosen_tokens
                )
                ref_rejected = get_batch_logps(
                    ref_model(rejected_tokens), rejected_tokens
                )

            loss, chosen_reward, rejected_reward = dpo_loss(
                policy_chosen,
                policy_rejected,
                ref_chosen,
                ref_rejected,
                beta,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            total_loss += loss.item()
            steps += 1

            if log_steps and batch_idx % log_steps == 0:
                reward_difference = (chosen_reward - rejected_reward).item()
                print(
                    f"[DPO] Epoch {epoch + 1} Step {batch_idx} "
                    f"Loss: {loss.item():.4f} "
                    f"Reward Diff: {reward_difference:.4f}"
                )

        if steps:
            print(
                f"[DPO] End of Epoch {epoch + 1} "
                f"Avg Loss: {total_loss / steps:.4f}"
            )

    return losses


# ---------------------------------------------------------------------------
# Safe OpenAI preference-data generation
# ---------------------------------------------------------------------------

CHOSEN_INSTRUCTIONS = """
Rewrite the story so it is clear, smooth, warm, and easy to read.
Use complete sentences and simple words. Add a few gentle details, keep the
events logical, and give the story a finished ending.
""".strip()

REJECTED_INSTRUCTIONS = """
Rewrite the story in a dull but logical way. Use short, flat sentences, plain
emotion, and simple descriptions so that it reads like a list of actions.
""".strip()


def _create_openai_client():
    """Create a client without accepting or printing an API key in source."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in your shell or Colab secrets; "
            "do not paste it into the notebook or source code."
        )

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "The openai package is required for preference generation. "
            "Install it with: pip install openai"
        ) from error

    return OpenAI()


def generate_preference_pair(
    prompt_text,
    max_tokens=70,
    *,
    client=None,
    chosen_model="gpt-4o",
    rejected_model="gpt-4o-mini",
):
    """Generate a same-prompt chosen/rejected pair.

    ``client`` may be injected for testing. When omitted, the client reads the
    key from ``OPENAI_API_KEY``. Neither the key nor the responses are printed.
    """
    if client is None:
        client = _create_openai_client()

    chosen_response = client.chat.completions.create(
        model=chosen_model,
        messages=[
            {"role": "system", "content": CHOSEN_INSTRUCTIONS},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )

    rejected_response = client.chat.completions.create(
        model=rejected_model,
        messages=[
            {"role": "system", "content": REJECTED_INSTRUCTIONS},
            {"role": "user", "content": prompt_text},
        ],
        max_tokens=max_tokens,
        temperature=1.6,
    )

    chosen_text = chosen_response.choices[0].message.content.strip()
    rejected_text = rejected_response.choices[0].message.content.strip()
    return chosen_text, rejected_text


def save_dpo_data(dpo_data, filepath="dpo_dataset_1000.json"):
    """Save the historical prompt/chosen/rejected token-list schema."""
    def as_list(value):
        return value.tolist() if hasattr(value, "tolist") else list(value)

    serializable = []
    for entry in dpo_data:
        serializable.append(
            {
                "prompt": as_list(entry["prompt"]),
                "chosen": as_list(entry["chosen"]),
                "rejected": as_list(entry["rejected"]),
            }
        )

    path = Path(filepath)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_dpo_data(filepath="dpo_dataset_1000.json"):
    """Load the JSON cache produced by the previous notebook."""
    path = Path(filepath)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_dpo_dataset_with_openai(
    num_examples=50,
    max_prompt_len=40,
    max_completion_len=80,
    use_cache=True,
    cache_file="dpo_dataset_1000.json",
    *,
    chosen_model="gpt-4o",
    rejected_model="gpt-4o-mini",
    seed=None,
):
    """Create preference records from random TinyStories prompts."""
    if use_cache:
        cached = load_dpo_data(cache_file)
        if cached is not None:
            return cached

    from datasets import load_dataset
    import tiktoken

    random_generator = random.Random(seed)
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    encoder = tiktoken.get_encoding("gpt2")
    client = _create_openai_client()
    dpo_data = []

    attempts = 0
    maximum_attempts = num_examples * 5
    while len(dpo_data) < num_examples and attempts < maximum_attempts:
        attempts += 1
        story = dataset[random_generator.randrange(len(dataset))]["text"]
        story_tokens = encoder.encode(story)
        if len(story_tokens) < max_prompt_len + 5:
            continue

        prompt_tokens = story_tokens[:max_prompt_len]
        prompt_text = encoder.decode(prompt_tokens)
        chosen_text, rejected_text = generate_preference_pair(
            prompt_text,
            max_completion_len,
            client=client,
            chosen_model=chosen_model,
            rejected_model=rejected_model,
        )

        dpo_data.append(
            {
                "prompt": prompt_tokens,
                "chosen": encoder.encode(chosen_text)[:max_completion_len],
                "rejected": encoder.encode(rejected_text)[:max_completion_len],
            }
        )

    if len(dpo_data) < num_examples:
        raise RuntimeError(
            f"Generated only {len(dpo_data)} of {num_examples} requested pairs"
        )

    if use_cache:
        save_dpo_data(dpo_data, cache_file)
    return dpo_data


# Historical function name retained for old notebook calls.
prepare_dpo_dataset_with_gpt4 = prepare_dpo_dataset_with_openai


# ---------------------------------------------------------------------------
# Same-prompt DPO training used by FINALPROJECT_TESTONLY.ipynb
# ---------------------------------------------------------------------------

def _completion_logprob(model, prompt, completion, device):
    if not prompt or not completion:
        raise ValueError("prompt and completion must both contain tokens")

    sequence = prompt + completion
    sequence_tensor = torch.tensor(
        sequence, dtype=torch.long, device=device
    ).unsqueeze(1)
    logits = model(sequence_tensor)

    # logits[t] predicts sequence[t + 1]. The first completion token is at
    # sequence[len(prompt)], so its predictive logit is at len(prompt) - 1.
    shifted_logits = logits[:-1, 0]
    shifted_targets = sequence_tensor[1:, 0]
    first_completion_logit = len(prompt) - 1
    last_completion_logit = first_completion_logit + len(completion)

    completion_logits = shifted_logits[
        first_completion_logit:last_completion_logit
    ]
    completion_targets = shifted_targets[
        first_completion_logit:last_completion_logit
    ]
    token_logps = F.log_softmax(completion_logits, dim=-1)
    return token_logps.gather(
        dim=-1, index=completion_targets.unsqueeze(-1)
    ).sum()


def compute_dpo_loss(
    policy_model,
    ref_model,
    prompt,
    chosen,
    rejected,
    beta=0.1,
    device="cpu",
):
    """Compute DPO loss for one same-prompt preference record."""
    policy_chosen = _completion_logprob(
        policy_model, prompt, chosen, device
    )
    policy_rejected = _completion_logprob(
        policy_model, prompt, rejected, device
    )

    with torch.no_grad():
        ref_chosen = _completion_logprob(ref_model, prompt, chosen, device)
        ref_rejected = _completion_logprob(
            ref_model, prompt, rejected, device
        )

    loss, _, _ = dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
    )
    return loss


def _train_dpo_records(
    policy_model,
    ref_model,
    dpo_data,
    epochs,
    device,
    lr,
    beta,
    log_steps,
):
    optimizer = optim.Adam(policy_model.parameters(), lr=lr)
    policy_model.train()
    ref_model.eval()
    losses = []

    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)

    for epoch in range(epochs):
        random.shuffle(dpo_data)
        for step, example in enumerate(dpo_data, start=1):
            loss = compute_dpo_loss(
                policy_model,
                ref_model,
                example["prompt"],
                example["chosen"],
                example["rejected"],
                beta,
                device,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if log_steps and step % log_steps == 0:
                print(
                    f"[DPO] Epoch {epoch + 1} Step {step}/{len(dpo_data)} "
                    f"Loss: {loss.item():.4f}"
                )

    return losses


def train_dpo(
    model=None,
    ref_model=None,
    loader=None,
    epochs=2,
    device="cpu",
    lr=1e-5,
    beta=0.1,
    log_steps=10,
    *,
    policy_model=None,
    dpo_data=None,
):
    """Train with either the old loader API or the later record API.

    Old call, retained unchanged::

        train_dpo(model, ref_model, loader, epochs, device)

    Later notebook call::

        train_dpo(policy_model=model, ref_model=ref_model, dpo_data=records,
                  epochs=3, device=device)
    """
    if policy_model is None:
        policy_model = model
    if policy_model is None or ref_model is None:
        raise ValueError("policy model and reference model are required")

    if dpo_data is not None:
        return _train_dpo_records(
            policy_model,
            ref_model,
            dpo_data,
            epochs,
            device,
            lr,
            beta,
            log_steps,
        )

    if loader is None:
        raise ValueError("provide either loader or dpo_data")
    return _train_dpo_loader(
        policy_model,
        ref_model,
        loader,
        epochs,
        device,
        lr,
        beta,
        log_steps,
    )
