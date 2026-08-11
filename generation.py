import torch
import torch.nn.functional as F


def nucleus_sampling(logits, p=0.95):
  #Implements Top-P (Nucleus) sampling, a technique for generating more diverse and coherent text.
#It filters the vocabulary to only include tokens whose cumulative probability exceeds a threshold p.
#The function then samples from the remaining subset.
    if p >= 1.0:
        probs = F.softmax(logits, dim=-1)
        next_token_idx = torch.multinomial(probs, num_samples=1)
        return next_token_idx.item()
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs_to_remove = cum_probs > p
    sorted_probs_to_remove[..., 1:] = sorted_probs_to_remove[..., :-1].clone()
    sorted_probs_to_remove[..., 0] = False
    probs_to_keep = torch.zeros_like(sorted_probs)
    probs_to_keep[~sorted_probs_to_remove] = sorted_probs[~sorted_probs_to_remove]
    if probs_to_keep.sum() < 1e-6:
        return sorted_indices[0].item()
    probs_to_keep = probs_to_keep / probs_to_keep.sum()
    next_token_sorted_idx = torch.multinomial(probs_to_keep, num_samples=1)
    next_token_idx = sorted_indices[next_token_sorted_idx]
    return next_token_idx.item()


def generate_text(model, enc, init_text, max_new_tokens=30, device="cpu", top_p=None,
                  monosemantic_info=None, do_monosemantic=False):
  #Handles the autoregressive text generation process.
#It feeds the prompt tokens into the model and uses the model's logits to predict the next token (greedy or Top-P sampling).
#It appends the prediction to the sequence and repeats for a set number of steps.
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        generated_tokens = []
        for step_i in range(max_new_tokens):
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)
            next_logits = logits_seq[-1, 0, :]
            if top_p is None:
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)
            context_tokens.append(chosen_token)
            generated_tokens.append((chosen_token, []))
    model.train(was_training)
    final_text = enc.decode(context_tokens)
    prefix_text = init_text
    annotated_strs = [prefix_text]
    for (tid, neighs) in generated_tokens:
        token_str = enc.decode([tid])
        annotated_strs.append(token_str)
    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


def simple_search(model, enc, prompt, device="cpu"):
    candidates = []
    for _ in range(5):
        text, _ = generate_text(model, enc, prompt, max_new_tokens=30, top_p=0.95, device=device)
        candidates.append(text)
    return candidates
