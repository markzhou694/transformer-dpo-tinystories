# Transformer Language Model with DPO Alignment

Group final project — NYU Courant, Fall 2025  
Course taught by Prof. Matus Telgarsky

This project trained a decoder-only Transformer language model on TinyStories,
followed by a Direct Preference Optimization (DPO) alignment stage. We also
implemented K-gram MLP and LSTM baselines for comparison.

## Project Overview

- Decoder-only Transformer in PyTorch with multi-head self-attention, RMSNorm,
  causal masking, and top-p sampling
- K-gram MLP and LSTM baselines trained on TinyStories
- Next-token prediction, training, evaluation, and text generation pipeline
- GPT-generated preference pairs for DPO
- Frozen reference model, sequence log-probability calculation, and DPO loss
  implemented with `F.logsigmoid`
- Embedding-dimension experiments at 512, 1024, and 2048 with a 90/10
  train/test split

## My Contributions

This was a four-person team project. My main contributions were:

- **K-gram MLP:** implemented the full `KGramMLPSeqModel`, including a
  vectorized forward pass that constructs per-token context windows through
  index gathering and one-hot encoding.
- **DPO framework:** implemented the sequence log-probability calculation,
  DPO objective, frozen reference-model setup, preference-data processing, and
  DPO training loop.
- **GPT preference data:** sampled prompts from TinyStories and used GPT to
  generate a clearer `chosen` completion and a flatter `rejected` completion
  from the same prompt.
- **All training runs:** ran the complete training pipeline for the K-gram
  MLP, LSTM, and Transformer, as well as the DPO fine-tuning run.
- **Embedding-dimension experiments:** trained the models with embedding sizes
  of 512, 1024, and 2048 and compared their language-modeling behavior.
- **Transformer integration:** verified and edited the Transformer code and
  integrated it with the data, training, evaluation, generation, and DPO
  pipelines.

The main Transformer architecture and LSTM implementation were written by a
teammate. Other team work included debugging and hyperparameter tuning.

## DPO Setup

The original TinyStories text was used as the source of prompts. For each
preference example, the first 40 tokens of a sampled story were used as the
prompt. GPT then generated two completions from that same prompt:

- `chosen`: clearer, smoother, and more detailed
- `rejected`: simpler, flatter, and less engaging

The generated data was tokenized with the GPT-2 tokenizer and stored as
`prompt`, `chosen`, and `rejected` token sequences. A total of 1,000 preference
pairs were generated and cached in JSON format.

For DPO training, the pretrained Transformer was used as the policy model and
a frozen copy was used as the reference model. I trained on 500 of the 1,000
preference pairs for three epochs, giving 1,500 optimization steps. The
recorded DPO training loss decreased from 0.6931 to 0.3627.

This loss reduction shows that the policy learned the training preferences,
but it is not a complete alignment evaluation because there was no separate
held-out preference benchmark.

## Code Structure

- `main.py` — full training, evaluation, generation, and DPO experiment
- `pico_llm/models.py` — K-gram MLP, LSTM, RMSNorm, and Transformer models
- `pico_llm/data.py` — dataset handling and sequence padding
- `pico_llm/training.py` — next-token loss, training, and evaluation
- `pico_llm/generation.py` — greedy decoding and top-p sampling
- `pico_llm/dpo.py` — GPT preference generation and DPO training
- `pico_llm/plotting.py` — training and test loss visualization
