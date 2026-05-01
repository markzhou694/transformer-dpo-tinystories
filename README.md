# Transformer Language Model with DPO Alignment

Group final project — NYU Courant, Fall 2025.  
Course taught by Prof. Matus Telgarsky.

A decoder-only Transformer language model trained on TinyStories, 
followed by a Direct Preference Optimization (DPO) alignment pipeline. 
Includes K-Gram MLP and LSTM baselines for comparison.

## Project Overview

- Decoder-only Transformer in PyTorch (multi-head self-attention, 
  RMSNorm, causal masking, top-p / nucleus sampling)
- Baselines: K-Gram MLP and LSTM on TinyStories
- DPO alignment pipeline: reference model freezing, per-sequence 
  log-probability computation, DPO loss with numerical-stability 
  tricks (`F.logsigmoid`)
- Embedding-dimension ablation (512 / 1024 / 2048) with 90/10 
  train/test split

## My Contributions

This was a 4-person team project. My specific contributions:

- **K-Gram MLP model** (Cell 3, Section 3): wrote the full 
  `KGramMLPSeqModel` including a vectorized forward pass that builds 
  per-token context windows via index gather + one-hot encoding, 
  avoiding the explicit Python loop in the starter formulation.
- **DPO alignment framework** (Cell 2): wrote the main DPO pipeline 
  — `get_batch_logps` for per-sequence log-probability computation 
  with proper logit/label shifting, the DPO loss using `F.logsigmoid` 
  for numerical stability, the synthetic `PreferenceDataset`, the 
  `dpo_collate_fn`, and the `train_dpo` training loop with frozen 
  reference model.
- **All training runs**: ran the full pre-training pipeline for all 
  three models (K-Gram MLP, LSTM, Transformer) on TinyStories, plus 
  the post-pretraining DPO alignment run on the Transformer.
- **Embedding-dimension ablation** (marked with `#XIAODI` in 
  Cell 3): trained at 512 / 1024 / 2048 with a 90/10 train/test 
  split, analyzing the effect of model capacity on language-modeling 
  performance.
- **Transformer (Cell 3, Section 5)**: my teammate authored the main 
  architecture; my role was verification, edits, and integration 
  with the rest of the pipeline.

My teammate's contributions: the LSTM model (Cell 3, Section 4), 
the Transformer architecture, and assistance with debugging and 
hyperparameter tuning.

## Notes on the DPO Setup

Without an external preference dataset (e.g. UltraFeedback, HH-RLHF), 
preference pairs are constructed synthetically by treating two 
r
