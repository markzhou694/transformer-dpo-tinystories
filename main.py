import argparse
import copy

import tiktoken
import torch
from datasets import load_dataset

from pico_llm.data import MixedSequenceDataset, seq_collate_fn
from pico_llm.dpo import PreferenceDataset, dpo_collate_fn, train_dpo
from pico_llm.generation import generate_text, simple_search
from pico_llm.models import KGramMLPSeqModel, LSTMSeqModel, TransformerModel
from pico_llm.plotting import plot_train_test_loss_curves
from pico_llm.training import evaluate_model, train_one_model


def parse_args():#Parses command-line arguments (e.g., embedding size, block size, prompt, number of heads, model layers).
#Returns an object containing all configuration settings for the training run.
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None)
    parser.add_argument("--tinystories_weight", type=float, default=0.5)
    parser.add_argument("--max_steps_per_epoch", type=int, default=None)
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1)
    parser.add_argument("--monosemantic_enabled", action="store_true")
    parser.set_defaults(monosemantic_enabled=False)
    parser.add_argument("--kgram_k", type=int, default=3)
    parser.add_argument("--kgram_chunk_size", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--embed_size", type=int, default=512)
    parser.add_argument("--prompt", type=str, default="Once upon a")
    parser.add_argument("--n_heads", type=int, default= 8)
    parser.add_argument("--n_blocks", type=int, default=8)
    parser.add_argument("--device_id", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


def main():
  #The primary entry point of the script. It orchestrates the entire process.
#It loads arguments, initializes the device, loads and tokenizes the data, and splits it into train/test sets.
#It instantiates the selected models (currently only transformer), initiates training, performs final evaluation, and saves the loss plot.



    args = parse_args()
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size
    embed_size = args.embed_size
    batch_size = 16
    num_epochs = 3
    learning_rate = 1e-3
    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30
    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers
    requested_device_id = args.device_id
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        try:
            device = torch.device(requested_device_id)
            _ = torch.tensor([1]).to(device)
        except Exception:
            device = torch.device("cpu")

    print(f"Using device: {device}")

    tinystories_seqs = []
    other_seqs = []
    if args.tinystories_weight > 0.0:
        try:
            dataset = load_dataset("roneneldan/TinyStories", split="train")
            dataset = dataset.select(range(train_subset_size))
        except Exception:
            dataset = None
    else:
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)

    if args.input_files:
        for filepath in args.input_files:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)

    if len(tinystories_seqs) == 0 and len(other_seqs) == 0:
        print("No data loaded, exiting.")
        return

    combined_dataset = MixedSequenceDataset(tinystories_seqs, other_seqs, args.tinystories_weight)

    # *** OUR CHANGE ***
    # calculates a 90% training size and a 10% test size.
    # uses torch.utils.data.random_split to divide the data into train_set and test_set.
    # creates a test_loader to handle batching and loading of the test data for evaluation.
    train_size = int(0.9 * len(combined_dataset))
    test_size = len(combined_dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(combined_dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=seq_collate_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=seq_collate_fn)
    # *** END OUR CHANGE ***
    # =MODELS ===
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=embed_size,
        n_heads=args.n_heads,
        n_blocks=args.n_blocks
    ).to(device)

    models = {
        #"kgram_mlp_seq": kgram_model,
       # "lstm_seq": lstm_model,
        "transformer": transformer,
    }

    # *** OUR CHANGE ***
    #Initializes dictionaries to store the training loss history and final test loss for all models.
    all_train_losses = {}
    all_test_losses = {}
    # *** END OUR CHANGE ***

    # === trian validate ==
    for model_name, model in models.items():
        print(f"\n=== Training model: {model_name} ===")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[{model_name}] Parameters: {params:,}")

        current_lr = 1e-3 if model_name == "transformer" else learning_rate #I increased the LR (old:3e-4)
        print(f"[{model_name}] Learning rate: {current_lr}")

        train_losses = train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=current_lr,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt
        )
        all_train_losses[model_name] = train_losses

        # *** OUR CHANGE ***
        #Calls the new evaluate_model function for each trained model using the test_loader.
        #Stores the final test loss and prints it to the console.
        test_loss = evaluate_model(model, test_loader, device)
        all_test_losses[model_name] = test_loss
        print(f"[{model_name}] Final Test Loss: {test_loss:.4f}")
        # *** END OUR CHANGE ***

        # GENERATING TEXT
        with torch.no_grad():
            text_greedy, _ = generate_text(model, enc, args.prompt, max_new_tokens=20, device=device, top_p=None)
            text_topp, _ = generate_text(model, enc, args.prompt, max_new_tokens=20, device=device, top_p=0.95)
            text_topp1, _ = generate_text(model, enc, args.prompt, max_new_tokens=20, device=device, top_p=1.0)

        print(f"\n[{model_name}] Final sample (greedy):\n{text_greedy}")
        print(f"[{model_name}] Final sample (top-p=0.95):\n{text_topp}")
        print(f"[{model_name}] Final sample (top-p=1.0):\n{text_topp1}")
        print("--------------------------------------------------")

        # TEST-TIME SEARCH EXTENSION =
        print("\n[Test-Time Search] Multiple Candidates:")
        search_results = simple_search(model, enc, args.prompt, device=device)

        for i, s in enumerate(search_results):
          print(f"Candidate {i+1}: {s}\n")

    # *** OUR CHANGE ***
    #Calls the plot_train_test_loss_curves function after all models have been trained and evaluated, generating the final loss comparison graph.
    plot_train_test_loss_curves(all_train_losses, all_test_losses, log_interval_steps)
    # *** END OUR CHANGE ***
    # ... inside main(), after plot_train_test_loss_curves call ...

    # ==========================================
    # === START DPO ALIGNMENT STEP ===
    # ==========================================

    # We only apply DPO to the Transformer model as requested
    if "transformer" in models:
        print("\n=== Starting DPO (Alignment) Phase for Transformer ===")

        # 1. Create the Reference Model (Frozen Copy)
        # Deepcopy the model currently in GPU memory
        ref_model = copy.deepcopy(transformer)
        ref_model.to(device)
        ref_model.eval()
        # Freeze parameters of reference model to save memory/computation
        for param in ref_model.parameters():
            param.requires_grad = False

        # 2. Prepare Preference Data
        # Since we don't have an external preference dataset, we create a synthetic one
        # using the TinyStories data we already loaded.
        # Ideally, 'chosen' is high quality and 'rejected' is low quality.
        dpo_dataset = PreferenceDataset(tinystories_seqs[:2000]) # Use subset for speed
        dpo_loader = torch.utils.data.DataLoader(
            dpo_dataset,
            batch_size=8, # Smaller batch size for DPO due to double forward pass
            shuffle=True,
            collate_fn=dpo_collate_fn
        )

        # 3. Run DPO Training
        # We use a much smaller learning rate (1e-5 or 5e-6) to avoid breaking the model
        train_dpo(transformer, ref_model, dpo_loader, epochs=1, device=device, lr=5e-6, beta=0.1)

        # 4. Evaluate Post-DPO
        print("\n[DPO] Generating text after alignment:")
        with torch.no_grad():
             text_dpo, _ = generate_text(transformer, enc, args.prompt, max_new_tokens=30, device=device)
        print(f"[Post-DPO] Sample: {text_dpo}")

        # Clean up ref model to free memory
        del ref_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ==========================================
    # === END DPO ALIGNMENT STEP ===
    # ==========================================


if __name__ == "__main__":
    main()
