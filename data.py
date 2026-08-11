import random

import torch


class MixedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
      #Initializes the dataset, which combines sequences from the TinyStories dataset and any custom input files.
      # calculates the total length and stores the probability (p_tiny) for sampling from TinyStories.
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny
        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)
        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):#Returns the total number of sequences available in the combined dataset.
        return self.total_length

    def __getitem__(self, idx):
      #Samples and returns a single token sequence as a PyTorch tensor.
#It uses random selection, weighted by p_tiny, to choose between a TinyStories sequence and a custom input sequence.
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]
        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
  #A collate function used by the PyTorch DataLoader.
#Since input sequences have varying lengths, this function finds the maximum length.
#It pads all sequences with zeros to that maximum length, returning a single padded tensor.
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)
    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq
    return padded
