import torch
import torch.nn as nn
import torch.nn.functional as F


class KGramMLPSeqModel(nn.Module):
    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1):
      #Initializes the K-Gram MLP model.
#This model predicts the next token based on a fixed window of the previous k tokens.
#It sets up the sequential MLP layers.
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size
        input_dim = k * vocab_size
        hidden_dim = embed_size
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(num_inner_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, tokens_seq):
       #Performs the forward pass by constructing a context window of size $k$ for every token.
#It converts this context into a large, concatenated one-hot input vector.
#The vector is passed through the multi-layer perceptron (MLP) to get the next-token logits.
        """
        Fully vectorized K-gram forward pass.
        tokens_seq: (seq_len, batch)
        returns logits: (seq_len, batch, vocab_size)
        """
        seq_len, batch = tokens_seq.shape
        device = tokens_seq.device

        # Build index matrix:
        # each row t contains [t-(k-1), ..., t]
        idx = torch.arange(seq_len, device=device).unsqueeze(1) - torch.arange(self.k-1, -1, -1, device=device)
        # shape: (seq_len, k)

        # Mark padding positions
        pad_mask = idx < 0
        idx = idx.clamp(min=0)

        # Gather contexts from tokens_seq
        # tokens_seq: (seq_len, batch)
        contexts = tokens_seq[idx, :]   # (seq_len, k, batch)
        contexts = contexts.permute(0, 2, 1)  # -> (seq_len, batch, k)

        # Replace pad positions with 0
        contexts[pad_mask.unsqueeze(1).expand_as(contexts)] = 0

        # One-hot encode all at once
        one_hot = F.one_hot(contexts, num_classes=self.vocab_size).float()
        flat = one_hot.flatten(-2, -1)  # (seq_len, batch, k*vocab)

        # Run through MLP in ONE pass
        logits = self.net(flat)

        return logits


class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
      #Initializes the standard Long Short-Term Memory (LSTM) model.
#This includes the token embedding layer and the LSTM recurrent layer.
#A final linear layer projects the hidden state back to the vocabulary size (logits).
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
      #Performs the forward pass: embeds input tokens.
#Passes the embeddings through the LSTM layer.
#Produces the next-token logits via a final linear layer.
        emb = self.embedding(tokens_seq)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)
        logits = self.linear(out)
        return logits


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
      #Initializes the RMS Normalization layer.
#This is an alternative to LayerNorm often used in modern Transformer architectures like LLama.
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
      #Applies the RMS normalization process.
#This involves dividing the input by its Root Mean Square.
#Multiplies the result by a learned weight parameter.
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
      #Initializes a single Transformer block.
#It consists of a multi-head self-attention layer and a feed-forward MLP.
#Each sub-layer is preceded by RMS Normalization.
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=False)
        self.norm2 = RMSNorm(d_model)
        hidden_dim = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x, mask=None):
      #Executes the forward pass through a single block.
#Applies RMSNorm and Attention (with a mask), followed by a residual connection.
#Then applies another RMSNorm and the MLP.
#Concludes with a final residual connection.
        normed_x = self.norm1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x, attn_mask=mask, is_causal=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=50257, d_model=1024, n_heads=2, n_blocks=4):
      #Initializes the complete Transformer model.
#This includes the token embedding and a stack of TransformerBlocks.
#It uses a final RMSNorm layer and the unembedding linear layer.
        super().__init__()
        self.vocab_size = vocab_size
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_blocks)])
        self.norm = RMSNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size)
        self.causal_mask = None

    def _prepare_causal_mask(self, seq_len, device):
      #Creates the causal (look-ahead) mask, which is an upper-triangular matrix.
#This mask is used in the self-attention mechanism.
#It ensures a token only attends to previous tokens in the sequence.
        if self.causal_mask is not None and self.causal_mask.size(0) >= seq_len:
            return self.causal_mask[:seq_len, :seq_len].to(device)
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        self.causal_mask = mask
        return mask

    def forward(self, tokens_seq):
      #Performs the full forward pass, starting with token embedding.
#Applies the stack of Transformer attention blocks.
#Concludes with final normalization and generating vocabulary logits.
        seq_len, batch_size = tokens_seq.shape
        device = tokens_seq.device
        x = self.tok_embed(tokens_seq)
        mask = self._prepare_causal_mask(seq_len, device)
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x)
        logits = self.unembed(x)
        return logits
