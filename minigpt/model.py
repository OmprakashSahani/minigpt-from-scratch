import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, block_size):
        super().__init__()

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (C ** 0.5)
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, block_size):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(embed_dim, block_size)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))

        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.block = TransformerBlock(embed_dim, block_size)
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        token_embeddings = self.token_embedding(x)

        positions = torch.arange(T, device=x.device)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.block(x)
        x = self.ln_final(x)

        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss