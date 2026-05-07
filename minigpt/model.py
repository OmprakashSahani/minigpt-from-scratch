import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, block_size, num_heads):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x, past_k=None, past_v=None):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if past_k is not None and past_v is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

            max_cache_length = self.mask.size(0)
            k = k[:, :, -max_cache_length:, :]
            v = v[:, :, -max_cache_length:, :]

        total_length = k.size(2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        mask = self.mask[total_length - T:total_length, :total_length]
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)

        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.proj(out)

        return out, k, v


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
    def __init__(self, embed_dim, block_size, num_heads):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadCausalSelfAttention(embed_dim, block_size, num_heads)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.feed_forward = FeedForward(embed_dim)

    def forward(self, x, past_k=None, past_v=None):
        attention_out, k, v = self.attention(
            self.ln1(x),
            past_k=past_k,
            past_v=past_v,
        )

        x = x + attention_out
        x = x + self.feed_forward(self.ln2(x))

        return x, k, v


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, block_size, num_heads=4, num_layers=2):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, block_size, num_heads) for _ in range(num_layers)]
        )
        self.ln_final = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, targets=None, cache=None):
        B, T = x.shape

        token_embeddings = self.token_embedding(x)

        past_length = 0
        if cache is not None:
            past_length = cache[0][0].size(2)

        positions = torch.arange(
            past_length,
            past_length + T,
            device=x.device,
        )

        positions = positions.clamp(max=self.position_embedding.num_embeddings - 1)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings

        new_cache = []

        for i, block in enumerate(self.blocks):
            past_k = None
            past_v = None

            if cache is not None:
                past_k, past_v = cache[i]

            x, k, v = block(
                x,
                past_k=past_k,
                past_v=past_v,
            )

            new_cache.append((k, v))

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss, new_cache