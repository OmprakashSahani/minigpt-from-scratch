import time

import torch

from minigpt.model import MiniGPT
from minigpt.generate import LoadedTokenizer


def benchmark_no_cache(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
):
    model.eval()

    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)

    start = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x_cond = x[:, -model.position_embedding.num_embeddings:]

            logits, _, _ = model(x_cond)

            logits = logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

            x = torch.cat([x, next_id], dim=1)

    end = time.perf_counter()

    return end - start


def benchmark_with_cache(
    model,
    tokenizer,
    prompt,
    max_new_tokens=100,
):
    model.eval()

    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)

    cache = None

    start = time.perf_counter()

    with torch.no_grad():
        for step in range(max_new_tokens):
            if cache is None:
                x_input = x
            else:
                x_input = x[:, -1:]

            logits, _, cache = model(
                x_input,
                cache=cache,
            )

            logits = logits[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

            x = torch.cat([x, next_id], dim=1)

    end = time.perf_counter()

    return end - start


def main():
    checkpoint = torch.load("model.pt", map_location="cpu")

    tokenizer = LoadedTokenizer(
        checkpoint["stoi"],
        checkpoint["itos"],
    )

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        block_size=checkpoint["block_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    prompt = "First"

    normal_time = benchmark_no_cache(
        model,
        tokenizer,
        prompt,
    )

    cache_time = benchmark_with_cache(
        model,
        tokenizer,
        prompt,
    )

    print("-" * 60)
    print("KV Cache Benchmark")
    print("-" * 60)

    print(f"Without KV Cache : {normal_time * 1000:.3f} ms")
    print(f"With KV Cache    : {cache_time * 1000:.3f} ms")

    speedup = normal_time / cache_time

    print(f"Speedup           : {speedup:.2f}x")


if __name__ == "__main__":
    main()
