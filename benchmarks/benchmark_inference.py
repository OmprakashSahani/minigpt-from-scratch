import time

import torch

from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer


def benchmark(model, x, num_runs=100, warmup_runs=20):
    model.eval()

    times = []

    with torch.no_grad():
        for _ in range(warmup_runs):
            model(x)

        for _ in range(num_runs):
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()

            times.append(end - start)

    avg_time = sum(times) / len(times)
    return avg_time


def main():
    text = open("data/input.txt").read()
    tokenizer = CharTokenizer(text)

    checkpoint = torch.load("model.pt", map_location="cpu")

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        block_size=checkpoint["block_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    sequence_lengths = [4, 8, 16]

    print("-" * 50)
    print("MiniGPT Inference Benchmark")
    print("-" * 50)

    for seq_len in sequence_lengths:
        sample = text[:seq_len]

        x = torch.tensor(
            [tokenizer.encode(sample)],
            dtype=torch.long
        )

        avg_time = benchmark(model, x)

        print(
            f"Sequence Length: {seq_len:2d} | "
            f"Average Latency: {avg_time * 1000:.3f} ms"
        )


if __name__ == "__main__":
    main()