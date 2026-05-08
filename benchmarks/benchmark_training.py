import time

import torch

from minigpt import config
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer
from minigpt.utils import get_device


def get_batch(data, block_size, batch_size):
    ix = torch.randint(
        0,
        len(data) - block_size - 1,
        (batch_size,),
        device=data.device,
    )

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x, y


def main():
    device = get_device(config.DEVICE)
    print(f"Using device: {device}")

    text = open("data/input.txt").read()
    tokenizer = CharTokenizer(text)

    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
        device=device,
    )

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.EMBED_DIM,
        block_size=config.BLOCK_SIZE,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    warmup_steps = 10
    benchmark_steps = 50

    for _ in range(warmup_steps):
        x, y = get_batch(data, config.BLOCK_SIZE, config.BATCH_SIZE)
        _, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()

    for _ in range(benchmark_steps):
        x, y = get_batch(data, config.BLOCK_SIZE, config.BATCH_SIZE)
        _, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    total_time = end - start
    avg_step_time = total_time / benchmark_steps
    tokens_per_step = config.BATCH_SIZE * config.BLOCK_SIZE
    tokens_per_second = tokens_per_step / avg_step_time

    print("-" * 60)
    print("MiniGPT Training Throughput Benchmark")
    print("-" * 60)
    print(f"Device             : {device}")
    print(f"Batch size         : {config.BATCH_SIZE}")
    print(f"Block size         : {config.BLOCK_SIZE}")
    print(f"Tokens per step    : {tokens_per_step}")
    print(f"Average step time  : {avg_step_time * 1000:.3f} ms")
    print(f"Tokens/sec         : {tokens_per_second:.2f}")

    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak GPU memory    : {peak_memory:.2f} MB")


if __name__ == "__main__":
    main()