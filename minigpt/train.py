import torch

from minigpt import config
from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer
from minigpt.utils import (
    count_parameters,
    count_trainable_parameters,
    get_device,
)


def get_batch(split_data, block_size, batch_size):
    ix = torch.randint(
        0,
        len(split_data) - block_size - 1,
        (batch_size,),
        device=split_data.device,
    )

    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + block_size + 1] for i in ix])

    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=20):
    model.eval()
    out = {}

    for split, split_data in [("train", train_data), ("val", val_data)]:
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(split_data, block_size, batch_size)
            _, loss, _ = model(x, y)
            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def main():
    text = open("data/input.txt").read()
    tokenizer = CharTokenizer(text)

    device = get_device(config.DEVICE)
    print(f"Using device: {device}")

    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
        device=device,
    )

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    block_size = config.BLOCK_SIZE
    embed_dim = config.EMBED_DIM
    batch_size = config.BATCH_SIZE
    steps = config.STEPS
    lr = config.LEARNING_RATE
    num_heads = config.NUM_HEADS
    num_layers = config.NUM_LAYERS

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        num_heads=num_heads,
        num_layers=num_layers,
    )

    model = model.to(device)

    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loss_history = []
    val_loss_history = []

    for step in range(steps):
        x, y = get_batch(train_data, block_size, batch_size)

        _, loss, _ = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=block_size,
                batch_size=batch_size,
            )

            train_loss_history.append(losses["train"])
            val_loss_history.append(losses["val"])

            print(
                f"step {step}, "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": embed_dim,
            "block_size": block_size,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        "model.pt",
    )

    print("Saved model to model.pt")

    with open("train_loss_history.txt", "w") as f:
        for value in train_loss_history:
            f.write(f"{value}\n")

    with open("val_loss_history.txt", "w") as f:
        for value in val_loss_history:
            f.write(f"{value}\n")

    print("Saved train loss history to train_loss_history.txt")
    print("Saved validation loss history to val_loss_history.txt")


if __name__ == "__main__":
    main()