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
    use_amp = config.USE_AMP and device.type == "cuda"

    print(f"Using device: {device}")
    print(f"Using AMP: {use_amp}")

    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long,
        device=device,
    )

    split_idx = int(0.9 * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.EMBED_DIM,
        block_size=config.BLOCK_SIZE,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
    ).to(device)

    print(f"Total parameters: {count_parameters(model):,}")
    print(f"Trainable parameters: {count_trainable_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    train_loss_history = []
    val_loss_history = []

    for step in range(config.STEPS):
        x, y = get_batch(train_data, config.BLOCK_SIZE, config.BATCH_SIZE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=use_amp):
            _, loss, _ = model(x, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=config.BLOCK_SIZE,
                batch_size=config.BATCH_SIZE,
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
            "embed_dim": config.EMBED_DIM,
            "block_size": config.BLOCK_SIZE,
            "num_heads": config.NUM_HEADS,
            "num_layers": config.NUM_LAYERS,
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