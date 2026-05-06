import torch

from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer


def main():
    text = open("data/input.txt").read()
    tokenizer = CharTokenizer(text)

    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    block_size = 32
    embed_dim = 64
    batch_size = 8
    steps = 3000
    lr = 1e-3

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

        logits, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step}, loss {loss.item():.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": tokenizer.vocab_size,
            "embed_dim": embed_dim,
            "block_size": block_size,
            "stoi": tokenizer.stoi,
            "itos": tokenizer.itos,
        },
        "model.pt",
    )

    print("Saved model to model.pt")

if __name__ == "__main__":
    main()
