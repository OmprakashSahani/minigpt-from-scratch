import torch

from minigpt.model import MiniGPT


class LoadedTokenizer:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = {int(k): v for k, v in itos.items()}

    def encode(self, text):
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


def generate(model, tokenizer, prompt, max_new_tokens=100):
    model.eval()

    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.position_embedding.num_embeddings:]

        logits, _ = model(x_cond)
        logits = logits[:, -1, :]

        next_id = torch.argmax(logits, dim=-1, keepdim=True)

        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


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
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(generate(model, tokenizer, prompt="machine", max_new_tokens=100))


if __name__ == "__main__":
    main()
