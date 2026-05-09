import torch

from minigpt.model import MiniGPT
from minigpt.tokenizer_loader import load_tokenizer


def generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=300,
    temperature=0.8,
    top_k=10,
):
    model.eval()

    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        x_cond = x[:, -model.position_embedding.num_embeddings:]

        logits, _, _ = model(x_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature

        values, _ = torch.topk(logits, top_k)
        min_value = values[:, -1].unsqueeze(1)

        logits = torch.where(
            logits < min_value,
            torch.full_like(logits, float("-inf")),
            logits,
        )

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


def main():
    checkpoint = torch.load("model.pt", map_location="cpu")

    tokenizer = load_tokenizer(checkpoint)

    model = MiniGPT(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        block_size=checkpoint["block_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    print(
        generate(
            model,
            tokenizer,
            prompt="First",
            max_new_tokens=300,
            temperature=0.8,
            top_k=10,
        )
    )


if __name__ == "__main__":
    main()