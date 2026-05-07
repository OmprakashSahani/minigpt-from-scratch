import torch

from minigpt.model import MiniGPT
from minigpt.tokenizer import CharTokenizer


def int_to_binary(n, bits=8):
    return format(n, f"0{bits}b")


def main():
    prompt = "What is LLM?"

    text = open("data/input.txt").read()
    tokenizer = CharTokenizer(text + prompt)

    checkpoint = torch.load("model.pt", map_location="cpu")

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=checkpoint["embed_dim"],
        block_size=checkpoint["block_size"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    )

    print("=" * 60)
    print("MiniGPT Pipeline Trace")
    print("=" * 60)

    print("\n1. Raw text input:")
    print(prompt)

    print("\n2. Character tokens:")
    print(list(prompt))

    token_ids = tokenizer.encode(prompt)

    print("\n3. Token IDs:")
    print(token_ids)

    print("\n4. Token IDs as binary:")
    for ch, token_id in zip(prompt, token_ids):
        print(f"{repr(ch):>4} -> {token_id:>3} -> {int_to_binary(token_id)}")

    x = torch.tensor([token_ids], dtype=torch.long)

    print("\n5. Input tensor shape:")
    print(x.shape)

    with torch.no_grad():
        token_embeddings = model.token_embedding(x)

    print("\n6. Token embedding shape:")
    print(token_embeddings.shape)

    with torch.no_grad():
        logits, _ = model(x)

    print("\n7. Model output logits shape:")
    print(logits.shape)

    last_logits = logits[:, -1, :]
    predicted_id = torch.argmax(last_logits, dim=-1).item()

    print("\n8. Predicted next token ID:")
    print(predicted_id)

    print("\n9. Predicted next token as binary:")
    print(int_to_binary(predicted_id))

    print("\n10. Decoded predicted character:")
    print(repr(tokenizer.decode([predicted_id])))

    generated_ids = token_ids + [predicted_id]

    print("\n11. Text after one generated token:")
    print(tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
