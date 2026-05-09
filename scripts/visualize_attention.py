import matplotlib.pyplot as plt
import torch

from minigpt.generate import LoadedTokenizer
from minigpt.model import MiniGPT


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
    model.eval()

    prompt = "First Citizen"
    token_ids = tokenizer.encode(prompt)

    x = torch.tensor([token_ids], dtype=torch.long)

    with torch.no_grad():
        model(x)

    first_block = model.blocks[0]
    attention = first_block.attention.last_attention_weights

    # Shape: [batch, heads, query_tokens, key_tokens]
    head_index = 0
    attn = attention[0, head_index].cpu().numpy()

    tokens = list(prompt)

    plt.figure(figsize=(8, 6))
    plt.imshow(attn)
    plt.xticks(range(len(tokens)), tokens)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.title("MiniGPT Attention Heatmap - Layer 0, Head 0")
    plt.colorbar(label="Attention Weight")

    plt.savefig("attention_heatmap.png", dpi=200, bbox_inches="tight")
    print("Saved attention heatmap to attention_heatmap.png")


if __name__ == "__main__":
    main()