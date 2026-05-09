from minigpt.bpe_tokenizer import BPETokenizer
from minigpt.tokenizer import CharTokenizer


def load_tokenizer(checkpoint):
    tokenizer_type = checkpoint["tokenizer_type"]

    if tokenizer_type == "char":
        tokenizer = CharTokenizer("")

        tokenizer.stoi = checkpoint["stoi"]
        tokenizer.itos = checkpoint["itos"]

        return tokenizer

    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer()

        tokenizer.vocab = checkpoint["vocab"]
        tokenizer.inverse_vocab = checkpoint["inverse_vocab"]

        return tokenizer

    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")