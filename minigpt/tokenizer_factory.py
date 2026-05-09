from minigpt import config
from minigpt.bpe_tokenizer import BPETokenizer
from minigpt.tokenizer import CharTokenizer


def build_tokenizer(text):
    if config.TOKENIZER_TYPE == "bpe":
        tokenizer = BPETokenizer(vocab_size=config.BPE_VOCAB_SIZE)
        tokenizer.train(text)
        return tokenizer

    if config.TOKENIZER_TYPE == "char":
        return CharTokenizer(text)

    raise ValueError(f"Unsupported tokenizer type: {config.TOKENIZER_TYPE}")