class BPETokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def get_stats(self, tokens):
        pairs = {}

        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pairs[pair] = pairs.get(pair, 0) + 1

        return pairs

    def merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0

        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def train(self, text):
        tokens = list(text)

        current_vocab = set(tokens)
        next_token_id = 0

        self.vocab = {token: idx for idx, token in enumerate(sorted(current_vocab))}
        next_token_id = len(self.vocab)

        while len(self.vocab) < self.vocab_size:
            pairs = self.get_stats(tokens)

            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            new_token = "".join(best_pair)

            tokens = self.merge_pair(tokens, best_pair, new_token)

            if new_token not in self.vocab:
                self.vocab[new_token] = next_token_id
                next_token_id += 1
                self.merges[best_pair] = new_token

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, text):
        tokens = list(text)

        for pair, new_token in self.merges.items():
            tokens = self.merge_pair(tokens, pair, new_token)

        return [self.vocab[token] for token in tokens if token in self.vocab]

    def decode(self, ids):
        tokens = [self.inverse_vocab[int(i)] for i in ids]
        return "".join(tokens)