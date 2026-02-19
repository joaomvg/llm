# gpt_from_scratch/tokenizer.py
from collections import Counter


class CharTokenizer:
    def __init__(self, vocab, eos_token="<eos>", unk_token="<unk>"):
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.itos = [unk_token, eos_token] + sorted(vocab)
        self.stoi = {s:i for i,s in enumerate(self.itos)}
        self.unk_id = self.stoi[unk_token]
        self.eos_id = self.stoi[eos_token]

    @classmethod
    def train_from_files(cls, files, max_chars=2000):
        cnt = Counter()
        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    cnt.update(list(line.rstrip("\n")))
        vocab = [c for c, _ in cnt.most_common(max_chars)]
        return cls(vocab=vocab)

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i < len(self.itos))