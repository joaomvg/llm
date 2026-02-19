# gpt_from_scratch/data.py
import random
import torch
from torch.utils.data import IterableDataset


class TextChunkDataset(IterableDataset):
    """
    Streams text files, tokenizes, and yields fixed-length sequences.
    """
    def __init__(self, files, tokenizer, ctx_len=256, shuffle_files=True):
        super().__init__()
        self.files = list(files)
        self.tok = tokenizer
        self.ctx_len = ctx_len
        self.shuffle_files = shuffle_files

    def parse_file(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    def __iter__(self):
        files = self.files[:]
        if self.shuffle_files:
            random.shuffle(files)

        buffer = []
        for fp in files:
            for text in self.parse_file(fp):
                ids = self.tok.encode(text)
                buffer.extend(ids + [self.tok.eos_id])

                while len(buffer) >= self.ctx_len + 1:
                    chunk = buffer[: self.ctx_len + 1]
                    buffer = buffer[self.ctx_len + 1 :]  # simple non-overlap

                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y