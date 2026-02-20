# sft/data.py
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset


@dataclass
class SFTExample:
    input_ids: List[int]
    labels: List[int]
    loss_mask: List[int]  # 1 where we compute loss, 0 where we ignore


class SFTJsonlDataset(Dataset):
    """
    Reads JSONL with {"prompt": ..., "response": ...}.
    Builds a single sequence: prompt + response (+ eos),
    and masks loss so ONLY response tokens contribute.
    """
    def __init__(
        self,
        path: str,
        tokenizer,
        eos_id: int,
        max_len: int = 1024,
        add_eos: bool = True,
    ):
        self.path = path
        self.tok = tokenizer
        self.eos_id = eos_id
        self.max_len = max_len
        self.add_eos = add_eos

        # Load index of file offsets for random access (simple, ok for mid-size)
        self._offsets = []
        with open(self.path, "rb") as f:
            off = 0
            for line in f:
                self._offsets.append(off)
                off += len(line)

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx: int) -> SFTExample:
        off = self._offsets[idx]
        with open(self.path, "rb") as f:
            f.seek(off)
            line = f.readline().decode("utf-8")
        obj = json.loads(line)

        prompt = obj["prompt"]
        response = obj["response"]

        prompt_ids = self.tok.encode(prompt)
        resp_ids = self.tok.encode(response)

        if self.add_eos:
            resp_ids = resp_ids + [self.eos_id]

        # Build full sequence
        input_ids = (prompt_ids + resp_ids)[: self.max_len]
        # Labels are next-token targets
        labels = input_ids[1:] + [self.eos_id]  # last label doesn't matter much; keep consistent

        # Loss mask: ignore prompt region, learn only response region
        # We want mask aligned with labels (positions where we predict next token).
        # If prompt length = P, response length = R, then:
        # tokens positions: 0..P-1 are prompt, P..P+R-1 are response
        # labels correspond to predicting token at position t+1
        # We typically ignore predictions that target prompt tokens.
        P = min(len(prompt_ids), len(input_ids))  # truncated prompt length
        mask = [0] * (P - 1)  # predictions within prompt (t < P-1) => targets are prompt tokens
        # From t = P-1 onward, target is either response token 0,1,... so we include
        remaining = len(labels) - len(mask)
        mask += [1] * max(0, remaining)

        # Ensure same length
        mask = mask[: len(labels)]
        if len(mask) < len(labels):
            mask += [1] * (len(labels) - len(mask))

        return SFTExample(input_ids=input_ids, labels=labels, loss_mask=mask)