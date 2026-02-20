# sft/collate.py
from typing import List, Dict
import torch
from .data import SFTExample


def sft_collate(batch: List[SFTExample], pad_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(len(ex.input_ids) for ex in batch)

    input_ids = []
    labels = []
    loss_mask = []
    attn_mask = []

    for ex in batch:
        L = len(ex.input_ids)
        pad = max_len - L

        input_ids.append(ex.input_ids + [pad_id] * pad)
        labels.append(ex.labels + [pad_id] * pad)        # label padding will be masked anyway
        loss_mask.append(ex.loss_mask + [0] * pad)       # don't compute loss on padding
        attn_mask.append([1] * L + [0] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.float32),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
    }