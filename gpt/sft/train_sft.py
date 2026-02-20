# sft/train_sft.py
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast

from gpt.sft.data import SFTJsonlDataset
from gpt.sft.collate import sft_collate
from gpt.sft.loss import masked_ce_loss


def save_ckpt(path, model, optim, step, extra: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        **extra
    }, path)


def train_sft(
    model,
    tokenizer,
    train_jsonl: str,
    *,
    pad_id: int,
    eos_id: int,
    device: str = None,
    max_len: int = 1024,
    batch_size: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.1,
    grad_accum: int = 4,
    max_steps: int = 2000,
    log_every: int = 25,
    ckpt_every: int = 500,
    out_dir: str = "./checkpoints_sft",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    ds = SFTJsonlDataset(train_jsonl, tokenizer, eos_id=eos_id, max_len=max_len, add_eos=True)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda b: sft_collate(b, pad_id=pad_id),
        drop_last=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=(device == "cuda"))

    step = 0
    micro = 0
    t0 = time.time()

    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        loss_mask = batch["loss_mask"].to(device)

        with autocast(device_type= 'cuda', enabled=(device == "cuda")):
            logits, _ = model(input_ids, targets=None)  # logits: (B,T,V)
            loss = masked_ce_loss(logits, labels, loss_mask) / grad_accum

        scaler.scale(loss).backward()
        micro += 1

        if micro % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad(set_to_none=True)

            if step % log_every == 0:
                dt = time.time() - t0
                print(f"sft step={step:06d} loss={(loss.item()*grad_accum):.4f} dt={dt:.1f}s")
                t0 = time.time()

            if step > 0 and step % ckpt_every == 0:
                save_ckpt(
                    os.path.join(out_dir, f"sft_{step:06d}.pt"),
                    model,
                    optim,
                    step,
                    extra={"max_len": max_len},
                )

            step += 1
            if step >= max_steps:
                break

    # final checkpoint
    save_ckpt(os.path.join(out_dir, "sft_final.pt"), model, optim, step, extra={"max_len": max_len})
    return model