# train.py
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from gpt.model import GPT, GPTConfig
from gpt.tokenizer import CharTokenizer
from gpt.data import TextChunkDataset


def save_ckpt(path, model, optim, step, cfg, tok):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "step": step,
        "cfg": cfg.__dict__,
        "tok_itos": tok.itos,
    }, path)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = "./data"  # put .txt files here
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")]
    assert files, "Put some .txt files into ./data"

    tok = CharTokenizer.train_from_files(files, max_chars=3000)

    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        ctx_len=256,
        n_layers=6,
        n_heads=6,
        d_model=384,
        d_ff=1536,
        dropout=0.1,
    )
    model = GPT(cfg).to(device)

    ds = TextChunkDataset(files, tok, ctx_len=cfg.ctx_len, shuffle_files=True)
    dl = DataLoader(ds, batch_size=32, num_workers=2)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    scaler = GradScaler(enabled=(device == "cuda"))

    model.train()
    step = 0
    log_every = 50
    ckpt_every = 1000

    t0 = time.time()
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)

        with autocast(enabled=(device == "cuda")):
            _, loss = model(x, y)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optim)
        scaler.update()

        if step % log_every == 0:
            dt = time.time() - t0
            print(f"step={step:06d} loss={loss.item():.4f} dt={dt:.1f}s")
            t0 = time.time()

        if step > 0 and step % ckpt_every == 0:
            save_ckpt(f"./checkpoints/ckpt_{step:06d}.pt", model, optim, step, cfg, tok)

        step += 1

if __name__ == "__main__":
    main()