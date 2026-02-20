# run_sft.py
import torch
from gpt.model import GPT, GPTConfig
from gpt.tokenizer import CharTokenizer
from gpt.sft.train_sft import train_sft

# load your tokenizer / vocab
tok = CharTokenizer.train_from_files(["./data/pretrain.txt"], max_chars=3000)

ckpt = torch.load("checkpoints/pretrain_final.pt", map_location="cpu")

cfg = GPTConfig(**ckpt["cfg"])
model = GPT(cfg)
model.load_state_dict(ckpt["model"])

# For CharTokenizer we don't have pad_id; choose something (often eos or unk).
pad_id = tok.unk_id
eos_id = tok.eos_id

train_sft(
    model=model,
    tokenizer=tok,
    train_jsonl="./data/sft.jsonl",
    pad_id=pad_id,
    eos_id=eos_id,
    max_len=cfg.ctx_len,
    batch_size=32,
    lr=2e-5,
    grad_accum=2,
    max_steps=2000,
)