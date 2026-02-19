# sample.py
import torch
from gpt_from_scratch.model import GPT, GPTConfig
from gpt_from_scratch.tokenizer import CharTokenizer

@torch.no_grad()
def generate(model, idx, max_new_tokens=200, temperature=1.0, top_k=40):
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.ctx_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-6)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

    return idx

def load_ckpt(path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    tok = CharTokenizer(vocab=[])
    tok.itos = ckpt["tok_itos"]
    tok.stoi = {s:i for i,s in enumerate(tok.itos)}
    tok.unk_id = tok.stoi["<unk>"]
    tok.eos_id = tok.stoi["<eos>"]

    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model, tok

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = load_ckpt("./checkpoints/ckpt_001000.pt", device=device)

    prompt = "Once upon a time"
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    out = generate(model, idx, max_new_tokens=300, temperature=0.9, top_k=50)
    print(tok.decode(out[0].tolist()))