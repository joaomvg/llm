# sft/loss.py
import torch
import torch.nn.functional as F


def masked_ce_loss(
    logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor
) -> torch.Tensor:
    """
    logits: (B, T, V)
    labels: (B, T)
    loss_mask: (B, T) float {0,1}
    """
    B, T, V = logits.shape
    logits_2d = logits.view(B * T, V)
    labels_1d = labels.view(B * T)
    mask_1d = loss_mask.view(B * T)

    # token-wise CE
    ce = F.cross_entropy(logits_2d, labels_1d, reduction="none")  # (B*T,)
    ce = ce * mask_1d

    denom = mask_1d.sum().clamp(min=1.0)
    return ce.sum() / denom
