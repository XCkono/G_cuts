# losses.py
import torch
import torch.nn.functional as F

def nt_xent_loss(z_anchor, z_pos, z_neg, tau=0.2):
    z_a = F.normalize(z_anchor, dim=-1)
    z_p = F.normalize(z_pos, dim=-1)   # (B, P, d)
    z_n = F.normalize(z_neg, dim=-1)   # (B, Q, d)

    pos_sim = (z_p * z_a.unsqueeze(1)).sum(-1) / tau   # (B, P)
    neg_sim = (z_n * z_a.unsqueeze(1)).sum(-1) / tau   # (B, Q)

    max_all = torch.maximum(pos_sim.max(1, keepdim=True).values,
                            neg_sim.max(1, keepdim=True).values)
    pos_exp = torch.exp(pos_sim - max_all)
    neg_exp = torch.exp(neg_sim - max_all)
    pos_sum = pos_exp.sum(1); neg_sum = neg_exp.sum(1)
    loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-9)).mean()
    return loss

def mse_loss(x_hat, x):
    return F.mse_loss(x_hat, x)

def bce_adj_recon_loss(pos_logits, neg_logits):
    pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    return 0.5 * (pos + neg)

def combined_loss(lc, lr, lam=0.5):
    return lam * lc + (1.0 - lam) * lr
