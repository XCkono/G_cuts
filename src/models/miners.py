# miners.py
import numpy as np
import torch
import torch.nn.functional as F

def cosine_sim(x, y):
    x = F.normalize(x, dim=-1); y = F.normalize(y, dim=-1)
    return (x * y).sum(-1)

def jaccard_for_pair(adj_lists, u, v):
    Nu = adj_lists[int(u)]; Nv = adj_lists[int(v)]
    inter = len(Nu & Nv); union = len(Nu | Nv) if (Nu or Nv) else 1
    return inter / union

def k_hop_candidates(adj_lists, v, k=2):
    """返回 <=k-hop 的候选集合（不含自身）"""
    frontier = {int(v)}; visited = {int(v)}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt |= set(adj_lists[int(u)])
        nxt -= visited
        visited |= nxt; frontier = nxt
    visited.remove(int(v))
    return visited

def mine_positive_pairs(x, adj_lists, r=2, theta_J=0.35, theta_F=0.55, P=8):
    """
    返回 (src_idx, pos_idx)，二者等长的一维张量
    近: <=r hop；像: Jaccard>=theta_J 或 cosine>=theta_F
    """
    N = x.size(0)
    src, pos = [], []
    with torch.no_grad():
        for v in range(N):
            cand = list(k_hop_candidates(adj_lists, v, k=r))
            if not cand: continue
            # 属性相似（若有属性）
            xv = x[v].unsqueeze(0).expand(len(cand), -1)
            xc = x[torch.tensor(cand)]
            cos = cosine_sim(xv, xc).cpu().numpy()
            kept = []
            for j, u in enumerate(cand):
                jac = jaccard_for_pair(adj_lists, v, u)
                if jac >= theta_J or cos[j] >= theta_F:
                    kept.append(u)
            kept = kept[:P]
            for u in kept:
                src.append(v); pos.append(u)
    if not src:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.tensor(src, dtype=torch.long), torch.tensor(pos, dtype=torch.long)

def sample_negative_pairs(N, pos_pairs, Q=64, adj_lists=None):
    """远负：>=3-hop 或 双低相似（可简化为随机负采+去重）"""
    src_pos, pos = pos_pairs
    src_neg, dst_neg = [], []
    for v in range(N):
        forbid = set(pos[src_pos == v].tolist() if src_pos.numel() else [])
        forbid.add(v)
        # 简单随机负采
        chosen = []
        while len(chosen) < Q and len(chosen) < N - len(forbid):
            u = np.random.randint(0, N)
            if u not in forbid:
                chosen.append(u)
        if chosen:
            src_neg.append(torch.full((len(chosen),), v, dtype=torch.long))
            dst_neg.append(torch.tensor(chosen, dtype=torch.long))
    if not src_neg:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    return torch.cat(src_neg), torch.cat(dst_neg)
