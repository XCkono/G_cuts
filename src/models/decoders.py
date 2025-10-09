# decoders.py
import torch
import torch.nn as nn

class Projector(nn.Module):
    """ z -> 128 -> 64, L2 在损失里做 """
    def __init__(self, in_dim, hidden=128, out=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out))
    def forward(self, z): return self.net(z)

class AttrDecoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))
    def forward(self, z): return self.net(z)

class AdjDecoder(nn.Module):
    """ 双线性打分器 """
    def __init__(self, in_dim):
        super().__init__()
        self.score = nn.Bilinear(in_dim, in_dim, 1)
    def pair_score(self, z_src, z_dst): return self.score(z_src, z_dst).squeeze(-1)
    def forward(self, z, pos_edges, neg_edges):
        pos = self.pair_score(z[pos_edges[:,0]], z[pos_edges[:,1]])
        neg = self.pair_score(z[neg_edges[:,0]], z[neg_edges[:,1]])
        return pos, neg
