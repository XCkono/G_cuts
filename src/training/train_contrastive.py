# train_contrastive.py
import numpy as np
import torch, torch.nn as nn
import sys
from pathlib import Path

# 加入项目的上一级目录（src）
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.aggregators import MeanAggregator
from models.encoders import Encoder
from models.miners import mine_positive_pairs
from models.loss import nt_xent_loss, mse_loss, bce_adj_recon_loss, combined_loss
from models.decoders import Projector, AttrDecoder, AdjDecoder
from models.model import load_cora  # ← 这里导入你的数据加载函数

# ---------------------- #
#   辅助函数
# ---------------------- #
def to_embedding_layer(feat_data):
    N, F = feat_data.shape
    emb = nn.Embedding(N, F)
    emb.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    return emb


def sample_negs_per_anchor(N, anchors, pos_one, Q, adj_lists):
    """返回 neg_mat: (B, Q) 的节点索引矩阵，每个锚点 v 采 Q 个负样本"""
    B = anchors.numel()
    neg_mat = torch.empty((B, Q), dtype=torch.long)
    for i, v in enumerate(anchors.tolist()):
        forbid = set(adj_lists[int(v)]) | {int(v), int(pos_one[i])}
        chosen = []
        while len(chosen) < Q:
            u = np.random.randint(0, N)
            if u not in forbid:
                chosen.append(u)
        neg_mat[i] = torch.tensor(chosen, dtype=torch.long)
    return neg_mat


def unique_params(*modules):
    """去重参数以防 AdamW duplicate warning"""
    seen, out = set(), []
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                out.append(p)
    return out


# ---------------------- #
#   主训练函数
# ---------------------- #
def run_unsup_cora(
    out_dir="runs/cora_unsup",
    emb_dim=128, proj_dim=64,
    tau=0.2, lam=0.5,
    recon_mode="attrs",   # "attrs" | "adj"
    epochs=200, seed=42
):
    np.random.seed(seed)
    torch.manual_seed(seed)

    feat_data, labels, adj_lists = load_cora()
    N, F = feat_data.shape
    features = to_embedding_layer(feat_data)

    # 两层 SAGE 编码器（复用已有实现）
    agg1 = MeanAggregator(features, cuda=False, gcn=True)
    enc1 = Encoder(features, F, emb_dim, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes: enc1(nodes).t(), emb_dim, emb_dim, adj_lists, agg2,
                   base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 5
    enc2.num_samples = 5

    projector = Projector(emb_dim, 128, proj_dim)
    decoder = AttrDecoder(emb_dim, F) if recon_mode == "attrs" else AdjDecoder(emb_dim)

    # 优化器：仅 enc2 + projector + decoder（enc1 已包含）
    opt = torch.optim.AdamW(
        unique_params(enc2, projector, decoder),
        lr=1e-3, weight_decay=1e-4
    )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    x = features.weight.data.clone()  # (N,F)

    for ep in range(1, epochs + 1):
        opt.zero_grad()
        z = enc2(list(range(N))).t()  # (N, emb_dim)
        h = projector(z)              # (N, proj_dim)

        # === 正样本挖掘 ===
        src, pos = mine_positive_pairs(x, adj_lists, r=2, theta_J=0.35, theta_F=0.55, P=8)
        if src.numel() == 0:
            print("[WARN] 无正样本，请检查阈值/数据路径")
            break

        # 每个锚仅取一个正样本（P=1）
        anchors, pos_one, seen = [], [], set()
        for s, p in zip(src.tolist(), pos.tolist()):
            if s not in seen:
                anchors.append(s)
                pos_one.append(p)
                seen.add(s)
        anchors = torch.tensor(anchors, dtype=torch.long)
        pos_one = torch.tensor(pos_one, dtype=torch.long)
        B = anchors.numel()

        # === 负样本采样 (B, Q)
        neg_mat = sample_negs_per_anchor(N, anchors, pos_one, Q=64, adj_lists=adj_lists)

        # === 组装损失输入 ===
        z_anchor = h[anchors]               # (B, d)
        z_pos = h[pos_one].unsqueeze(1)     # (B, 1, d)
        z_neg = h[neg_mat]                  # (B, Q, d)

        l_con = nt_xent_loss(z_anchor, z_pos, z_neg, tau=tau)

        # === 重构损失 ===
        if recon_mode == "attrs":
            x_hat = decoder(z)
            l_rec = mse_loss(x_hat, x)
        else:
            # 正边：邻接中的边
            pos_edges = []
            for v in range(N):
                for u in adj_lists[int(v)]:
                    if v < u:
                        pos_edges.append([v, u])
            pos_edges = torch.tensor(pos_edges, dtype=torch.long)
            # 负边：随机采样
            neg_edges = torch.randint(0, N, (len(pos_edges), 2), dtype=torch.long)
            pos_logit, neg_logit = decoder(z, pos_edges, neg_edges)
            l_rec = bce_adj_recon_loss(pos_logit, neg_logit)

        # === 联合优化 ===
        loss = combined_loss(l_con, l_rec, lam=lam)
        loss.backward()
        opt.step()

        if ep % 10 == 0:
            print(f"[Ep {ep:03d}]  Lc={l_con.item():.4f}  Lr={l_rec.item():.4f}  L={loss.item():.4f}")

    # === 保存嵌入，供阶段B使用 ===
    with torch.no_grad():
        z = enc2(list(range(N))).t().cpu().numpy()
    np.save(Path(out_dir) / "embeddings.npy", z)
    np.save(Path(out_dir) / "labels.npy", labels.squeeze())
    print(f"[DONE] embeddings saved to {out_dir}")


if __name__ == "__main__":
    run_unsup_cora()
