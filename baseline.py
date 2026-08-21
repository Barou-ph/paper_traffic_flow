"""
Baseline Models — Spec tuần 1 (Thành viên B)
=============================================
Interface chuẩn:
    forward(X, Z=None, time_idx=None, A_static=None)
    X       : [B, N, in_channels]  in_channels = T_in*F = 12*9 = 108
    Z       : [N, K]
    time_idx: [B]
    A_static: [N, N]
    → output: [B, N, T_out=3]

Models:
  1. LSTM     — temporal only, bỏ qua graph
  2. GCN-GRU  — GCN tĩnh + GRU
  3. STGCN    — Spatio-Temporal GCN (Yu et al. 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json, os, time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_model import (
    NodeCongestionDataset,
    load_dataset,
    compute_metrics,
    eval_epoch,
    CFG,
)

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────
# BASELINE 1: LSTM
# Chứng minh spatial structure quan trọng
# ─────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, hidden_dim=128, **kwargs):
        super().__init__()
        # in_channels = T_in * F (đã flatten)
        self.lstm = nn.LSTM(
            num_nodes * in_channels,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.fc = nn.Linear(hidden_dim, num_nodes * out_channels)
        self.num_nodes = num_nodes
        self.out_channels = out_channels

    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """X: [B, N, in_channels]"""
        B, N, C = X.shape
        x = X.reshape(B, 1, N * C)  # [B, 1, N*C]
        h, _ = self.lstm(x)  # [B, 1, hid]
        out = self.fc(h[:, -1, :])  # [B, N*T_out]
        return out.reshape(B, N, self.out_channels)


# ─────────────────────────────────────────────
# BASELINE 2: GCN-GRU
# Đánh giá adaptive adj + node-specific weight
# ─────────────────────────────────────────────
class StandardGCNLayer(nn.Module):
    """GCN chuẩn: H^{l+1} = σ(D^{-1/2} Ã D^{-1/2} H W)"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A):
        # Normalize A
        D_inv = torch.diag(1.0 / (A.sum(1) + 1e-8))
        A_hat = D_inv @ (A + torch.eye(A.shape[0], device=A.device))
        H_agg = torch.einsum("nm,bmd->bnd", A_hat, H)
        return self.norm(F.relu(self.W(H_agg)))


class GCNGRUBaseline(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, hidden_dim=32, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.gcn1 = StandardGCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = StandardGCNLayer(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_channels)
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """X: [B, N, in_channels]"""
        B, N, _ = X.shape
        H = self.input_proj(X)  # [B, N, hid]
        A = A_static if A_static is not None else torch.eye(N, device=X.device)
        H = self.gcn1(H, A)
        H = self.gcn2(H, A)  # [B, N, hid]
        # GRU theo từng nút
        H_r = H.reshape(B * N, 1, self.hidden_dim)
        H_g, _ = self.gru(H_r)  # [B*N, 1, hid]
        H_out = H_g[:, -1].reshape(B, N, self.hidden_dim)
        return self.fc(H_out)  # [B, N, T_out]


# ─────────────────────────────────────────────
# BASELINE 3: STGCN (Yu et al. 2018)
# ST-Conv Block: Temporal → Spatial → Temporal
# ─────────────────────────────────────────────
class TemporalGatedConv(nn.Module):
    """Temporal Gated Conv: output = tanh(W1*x) ⊙ σ(W2*x)"""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv_tanh = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv_sig = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, T, C]
        xt = x.transpose(1, 2)  # [B, C, T]
        out = torch.tanh(self.conv_tanh(xt)) * torch.sigmoid(
            self.conv_sig(xt)
        )  # [B, C, T]
        return self.norm(out.transpose(1, 2))  # [B, T, C]


class STConvBlock(nn.Module):
    def __init__(self, n_nodes, channels):
        super().__init__()
        self.tconv1 = TemporalGatedConv(channels)
        self.gcn = StandardGCNLayer(channels, channels)
        self.tconv2 = TemporalGatedConv(channels)
        self.n_nodes = n_nodes

    def forward(self, H, A):
        # H: [B, T, N, C]
        B, T, N, C = H.shape
        # Temporal 1: per node
        H_r = H.permute(0, 2, 1, 3).reshape(B * N, T, C)
        H_t1 = self.tconv1(H_r).reshape(B, N, T, C).permute(0, 2, 1, 3)
        # Spatial: per timestep
        outs = [self.gcn(H_t1[:, t], A) for t in range(T)]
        H_s = torch.stack(outs, dim=1)  # [B, T, N, C]
        # Temporal 2: per node
        H_r2 = H_s.permute(0, 2, 1, 3).reshape(B * N, T, C)
        H_t2 = self.tconv2(H_r2).reshape(B, N, T, C).permute(0, 2, 1, 3)
        return H_t2


class STGCNBaseline(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, hidden_dim=32, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.st_block1 = STConvBlock(num_nodes, hidden_dim)
        self.st_block2 = STConvBlock(num_nodes, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out_channels)
        self.hidden_dim = hidden_dim

    def forward(self, X, Z=None, time_idx=None, A_static=None):
        """X: [B, N, in_channels]"""
        B, N, _ = X.shape
        A = A_static if A_static is not None else torch.eye(N, device=X.device)
        H = self.input_proj(X)  # [B, N, hid]
        H = H.unsqueeze(1)  # [B, 1, N, hid] — T=1
        H = self.st_block1(H, A)
        H = self.st_block2(H, A)
        H_out = H[:, -1, :, :]  # [B, N, hid]
        return self.fc(H_out)  # [B, N, T_out]


# ─────────────────────────────────────────────
# TRAIN BASELINE
# ─────────────────────────────────────────────
def train_baseline(
    model, name, ds_train, ds_val, ds_test, Z_t, A_t, y_mean, y_std, device
):
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    ltr = DataLoader(ds_train, CFG["batch_size"], shuffle=True, drop_last=True)
    lva = DataLoader(ds_val, CFG["batch_size"], shuffle=False)
    lte = DataLoader(ds_test, CFG["batch_size"], shuffle=False)

    best_val, patience, best_state = float("inf"), 0, None

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        tr = 0
        for Xf, y, ti in ltr:
            Xf, y, ti = Xf.to(device), y.to(device), ti.to(device)
            opt.zero_grad()
            yh = model(Xf, Z_t, ti, A_t)
            loss = crit(yh, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item()
        tr /= len(ltr)
        va, _, _ = eval_epoch(model, lva, crit, Z_t, A_t, device)
        sch.step(va)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train={tr:.4f} | Val={va:.4f}")
        if va < best_val:
            best_val = va
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CFG["patience"]:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    _, tp, tt = eval_epoch(model, lte, crit, Z_t, A_t, device)
    m = compute_metrics(tt, tp, y_mean, y_std)
    print(
        f"\n  📊 {name} → MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} "
        f"| MAPE={m['MAPE']:.2f}% | MAE_mz={m['MAE_multi_zone']:.4f}"
    )
    return m


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = CFG["device"]
    ds_train, ds_val, ds_test, Z, gps, A_phys, y_mean, y_std, meta = load_dataset()

    Z_t = torch.FloatTensor(Z).to(device)
    A_t = torch.FloatTensor(A_phys).to(device)

    N = len(meta["node_ids"])
    in_channels = CFG["T_in"] * CFG["F_feat"]  # 108
    out_channels = CFG["T_out"]  # 3

    baselines = {
        "LSTM": LSTMBaseline(N, in_channels, out_channels),
        "GCN-GRU": GCNGRUBaseline(N, in_channels, out_channels),
        "STGCN": STGCNBaseline(N, in_channels, out_channels),
    }

    all_metrics = {}
    for name, model in baselines.items():
        all_metrics[name] = train_baseline(
            model, name, ds_train, ds_val, ds_test, Z_t, A_t, y_mean, y_std, device
        )

    # Load ZAH-GNN nếu đã train
    if os.path.exists("results/metrics.json"):
        with open("results/metrics.json", encoding="utf-8") as f:
            r = json.load(f)
        all_metrics["ZAH-GNN"] = r["metrics"]

    with open("results/baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Print bảng theo format spec
    print(f"\n{'='*70}")
    print(f"  📊 COMPARISON TABLE (Bảng 1 trong spec)")
    print(f"{'='*70}")
    print(
        f"  {'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE(%)':>10} {'MAE_mz':>12} {'Params':>10}"
    )
    print(f"  {'-'*65}")
    param_counts = {
        "LSTM": sum(
            p.numel() for p in LSTMBaseline(N, in_channels, out_channels).parameters()
        ),
        "GCN-GRU": sum(
            p.numel() for p in GCNGRUBaseline(N, in_channels, out_channels).parameters()
        ),
        "STGCN": sum(
            p.numel() for p in STGCNBaseline(N, in_channels, out_channels).parameters()
        ),
        "ZAH-GNN": "—",
    }
    for name, m in all_metrics.items():
        marker = " ←" if name == "ZAH-GNN" else ""
        pc = param_counts.get(name, "—")
        print(
            f"  {name:<12} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} "
            f"{m['MAPE']:>10.2f} {m['MAE_multi_zone']:>12.4f} {str(pc):>10}{marker}"
        )

    # Bar chart
    models = list(all_metrics.keys())
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics_keys = ["MAE", "RMSE", "MAPE", "MAE_multi_zone"]
    colors = ["steelblue", "coral", "mediumseagreen", "mediumpurple"]
    for ax, key, col in zip(axes, metrics_keys, colors):
        vals = [all_metrics[m][key] for m in models]
        bars = ax.bar(models, vals, color=col, alpha=0.8, edgecolor="black")
        ax.set_title(key, fontweight="bold")
        ax.set_xticklabels(models, rotation=15, ha="right")
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.001,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    plt.suptitle(
        "Zone-Aware HetGNN vs Baselines — Node-level Congestion", fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Done! → results/baseline_metrics.json")


if __name__ == "__main__":
    main()
