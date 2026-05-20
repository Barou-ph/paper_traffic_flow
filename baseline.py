"""
Baseline Models để so sánh với AH-GNN
======================================
Models:
  1. LSTM         — temporal only, không dùng graph
  2. GCN_GRU      — GCN chuẩn (weight sharing) + GRU
  3. STGCN_simple — Spatial-Temporal GCN đơn giản

Dùng: python baseline.py
Kết quả lưu vào results/baseline_metrics.json
"""

import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import numpy as np
import json
import os
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from train_model import load_dataset, ETADataset, compute_metrics, eval_epoch, CFG

os.makedirs("results", exist_ok=True)


# ─────────────────────────────────────────────
# BASELINE 1: LSTM (không dùng graph)
# ─────────────────────────────────────────────
class LSTMBaseline(nn.Module):
    """Flatten tất cả node features → LSTM → predict ETA"""

    def __init__(self, n_nodes, n_edges, in_dim=9, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            n_nodes * in_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )
        self.fc = nn.Linear(hidden_dim, n_edges)

    def forward(self, X_seq, edge_index, edge_feats):
        B, T, N, F = X_seq.shape
        x = X_seq.reshape(B, T, N * F)  # [B, T, N*F]
        h, _ = self.lstm(x)  # [B, T, hid]
        out = self.fc(h[:, -1, :])  # [B, E]
        return torch_F.relu(out)


# ─────────────────────────────────────────────
# BASELINE 2: Standard GCN + GRU
# ─────────────────────────────────────────────
class StandardGCN(nn.Module):
    """GCN với weight sharing (không node-specific) — Eq.2 trong paper"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, H, A):
        # H_agg = A · H · W  (shared weight)
        H_agg = torch.einsum("nm,bmd->bnd", A, H)
        return self.norm(torch_F.relu(self.W(H_agg)))


class GCNGRUBaseline(nn.Module):
    def __init__(self, n_nodes, n_edges, in_dim=9, hidden_dim=32):
        super().__init__()
        self.gcn1 = StandardGCN(in_dim, hidden_dim)
        self.gcn2 = StandardGCN(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 + 3, 1)
        self.n_nodes = n_nodes
        self.hidden_dim = hidden_dim

        # Fixed adjacency (physical) — đọc từ file
        A = np.load("dataset/adj_physical.npy")
        # Normalize
        D = np.diag(A.sum(1) + 1e-8)
        D_inv = np.diag(1.0 / np.diag(D))
        A_norm = D_inv @ A
        self.register_buffer("A", torch.FloatTensor(A_norm))

    def forward(self, X_seq, edge_index, edge_feats):
        B, T, N, F = X_seq.shape
        outs = []
        for t in range(T):
            h = self.gcn1(X_seq[:, t], self.A)
            h = self.gcn2(h, self.A)
            outs.append(h)
        H = torch.stack(outs, dim=1)  # [B,T,N,hid]
        H_r = H.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        H_g, _ = self.gru(H_r)
        H_out = H_g[:, -1].reshape(B, N, self.hidden_dim)  # [B,N,hid]

        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        h_src = H_out[:, src, :]
        h_dst = H_out[:, dst, :]
        x = torch.cat([h_src, h_dst, edge_feats], dim=-1)
        return torch_F.relu(self.fc(x).squeeze(-1))


# ─────────────────────────────────────────────
# BASELINE 3: STGCN (simplified)
# ─────────────────────────────────────────────
class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # x: [B, T, channels]
        out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(torch_F.relu(out))


class STGCNBaseline(nn.Module):
    def __init__(self, n_nodes, n_edges, in_dim=9, hidden_dim=32):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.gcn = StandardGCN(hidden_dim, hidden_dim)
        self.tconv = TemporalConv(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2 + 3, 1)
        self.hidden_dim = hidden_dim

        A = np.load("dataset/adj_physical.npy")
        D_inv = np.diag(1.0 / (A.sum(1) + 1e-8))
        self.register_buffer("A", torch.FloatTensor(D_inv @ A))

    def forward(self, X_seq, edge_index, edge_feats):
        B, T, N, F = X_seq.shape
        X = self.input_proj(X_seq.reshape(B * T, N, F)).reshape(
            B, T, N, self.hidden_dim
        )

        # Spatial conv per timestep
        H = torch.stack(
            [self.gcn(X[:, t], self.A) for t in range(T)], dim=1
        )  # [B,T,N,hid]

        # Temporal conv per node
        H_r = H.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)
        H_t = self.tconv(H_r)  # [B*N, T, hid]
        H_out = H_t[:, -1].reshape(B, N, self.hidden_dim)  # [B,N,hid]

        src = edge_index[:, 0]
        dst = edge_index[:, 1]
        x = torch.cat([H_out[:, src], H_out[:, dst], edge_feats], dim=-1)
        return torch_F.relu(self.fc(x).squeeze(-1))


# ─────────────────────────────────────────────
# TRAIN BASELINE
# ─────────────────────────────────────────────
def train_baseline(
    model, name, ds_train, ds_val, ds_test, edge_index, y_mean, y_std, device
):
    print(f"\n{'='*50}")
    print(f"  Training: {name}")
    print(f"{'='*50}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.MSELoss()
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    loader_tr = DataLoader(ds_train, CFG["batch_size"], shuffle=True, drop_last=True)
    loader_va = DataLoader(ds_val, CFG["batch_size"], shuffle=False)
    loader_te = DataLoader(ds_test, CFG["batch_size"], shuffle=False)

    best_val = float("inf")
    patience = 0
    best_state = None

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        tr_loss = 0
        for Xs, y, ef in loader_tr:
            Xs, y, ef = Xs.to(device), y.to(device), ef.to(device)
            opt.zero_grad()
            yhat = model(Xs, edge_index.to(device), ef)
            loss = crit(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(loader_tr)

        val_loss, _, _ = eval_epoch(model, loader_va, crit, edge_index, device)
        sch.step(val_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | Train={tr_loss:.4f} | Val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= CFG["patience"]:
                print(f"  Early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    _, tp, tt = eval_epoch(model, loader_te, crit, edge_index, device)
    metrics = compute_metrics(tt, tp, y_mean, y_std)
    print(
        f"\n  📊 {name} Test → MAE={metrics['MAE']:.2f}s | "
        f"RMSE={metrics['RMSE']:.2f}s | MAPE={metrics['MAPE']:.2f}%"
    )
    return metrics, tp, tt


# ─────────────────────────────────────────────
# COMPARISON TABLE + PLOT
# ─────────────────────────────────────────────
def plot_comparison(all_metrics):
    models = list(all_metrics.keys())
    mae = [all_metrics[m]["MAE"] for m in models]
    rmse = [all_metrics[m]["RMSE"] for m in models]
    mape = [all_metrics[m]["MAPE"] for m in models]

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, vals, title, color in zip(
        axes,
        [mae, rmse, mape],
        ["MAE (s)", "RMSE (s)", "MAPE (%)"],
        ["steelblue", "coral", "mediumseagreen"],
    ):
        bars = ax.bar(x, vals, color=color, alpha=0.8, edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_title(title, fontweight="bold")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.suptitle(
        "Model Comparison — ETA Prediction (Quận 1, TP.HCM)",
        fontweight="bold",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig("results/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ Comparison plot → results/model_comparison.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    device = CFG["device"]
    ds_train, ds_val, ds_test, edge_index, y_mean, y_std, meta = load_dataset()

    N = len(meta["node_ids"])
    E = len(meta["edge_ids"])

    baselines = {
        "LSTM": LSTMBaseline(N, E),
        "GCN-GRU": GCNGRUBaseline(N, E),
        "STGCN": STGCNBaseline(N, E),
    }

    all_metrics = {}
    for name, model in baselines.items():
        metrics, _, _ = train_baseline(
            model, name, ds_train, ds_val, ds_test, edge_index, y_mean, y_std, device
        )
        all_metrics[name] = metrics

    # Load AH-GNN results nếu đã train
    ahgnn_path = "results/metrics.json"
    if os.path.exists(ahgnn_path):
        with open(ahgnn_path, encoding="utf-8") as f:
            ahgnn = json.load(f)
        all_metrics["AH-GNN"] = ahgnn["metrics"]
        print(
            f"\n  AH-GNN → MAE={ahgnn['metrics']['MAE']:.2f}s | "
            f"RMSE={ahgnn['metrics']['RMSE']:.2f}s | "
            f"MAPE={ahgnn['metrics']['MAPE']:.2f}%"
        )

    # Save
    with open("results/baseline_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Print bảng so sánh
    print(f"\n{'='*55}")
    print(f"  📊 COMPARISON TABLE (dùng cho paper)")
    print(f"{'='*55}")
    print(f"  {'Model':<12} {'MAE (s)':>10} {'RMSE (s)':>10} {'MAPE (%)':>10}")
    print(f"  {'-'*45}")
    for name, m in all_metrics.items():
        marker = " ← OURS" if name == "AH-GNN" else ""
        print(
            f"  {name:<12} {m['MAE']:>10.2f} {m['RMSE']:>10.2f} {m['MAPE']:>10.2f}{marker}"
        )

    plot_comparison(all_metrics)
    print(f"\n✅ Baseline done! → results/baseline_metrics.json")


if __name__ == "__main__":
    main()
