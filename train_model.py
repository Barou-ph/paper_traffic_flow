"""
Train AH-GNN cho bài toán ETA
==============================
Input : dataset/ (từ build_graph.py)
Output: checkpoints/best_model.pt
        results/metrics.json
        results/predictions.npy

Dùng: python train_model.py
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model import AHGNN

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CFG = {
    "dataset_dir": "dataset",
    "ckpt_dir": "checkpoints",
    "results_dir": "results",
    "seq_len": 6,  # dùng 6 timestep gần nhất để predict
    "pred_len": 1,  # predict 1 bước tiếp theo
    "hidden_dim": 32,
    "embed_dim": 16,
    "n_layers": 2,
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 100,
    "patience": 15,  # early stopping
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

print(f"Device: {CFG['device']}")


# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────
class ETADataset(Dataset):
    def __init__(self, X, Y, E_feat, seq_len=6):
        """
        X     : [T, N, F]
        Y     : [T, E]
        E_feat: [T, E, 3]
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y)
        self.E_feat = torch.FloatTensor(E_feat)
        self.seq_len = seq_len
        self.valid_idx = range(seq_len, len(X))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        t = self.valid_idx[idx]
        X_seq = self.X[t - self.seq_len : t]  # [seq_len, N, F]
        y = self.Y[t]  # [E]
        e_feat = self.E_feat[t]  # [E, 3]
        return X_seq, y, e_feat


def load_dataset():
    d = CFG["dataset_dir"]
    X = np.load(f"{d}/node_features.npy")
    Y = np.load(f"{d}/eta_labels.npy")
    E_feat = np.load(f"{d}/edge_features.npy")
    A_phys = np.load(f"{d}/adj_physical.npy")
    A_sem = np.load(f"{d}/adj_semantic.npy")

    with open(f"{d}/node_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    splits = meta["splits"]
    node_ids = meta["node_ids"]
    edge_ids = meta["edge_ids"]

    # Build edge_index [E, 2]
    node2idx = {n: i for i, n in enumerate(node_ids)}
    edge_index = []
    for eid in edge_ids:
        src, dst = eid.split("→")
        edge_index.append([node2idx[src], node2idx[dst]])
    edge_index = torch.LongTensor(edge_index)

    # Normalize Y (z-score)
    y_mean = Y.mean()
    y_std = Y.std() + 1e-8
    Y_norm = (Y - y_mean) / y_std

    print(f"✓ X: {X.shape}, Y: {Y.shape}, E_feat: {E_feat.shape}")
    print(f"  Y mean={y_mean:.1f}s, std={y_std:.1f}s")

    # Split
    tr0, tr1 = splits["train"]
    v0, v1 = splits["val"]
    te0, te1 = splits["test"]

    ds_train = ETADataset(X[tr0:tr1], Y_norm[tr0:tr1], E_feat[tr0:tr1], CFG["seq_len"])
    ds_val = ETADataset(X[v0:v1], Y_norm[v0:v1], E_feat[v0:v1], CFG["seq_len"])
    ds_test = ETADataset(X[te0:te1], Y_norm[te0:te1], E_feat[te0:te1], CFG["seq_len"])

    return ds_train, ds_val, ds_test, edge_index, y_mean, y_std, meta


# ─────────────────────────────────────────────
# 2. METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_mean, y_std):
    """Denormalize rồi tính MAE, RMSE, MAPE"""
    yt = y_true * y_std + y_mean
    yp = y_pred * y_std + y_mean
    yp = np.maximum(yp, 0)

    mae = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mask = yt > 10  # tránh chia 0
    mape = np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100
    return {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE": round(float(mape), 4),
    }


# ─────────────────────────────────────────────
# 3. TRAIN / EVAL LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, edge_index, device):
    model.train()
    total_loss = 0
    for X_seq, y, e_feat in loader:
        X_seq = X_seq.to(device)
        y = y.to(device)
        e_feat = e_feat.to(device)

        optimizer.zero_grad()
        y_hat = model(X_seq, edge_index.to(device), e_feat)
        loss = criterion(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, edge_index, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    for X_seq, y, e_feat in loader:
        X_seq = X_seq.to(device)
        y = y.to(device)
        e_feat = e_feat.to(device)
        y_hat = model(X_seq, edge_index.to(device), e_feat)
        loss = criterion(y_hat, y)
        total_loss += loss.item()
        preds.append(y_hat.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    return total_loss / len(loader), preds, trues


# ─────────────────────────────────────────────
# 4. PLOT
# ─────────────────────────────────────────────
def plot_loss(train_losses, val_losses, out_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("AH-GNN Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/loss_curve.png", dpi=150)
    plt.close()
    print(f"✓ Loss curve saved → {out_dir}/loss_curve.png")


def plot_prediction(trues, preds, y_mean, y_std, out_dir, n_samples=200):
    yt = trues[:n_samples, 0] * y_std + y_mean
    yp = preds[:n_samples, 0] * y_std + y_mean
    plt.figure(figsize=(10, 4))
    plt.plot(yt, label="Ground Truth (s)", alpha=0.8)
    plt.plot(yp, label="AH-GNN Pred (s)", alpha=0.8)
    plt.xlabel("Sample")
    plt.ylabel("Travel Time (s)")
    plt.title("AH-GNN: Predicted vs Actual ETA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/prediction.png", dpi=150)
    plt.close()
    print(f"✓ Prediction plot saved → {out_dir}/prediction.png")


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)
    os.makedirs(CFG["results_dir"], exist_ok=True)

    print("=" * 55)
    print("  Train AH-GNN — ETA Prediction")
    print("=" * 55)

    # Load data
    ds_train, ds_val, ds_test, edge_index, y_mean, y_std, meta = load_dataset()
    print(f"  Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    loader_train = DataLoader(ds_train, CFG["batch_size"], shuffle=True, drop_last=True)
    loader_val = DataLoader(ds_val, CFG["batch_size"], shuffle=False)
    loader_test = DataLoader(ds_test, CFG["batch_size"], shuffle=False)

    # Model
    device = CFG["device"]
    model = AHGNN(
        n_nodes=len(meta["node_ids"]),
        n_edges=len(meta["edge_ids"]),
        in_dim=9,
        hidden_dim=CFG["hidden_dim"],
        embed_dim=CFG["embed_dim"],
        n_layers=CFG["n_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    # Training loop
    best_val_loss = float("inf")
    patience_cnt = 0
    train_losses, val_losses = [], []

    print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} {'Time':>8}")
    print("-" * 50)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(
            model, loader_train, optimizer, criterion, edge_index, device
        )
        val_loss, vp, vt = eval_epoch(model, loader_val, criterion, edge_index, device)
        scheduler.step(val_loss)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0
        print(f"{epoch:>6} {tr_loss:>10.4f} {val_loss:>10.4f} {lr:>10.6f} {dt:>7.1f}s")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": val_loss,
                    "cfg": CFG,
                },
                f"{CFG['ckpt_dir']}/best_model.pt",
            )
            print(f"         ✓ Best model saved (val={val_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break

    # Load best → test
    print("\n" + "=" * 55)
    print("  Evaluating on Test Set")
    print("=" * 55)
    ckpt = torch.load(f"{CFG['ckpt_dir']}/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_preds, test_trues = eval_epoch(
        model, loader_test, criterion, edge_index, device
    )
    metrics = compute_metrics(test_trues, test_preds, y_mean, y_std)

    print(f"\n  📊 Test Results (denormalized):")
    print(f"  MAE  : {metrics['MAE']:.2f} s")
    print(f"  RMSE : {metrics['RMSE']:.2f} s")
    print(f"  MAPE : {metrics['MAPE']:.2f} %")

    # Save results
    results = {
        "model": "AH-GNN",
        "metrics": metrics,
        "config": CFG,
        "best_epoch": int(ckpt["epoch"]),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }
    with open(f"{CFG['results_dir']}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    np.save(f"{CFG['results_dir']}/predictions.npy", test_preds)
    np.save(f"{CFG['results_dir']}/ground_truth.npy", test_trues)

    # Plots
    plot_loss(train_losses, val_losses, CFG["results_dir"])
    plot_prediction(test_trues, test_preds, y_mean, y_std, CFG["results_dir"])

    # Learned adjacency
    A_learned = model.get_adaptive_adj()
    np.save(f"{CFG['results_dir']}/adj_learned.npy", A_learned)
    print(f"✓ Learned adjacency saved → dùng visualize trong paper")

    print(f"\n✅ Done! Results in '{CFG['results_dir']}/'")


if __name__ == "__main__":
    main()
