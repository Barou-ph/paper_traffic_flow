"""
Train ZAH-GNN (Zone-Aware Heterogeneous GNN)
=============================================
Spec mới: output [B, N, T_out=3] — node-level congestion
Interface: forward(X, Z, time_idx, A_static)

Dùng: python train_model.py
"""

import torch
import torch.nn as nn
import numpy as np
import json, os, time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from model import ZAHGNNModel

CFG = {
    "dataset_dir": "dataset",
    "ckpt_dir": "checkpoints",
    "results_dir": "results",
    "T_in": 12,  # số timestep input
    "F_feat": 9,  # số features mỗi nút
    "T_out": 3,  # predict 3 bước tiếp
    "hidden_dim": 32,
    "embed_dim": 16,
    "n_layers": 2,
    "batch_size": 16,
    "lr": 1e-3,
    "epochs": 100,
    "patience": 15,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}
print(f"Device: {CFG['device']}")


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
class NodeCongestionDataset(Dataset):
    """
    X_input : [T_in * F] per node → in_channels = T_in * F = 12*9 = 108
    Z       : [N, K] zone one-hot (static)
    time_idx: scalar per sample
    Y       : [N, T_out] node congestion labels
    """

    def __init__(self, X, Y_node, time_labels, seq_len=12):
        """
        X          : [T, N, F]
        Y_node     : [T, N, T_out]
        time_labels: [T] int (0-3)
        """
        self.X = torch.FloatTensor(X)
        self.Y = torch.FloatTensor(Y_node)
        self.time_labels = torch.LongTensor(time_labels)
        self.seq_len = seq_len
        # valid từ seq_len đến len(Y)
        self.valid_idx = range(seq_len, min(len(X), len(Y_node)))

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        t = self.valid_idx[idx]
        # X_seq: [T_in, N, F] → flatten → [N, T_in*F]
        X_seq = self.X[t - self.seq_len : t]  # [T_in, N, F]
        T, N, F = X_seq.shape
        X_flat = X_seq.permute(1, 0, 2).reshape(N, T * F)  # [N, T_in*F]
        y = (
            self.Y[t - len(self.Y) + len(self.valid_idx) - idx - 1]
            if t >= len(self.Y)
            else self.Y[t]
        )  # [N, T_out]
        # Lấy đúng index
        y_idx = t - self.seq_len
        y = self.Y[y_idx] if y_idx < len(self.Y) else self.Y[-1]
        ti = self.time_labels[t]
        return X_flat, y, ti


def load_dataset():
    d = CFG["dataset_dir"]

    X = np.load(f"{d}/node_features.npy")  # [T, N, 9]
    Y_node = np.load(f"{d}/node_labels.npy")  # [T_node, N, 3]
    Z = np.load(f"{d}/zone_onehot.npy")  # [N, K]
    gps = np.load(f"{d}/gps_coords.npy")  # [N, 2]
    A_phys = np.load(f"{d}/adj_physical.npy")  # [N, N]

    with open(f"{d}/node_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    splits = meta["splits_node"]
    T_in = CFG["T_in"]

    # Time labels từ feature index 4 (time_label_norm * 3 → 0-3)
    time_labels = np.round(X[:, 0, 4] * 3).astype(int).clip(0, 3)

    # Normalize X
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    # Normalize Y_node
    y_mean = Y_node.mean()
    y_std = Y_node.std() + 1e-8
    Y_norm = (Y_node - y_mean) / y_std

    # Align lengths (Y_node có T-3 rows)
    T_y = len(Y_node)
    T_x = len(X)
    X_use = X_norm[:T_y]
    TL = time_labels[:T_y]

    tr0, tr1 = splits["train"]
    v0, v1 = splits["val"]
    te0, te1 = splits["test"]

    ds_train = NodeCongestionDataset(X_use[tr0:tr1], Y_norm[tr0:tr1], TL[tr0:tr1], T_in)
    ds_val = NodeCongestionDataset(X_use[v0:v1], Y_norm[v0:v1], TL[v0:v1], T_in)
    ds_test = NodeCongestionDataset(X_use[te0:te1], Y_norm[te0:te1], TL[te0:te1], T_in)

    print(f"✓ X: {X.shape}, Y_node: {Y_node.shape}, Z: {Z.shape}")
    print(f"  Train: {len(ds_train)} | Val: {len(ds_val)} | Test: {len(ds_test)}")

    return ds_train, ds_val, ds_test, Z, gps, A_phys, y_mean, y_std, meta


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_mean, y_std):
    yt = y_true * y_std + y_mean
    yp = np.maximum(y_pred * y_std + y_mean, 0)
    mae = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mask = yt > 0.05
    mape = (
        float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)
        if mask.any()
        else 0.0
    )

    # MAE multi-zone: tính riêng từng nút rồi lấy mean (theo spec)
    mae_per_node = (
        np.mean(np.abs(yt - yp), axis=(0, 2))
        if yt.ndim == 3
        else np.mean(np.abs(yt - yp), axis=0)
    )
    mae_multi = float(mae_per_node.mean())

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "MAPE": round(mape, 4),
        "MAE_multi_zone": round(mae_multi, 4),
    }


# ─────────────────────────────────────────────
# TRAIN / EVAL
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, Z_t, A_t, device):
    model.train()
    total = 0
    for X_flat, y, ti in loader:
        X_flat, y, ti = X_flat.to(device), y.to(device), ti.to(device)
        optimizer.zero_grad()
        y_hat = model(X_flat, Z_t, ti, A_t)  # [B, N, T_out]
        loss = criterion(y_hat, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def eval_epoch(model, loader, criterion, Z_t, A_t, device):
    model.eval()
    total, preds, trues = 0, [], []
    for X_flat, y, ti in loader:
        X_flat, y, ti = X_flat.to(device), y.to(device), ti.to(device)
        y_hat = model(X_flat, Z_t, ti, A_t)
        total += criterion(y_hat, y).item()
        preds.append(y_hat.cpu().numpy())
        trues.append(y.cpu().numpy())
    return total / len(loader), np.concatenate(preds), np.concatenate(trues)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs(CFG["ckpt_dir"], exist_ok=True)
    os.makedirs(CFG["results_dir"], exist_ok=True)

    print("=" * 55)
    print("  Train ZAH-GNN — Node-level Congestion Prediction")
    print("=" * 55)

    ds_train, ds_val, ds_test, Z, gps, A_phys, y_mean, y_std, meta = load_dataset()

    device = CFG["device"]
    Z_t = torch.FloatTensor(Z).to(device)
    A_t = torch.FloatTensor(A_phys).to(device)

    in_channels = CFG["T_in"] * CFG["F_feat"]  # 12 * 9 = 108

    model = ZAHGNNModel(
        num_nodes=len(meta["node_ids"]),
        in_channels=in_channels,
        out_channels=CFG["T_out"],
        hidden_dim=CFG["hidden_dim"],
        embed_dim=CFG["embed_dim"],
        n_zone_types=len(
            meta.get(
                "poi_types",
                [
                    "commercial",
                    "arterial",
                    "roundabout",
                    "residential",
                    "bridge",
                    "mixed",
                ],
            )
        ),
        n_layers=CFG["n_layers"],
        gps_coords=gps,
        T_in=CFG["T_in"],
        F_feat=CFG["F_feat"],
    ).to(device)

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    loader_tr = DataLoader(ds_train, CFG["batch_size"], shuffle=True, drop_last=True)
    loader_va = DataLoader(ds_val, CFG["batch_size"], shuffle=False)
    loader_te = DataLoader(ds_test, CFG["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val, patience_cnt = float("inf"), 0
    tr_losses, va_losses = [], []

    print(f"\n{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10} {'Time':>8}")
    print("-" * 50)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, loader_tr, optimizer, criterion, Z_t, A_t, device)
        va_loss, _, _ = eval_epoch(model, loader_va, criterion, Z_t, A_t, device)
        scheduler.step(va_loss)

        tr_losses.append(tr_loss)
        va_losses.append(va_loss)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>6} {tr_loss:>10.4f} {va_loss:>10.4f} {lr:>10.6f} {time.time()-t0:>7.1f}s"
        )

        if va_loss < best_val:
            best_val, patience_cnt = va_loss, 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_loss": va_loss,
                    "cfg": CFG,
                },
                f"{CFG['ckpt_dir']}/best_model.pt",
            )
            print(f"         ✓ Best saved (val={va_loss:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= CFG["patience"]:
                print(f"\n⏹ Early stopping at epoch {epoch}")
                break

    # Test
    ckpt = torch.load(f"{CFG['ckpt_dir']}/best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, tp, tt = eval_epoch(model, loader_te, criterion, Z_t, A_t, device)
    metrics = compute_metrics(tt, tp, y_mean, y_std)

    print(f"\n{'='*55}")
    print(f"  📊 Test Results:")
    print(f"  MAE           : {metrics['MAE']:.4f}")
    print(f"  RMSE          : {metrics['RMSE']:.4f}")
    print(f"  MAPE          : {metrics['MAPE']:.2f}%")
    print(f"  MAE_multi_zone: {metrics['MAE_multi_zone']:.4f}")

    results = {
        "model": "ZAH-GNN",
        "metrics": metrics,
        "cfg": CFG,
        "best_epoch": int(ckpt["epoch"]),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }
    with open(f"{CFG['results_dir']}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Loss plot
    plt.figure(figsize=(8, 4))
    plt.plot(tr_losses, label="Train")
    plt.plot(va_losses, label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("ZAH-GNN Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{CFG['results_dir']}/loss_curve.png", dpi=150)
    plt.close()

    A_learned = model.get_adaptive_adj(A_t)
    np.save(f"{CFG['results_dir']}/adj_learned.npy", A_learned)
    print(f"\n✅ Done! → {CFG['results_dir']}/")


if __name__ == "__main__":
    main()
