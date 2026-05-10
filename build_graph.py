"""
Build Graph Dataset cho AH-GNN từ eta_data.csv
===============================================
Input : tomtom_data/eta_data.csv
Output: dataset/
    ├── node_features.npy     [T, N, F]  - input GNN
    ├── edge_features.npy     [T, E, 3]  - edge attributes
    ├── eta_labels.npy        [T, E]     - target Y
    ├── adj_physical.npy      [N, N]     - kề vật lý
    ├── adj_semantic.npy      [N, N]     - kề theo phân phối
    ├── non_id_ks_matrix.npy  [N, N]     - KS divergence (chứng minh Non-ID)
    ├── granger_causality.npy [N, N]     - Granger test (chứng minh Non-I)
    ├── node_meta.json                   - metadata
    └── dataset_stats.json               - thống kê để viết paper

Dùng: python build_graph.py
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.stats import ks_2samp
from numpy.linalg import lstsq
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_CSV = "tomtom_data/eta_data.csv"
OUTPUT_DIR = "dataset"

NODES_INFO = {
    "v01": {
        "lat": 10.7729,
        "lon": 106.6980,
        "poi_type": "commercial",
        "name": "Bến Thành",
    },
    "v02": {
        "lat": 10.7875,
        "lon": 106.6923,
        "poi_type": "arterial",
        "name": "Điện Biên Phủ-Hai BT",
    },
    "v03": {
        "lat": 10.7802,
        "lon": 106.6958,
        "poi_type": "roundabout",
        "name": "Vòng xoay Dân Chủ",
    },
    "v04": {
        "lat": 10.7848,
        "lon": 106.6889,
        "poi_type": "residential",
        "name": "Lê Văn Sỹ-N.Đ.Chiểu",
    },
    "v05": {
        "lat": 10.7867,
        "lon": 106.6850,
        "poi_type": "arterial",
        "name": "CMT8-Điện Biên Phủ",
    },
    "v06": {
        "lat": 10.7960,
        "lon": 106.7222,
        "poi_type": "bridge",
        "name": "Cầu Sài Gòn",
    },
    "v07": {
        "lat": 10.7740,
        "lon": 106.7030,
        "poi_type": "commercial",
        "name": "Nguyễn Huệ-Lê Lợi",
    },
    "v08": {
        "lat": 10.7723,
        "lon": 106.7040,
        "poi_type": "commercial",
        "name": "Hàm Nghi-TĐT",
    },
    "v09": {
        "lat": 10.7798,
        "lon": 106.6910,
        "poi_type": "residential",
        "name": "Võ Văn Tần-NKK",
    },
    "v10": {
        "lat": 10.7822,
        "lon": 106.6978,
        "poi_type": "mixed",
        "name": "Pasteur-Lê Duẩn",
    },
}

TIME_LABEL_MAP = {"night": 0, "normal": 1, "rush_morning": 2, "rush_evening": 3}
POI_MAP = {
    "residential": 0,
    "mixed": 1,
    "roundabout": 2,
    "commercial": 3,
    "arterial": 4,
    "bridge": 5,
}


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
    df = df.drop_duplicates()
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"✓ Loaded: {len(df)} rows, {df.timestamp.nunique()} snapshots")
    print(f"  time_label:\n{df['time_label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────
# 2. BUILD NODE FEATURES [T, N, F]
# ─────────────────────────────────────────────
def build_node_features(df):
    """
    Aggregate từ edge → node (message passing ngược)
    Features mỗi nút tại mỗi timestep:
      f0: avg_congestion_out   - mức tắc nghẽn trung bình đi ra
      f1: avg_travel_time_out  - thời gian đi trung bình đi ra (chuẩn hoá)
      f2: avg_delay_in         - delay trung bình đi vào
      f3: avg_congestion_in    - tắc nghẽn đi vào
      f4: time_label_encoded   - khung giờ (0-3)
      f5: poi_encoded          - loại nút
    """
    node_ids = sorted(NODES_INFO.keys())
    timestamps = sorted(df["timestamp"].unique())
    N = len(node_ids)
    T = len(timestamps)
    F = 6

    node2idx = {n: i for i, n in enumerate(node_ids)}
    time2idx = {t: i for i, t in enumerate(timestamps)}

    X = np.zeros((T, N, F), dtype=np.float32)

    # POI encoding (static)
    for nid, info in NODES_INFO.items():
        i = node2idx[nid]
        X[:, i, 5] = POI_MAP.get(info["poi_type"], 0)

    # Aggregate theo từng snapshot
    for ts, group in df.groupby("timestamp"):
        t = time2idx[ts]
        time_enc = TIME_LABEL_MAP.get(group["time_label"].iloc[0], 1)

        for nid in node_ids:
            i = node2idx[nid]
            X[t, i, 4] = time_enc

            # Outgoing edges từ nút này
            out = group[group["src_node"] == nid]
            if len(out) > 0:
                X[t, i, 0] = out["congestion_ratio"].mean()
                X[t, i, 1] = out["travel_time_s"].mean() / 1000.0  # normalize

            # Incoming edges vào nút này
            inc = group[group["dst_node"] == nid]
            if len(inc) > 0:
                X[t, i, 2] = inc["traffic_delay_s"].mean() / 100.0  # normalize
                X[t, i, 3] = inc["congestion_ratio"].mean()

    # Forward-fill các ô còn 0
    for n in range(N):
        for f in range(4):  # chỉ fill features động
            series = pd.Series(X[:, n, f])
            X[:, n, f] = series.replace(0, np.nan).ffill().bfill().fillna(1.0).values

    print(f"✓ Node features X: {X.shape}  (T={T}, N={N}, F={F})")
    return X, timestamps, node_ids, node2idx, time2idx


# ─────────────────────────────────────────────
# 3. BUILD EDGE FEATURES + ETA LABELS [T, E]
# ─────────────────────────────────────────────
def build_edge_data(df, timestamps, time2idx):
    """
    Edge features: [congestion_ratio, traffic_delay_s, length_m]
    Label Y      : travel_time_s (cái model phải predict)
    """
    df["edge_id"] = df["src_node"] + "→" + df["dst_node"]
    edge_ids = sorted(df["edge_id"].unique())
    E = len(edge_ids)
    T = len(timestamps)
    edge2idx = {e: i for i, e in enumerate(edge_ids)}

    Y = np.zeros((T, E), dtype=np.float32)
    E_feat = np.zeros((T, E, 3), dtype=np.float32)

    for _, row in df.iterrows():
        t = time2idx.get(row["timestamp"])
        e = edge2idx.get(row["edge_id"])
        if t is None or e is None:
            continue
        Y[t, e] = row["travel_time_s"]
        E_feat[t, e, 0] = row["congestion_ratio"]
        E_feat[t, e, 1] = row["traffic_delay_s"] / 100.0
        E_feat[t, e, 2] = row["length_m"] / 10000.0

    # Forward-fill
    for e in range(E):
        s = pd.Series(Y[:, e])
        Y[:, e] = s.replace(0, np.nan).ffill().bfill().fillna(s.median()).values

    print(f"✓ ETA labels Y:     {Y.shape}  (T, E={E})")
    print(f"✓ Edge features:    {E_feat.shape}")
    return Y, E_feat, edge_ids, edge2idx


# ─────────────────────────────────────────────
# 4. ADJACENCY MATRICES
# ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def build_physical_adj(node_ids, threshold_km=1.5):
    N = len(node_ids)
    A = np.zeros((N, N), dtype=np.float32)
    for i, ni in enumerate(node_ids):
        for j, nj in enumerate(node_ids):
            if i == j:
                continue
            d = haversine(
                NODES_INFO[ni]["lat"],
                NODES_INFO[ni]["lon"],
                NODES_INFO[nj]["lat"],
                NODES_INFO[nj]["lon"],
            )
            if d < threshold_km:
                A[i, j] = np.exp(-d)
    print(f"✓ Physical adj:     {int(A.sum())} edges (threshold={threshold_km}km)")
    return A


def build_semantic_adj(X, node_ids):
    """
    Dùng KS-test để đo Non-ID giữa các nút
    → Nút có phân phối tương đồng → kết nối trong A_sem
    → KS matrix = bằng chứng Non-ID cho Section 2 paper
    """
    N = len(node_ids)
    speed = X[:, :, 0]  # congestion_ratio feature
    A_sem = np.zeros((N, N), dtype=np.float32)
    KS_mat = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xi = speed[:, i]
            xj = speed[:, j]
            ks_stat, _ = ks_2samp(xi, xj)
            KS_mat[i, j] = round(ks_stat, 4)
            if (1 - ks_stat) > 0.6:  # similarity threshold
                A_sem[i, j] = round(1 - ks_stat, 4)

    print(f"✓ Semantic adj:     {int((A_sem>0).sum())} edges (KS similarity>0.6)")
    print(f"  → KS matrix = chứng minh Non-ID (Section 2 paper)")
    return A_sem, KS_mat


def build_granger_adj(X, node_ids, max_lag=3):
    """
    Granger Causality pairwise → A_causal
    → Chứng minh Non-I (Section 3 paper)
    """
    N = len(node_ids)
    T = X.shape[0]
    spd = X[:, :, 0]
    G = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            y = spd[max_lag:, i]
            Xr = np.column_stack(
                [spd[max_lag - k - 1 : T - k - 1, i] for k in range(max_lag)]
            )
            Xu = np.column_stack(
                [Xr] + [spd[max_lag - k - 1 : T - k - 1, j] for k in range(max_lag)]
            )
            if len(y) < 10:
                continue
            yr = y - Xr @ lstsq(Xr, y, rcond=None)[0]
            yu = y - Xu @ lstsq(Xu, y, rcond=None)[0]
            rss_r, rss_u = (yr**2).sum(), (yu**2).sum()
            if rss_r > 0:
                F = ((rss_r - rss_u) / max_lag) / max(
                    rss_u / (len(y) - 2 * max_lag - 1), 1e-9
                )
                G[i, j] = 1.0 if F > 3.84 else 0.0

    print(f"✓ Granger causality: {int(G.sum())} causal links")
    print(f"  → Chứng minh Non-I (Section 3 paper)")
    return G


# ─────────────────────────────────────────────
# 5. TRAIN/VAL/TEST SPLIT
# ─────────────────────────────────────────────
def split_dataset(T):
    t_train = int(T * 0.70)
    t_val = int(T * 0.85)
    return {"train": (0, t_train), "val": (t_train, t_val), "test": (t_val, T)}


# ─────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────
def save_all(
    X, Y, E_feat, edge_ids, timestamps, node_ids, A_phys, A_sem, KS_mat, G, splits
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/node_features.npy", X)
    np.save(f"{OUTPUT_DIR}/eta_labels.npy", Y)
    np.save(f"{OUTPUT_DIR}/edge_features.npy", E_feat)
    np.save(f"{OUTPUT_DIR}/adj_physical.npy", A_phys)
    np.save(f"{OUTPUT_DIR}/adj_semantic.npy", A_sem)
    np.save(f"{OUTPUT_DIR}/non_id_ks_matrix.npy", KS_mat)
    np.save(f"{OUTPUT_DIR}/granger_causality.npy", G)

    meta = {
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "T": len(timestamps),
        "N": len(node_ids),
        "E": len(edge_ids),
        "F": X.shape[2],
        "splits": splits,
        "nodes": NODES_INFO,
    }
    with open(f"{OUTPUT_DIR}/node_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Stats để viết paper
    stats = {
        "total_rows": int(len(timestamps) * len(edge_ids)),
        "total_snapshots": len(timestamps),
        "n_nodes": len(node_ids),
        "n_edges": len(edge_ids),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "Y_mean_travel_time_s": float(np.nanmean(Y)),
        "Y_std_travel_time_s": float(np.nanstd(Y)),
        "Y_min": float(np.nanmin(Y)),
        "Y_max": float(np.nanmax(Y)),
        "train_snapshots": splits["train"][1] - splits["train"][0],
        "val_snapshots": splits["val"][1] - splits["val"][0],
        "test_snapshots": splits["test"][1] - splits["test"][0],
        "KS_mean_divergence": float(np.mean(KS_mat)),
        "granger_causal_links": int(G.sum()),
    }
    with open(f"{OUTPUT_DIR}/dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  ✅ Dataset saved to '{OUTPUT_DIR}/'")
    print(f"{'='*55}")
    print(f"  node_features.npy  : {X.shape}")
    print(f"  eta_labels.npy     : {Y.shape}")
    print(f"  adj_physical.npy   : {A_phys.shape}")
    print(f"  adj_semantic.npy   : {A_sem.shape}")
    print(
        f"  Train/Val/Test     : {splits['train'][1]} / "
        f"{splits['val'][1]-splits['val'][0]} / "
        f"{splits['test'][1]-splits['test'][0]} snapshots"
    )
    print(f"\n  📊 Stats cho paper:")
    print(f"  Mean ETA    : {stats['Y_mean_travel_time_s']:.1f}s")
    print(f"  Std  ETA    : {stats['Y_std_travel_time_s']:.1f}s")
    print(f"  KS divergence (Non-ID): {stats['KS_mean_divergence']:.4f}")
    print(f"  Granger links (Non-I) : {stats['granger_causal_links']}")
    print(f"\n  → Tiếp theo: chạy train_model.py")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Build AH-GNN Dataset")
    print("=" * 55)

    print("\n[1/6] Loading data...")
    df = load_data()

    print("\n[2/6] Building node features...")
    X, timestamps, node_ids, node2idx, time2idx = build_node_features(df)

    print("\n[3/6] Building edge features + ETA labels...")
    Y, E_feat, edge_ids, edge2idx = build_edge_data(df, timestamps, time2idx)

    print("\n[4/6] Building physical adjacency...")
    A_phys = build_physical_adj(node_ids)

    print("\n[5/6] Building semantic adjacency (Non-ID)...")
    A_sem, KS_mat = build_semantic_adj(X, node_ids)

    print("\n[6/6] Granger causality test (Non-I)...")
    G = build_granger_adj(X, node_ids)

    splits = split_dataset(len(timestamps))
    save_all(
        X, Y, E_feat, edge_ids, timestamps, node_ids, A_phys, A_sem, KS_mat, G, splits
    )
