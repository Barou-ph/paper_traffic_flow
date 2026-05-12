"""
Build Graph Dataset cho AH-GNN từ eta_data.csv
===============================================
v2: thêm hour_of_day, day_of_week, is_rush features
Input : tomtom_data/eta_data.csv
Output: dataset/
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.stats import ks_2samp
from numpy.linalg import lstsq
import warnings

warnings.filterwarnings("ignore")

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
# 1. LOAD
# ─────────────────────────────────────────────
def load_data():
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"])
    df = df.drop_duplicates()
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek  # 0=Mon, 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_rush"] = df["time_label"].isin(["rush_morning", "rush_evening"]).astype(int)
    print(f"✓ Loaded: {len(df)} rows, {df.timestamp.nunique()} snapshots")
    print(
        f"  time_label distribution:\n{df.groupby('timestamp')['time_label'].first().value_counts().to_string()}"
    )
    return df


# ─────────────────────────────────────────────
# 2. NODE FEATURES [T, N, F=9]
# ─────────────────────────────────────────────
def build_node_features(df):
    """
    9 features mỗi nút:
      f0: avg_congestion_out   — tắc nghẽn trung bình đi ra
      f1: avg_travel_time_out  — travel time đi ra (chuẩn hoá /1000)
      f2: avg_delay_in         — delay đi vào (chuẩn hoá /100)
      f3: avg_congestion_in    — tắc nghẽn đi vào
      f4: time_label_enc       — 0=night,1=normal,2=rush_m,3=rush_e
      f5: poi_enc              — loại nút (static)
      f6: hour_sin             — sin(hour*2π/24) → chu kỳ ngày
      f7: hour_cos             — cos(hour*2π/24)
      f8: is_rush              — 0/1 giờ cao điểm
    """
    node_ids = sorted(NODES_INFO.keys())
    timestamps = sorted(df["timestamp"].unique())
    N, T, F = len(node_ids), len(timestamps), 9

    node2idx = {n: i for i, n in enumerate(node_ids)}
    time2idx = {t: i for i, t in enumerate(timestamps)}

    X = np.zeros((T, N, F), dtype=np.float32)

    # POI encoding (static, không đổi theo thời gian)
    for nid, info in NODES_INFO.items():
        i = node2idx[nid]
        X[:, i, 5] = POI_MAP.get(info["poi_type"], 0) / 5.0  # normalize 0-1

    for ts, group in df.groupby("timestamp"):
        t = time2idx[ts]
        hour = group["hour"].iloc[0]
        time_enc = TIME_LABEL_MAP.get(group["time_label"].iloc[0], 1)
        is_rush = group["is_rush"].iloc[0]

        # Temporal features (same cho tất cả nút tại timestep t)
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        for nid in node_ids:
            i = node2idx[nid]
            X[t, i, 4] = time_enc / 3.0  # normalize 0-1
            X[t, i, 6] = hour_sin
            X[t, i, 7] = hour_cos
            X[t, i, 8] = float(is_rush)

            out = group[group["src_node"] == nid]
            if len(out) > 0:
                X[t, i, 0] = out["congestion_ratio"].mean()
                X[t, i, 1] = out["travel_time_s"].mean() / 1000.0

            inc = group[group["dst_node"] == nid]
            if len(inc) > 0:
                X[t, i, 2] = inc["traffic_delay_s"].mean() / 100.0
                X[t, i, 3] = inc["congestion_ratio"].mean()

    # Forward-fill
    for n in range(N):
        for f in [0, 1, 2, 3]:
            s = pd.Series(X[:, n, f])
            X[:, n, f] = s.replace(0, np.nan).ffill().bfill().fillna(1.0).values

    print(f"✓ Node features X: {X.shape}  (T={T}, N={N}, F={F})")
    print(f"  Features: congestion_out, tt_out, delay_in, congestion_in,")
    print(f"            time_label, poi, hour_sin, hour_cos, is_rush")
    return X, timestamps, node_ids, node2idx, time2idx


# ─────────────────────────────────────────────
# 3. EDGE FEATURES + LABELS [T, E]
# ─────────────────────────────────────────────
def build_edge_data(df, timestamps, time2idx):
    df = df.copy()
    df["edge_id"] = df["src_node"] + "→" + df["dst_node"]
    edge_ids = sorted(df["edge_id"].unique())
    E, T = len(edge_ids), len(timestamps)
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

    for e in range(E):
        s = pd.Series(Y[:, e])
        Y[:, e] = s.replace(0, np.nan).ffill().bfill().fillna(s.median()).values

    print(f"✓ ETA labels Y:     {Y.shape}")
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
    """KS-test trên congestion_ratio → chứng minh Non-ID"""
    N = len(node_ids)
    cong = X[:, :, 0]
    A_sem = np.zeros((N, N), dtype=np.float32)
    KS = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            ks, _ = ks_2samp(cong[:, i], cong[:, j])
            KS[i, j] = round(ks, 4)
            if (1 - ks) > 0.6:
                A_sem[i, j] = round(1 - ks, 4)
    print(f"✓ Semantic adj:     {int((A_sem>0).sum())} edges")
    print(f"  KS mean={KS.mean():.4f} → Non-ID evidence (Section 2)")
    return A_sem, KS


def build_granger_adj(X, node_ids, max_lag=3):
    """Granger causality → chứng minh Non-I"""
    N, T = len(node_ids), X.shape[0]
    cong = X[:, :, 0]
    G = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            y = cong[max_lag:, i]
            Xr = np.column_stack(
                [cong[max_lag - k - 1 : T - k - 1, i] for k in range(max_lag)]
            )
            Xu = np.column_stack(
                [Xr] + [cong[max_lag - k - 1 : T - k - 1, j] for k in range(max_lag)]
            )
            if len(y) < 10:
                continue
            yr = y - Xr @ lstsq(Xr, y, rcond=None)[0]
            yu = y - Xu @ lstsq(Xu, y, rcond=None)[0]
            rr, ru = (yr**2).sum(), (yu**2).sum()
            if rr > 0:
                F = ((rr - ru) / max_lag) / max(ru / (len(y) - 2 * max_lag - 1), 1e-9)
                G[i, j] = 1.0 if F > 3.84 else 0.0
    print(
        f"✓ Granger links:    {int(G.sum())} causal links → Non-I evidence (Section 3)"
    )
    return G


# ─────────────────────────────────────────────
# 5. SPLIT + SAVE
# ─────────────────────────────────────────────
def save_all(X, Y, E_feat, edge_ids, timestamps, node_ids, A_phys, A_sem, KS, G):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(f"{OUTPUT_DIR}/node_features.npy", X)
    np.save(f"{OUTPUT_DIR}/eta_labels.npy", Y)
    np.save(f"{OUTPUT_DIR}/edge_features.npy", E_feat)
    np.save(f"{OUTPUT_DIR}/adj_physical.npy", A_phys)
    np.save(f"{OUTPUT_DIR}/adj_semantic.npy", A_sem)
    np.save(f"{OUTPUT_DIR}/non_id_ks_matrix.npy", KS)
    np.save(f"{OUTPUT_DIR}/granger_causality.npy", G)

    T = len(timestamps)
    splits = {
        "train": [0, int(T * 0.70)],
        "val": [int(T * 0.70), int(T * 0.85)],
        "test": [int(T * 0.85), T],
    }

    meta = {
        "node_ids": node_ids,
        "edge_ids": edge_ids,
        "T": T,
        "N": len(node_ids),
        "E": len(edge_ids),
        "F": X.shape[2],
        "splits": splits,
        "nodes": NODES_INFO,
        "feature_names": [
            "avg_congestion_out",
            "avg_travel_time_out_norm",
            "avg_delay_in_norm",
            "avg_congestion_in",
            "time_label_norm",
            "poi_norm",
            "hour_sin",
            "hour_cos",
            "is_rush",
        ],
    }
    with open(f"{OUTPUT_DIR}/node_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    stats = {
        "total_snapshots": T,
        "n_nodes": len(node_ids),
        "n_edges": len(edge_ids),
        "X_shape": list(X.shape),
        "Y_shape": list(Y.shape),
        "Y_mean_s": float(np.nanmean(Y)),
        "Y_std_s": float(np.nanstd(Y)),
        "Y_min_s": float(np.nanmin(Y)),
        "Y_max_s": float(np.nanmax(Y)),
        "train_snapshots": splits["train"][1],
        "val_snapshots": splits["val"][1] - splits["val"][0],
        "test_snapshots": splits["test"][1] - splits["test"][0],
        "KS_mean_divergence": float(KS.mean()),
        "granger_causal_links": int(G.sum()),
    }
    with open(f"{OUTPUT_DIR}/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  ✅ Dataset v2 saved to '{OUTPUT_DIR}/'")
    print(f"{'='*55}")
    print(f"  node_features.npy : {X.shape}  ← 9 features (tăng từ 6)")
    print(f"  eta_labels.npy    : {Y.shape}")
    print(
        f"  Train/Val/Test    : {splits['train'][1]} / "
        f"{splits['val'][1]-splits['val'][0]} / "
        f"{splits['test'][1]-splits['test'][0]} snapshots"
    )
    print(f"\n  📊 Stats cho paper:")
    print(f"  Mean ETA : {stats['Y_mean_s']:.1f}s  Std: {stats['Y_std_s']:.1f}s")
    print(f"  KS div   : {stats['KS_mean_divergence']:.4f}  (Non-ID)")
    print(f"  Granger  : {stats['granger_causal_links']} links  (Non-I)")
    print(f"\n  → Tiếp theo: cập nhật in_dim=9 trong train_model.py rồi chạy lại")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Build AH-GNN Dataset v2 (9 features)")
    print("=" * 55)

    print("\n[1/6] Loading data...")
    df = load_data()

    print("\n[2/6] Building node features (F=9)...")
    X, timestamps, node_ids, node2idx, time2idx = build_node_features(df)

    print("\n[3/6] Building edge features + labels...")
    Y, E_feat, edge_ids, edge2idx = build_edge_data(df, timestamps, time2idx)

    print("\n[4/6] Physical adjacency...")
    A_phys = build_physical_adj(node_ids)

    print("\n[5/6] Semantic adjacency (Non-ID)...")
    A_sem, KS = build_semantic_adj(X, node_ids)

    print("\n[6/6] Granger causality (Non-I)...")
    G = build_granger_adj(X, node_ids)

    save_all(X, Y, E_feat, edge_ids, timestamps, node_ids, A_phys, A_sem, KS, G)
