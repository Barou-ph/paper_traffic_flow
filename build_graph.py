"""
Build Dataset cho AH-GNN từ TomTom CSV
=======================================
Input : tomtom_data/flow_data.csv
        tomtom_data/eta_data.csv
Output: dataset/
    ├── node_features.npy     ← X(t): [T, N, F] - dùng cho GNN input
    ├── eta_labels.npy        ← Y: [T, E] - travel time labels
    ├── adj_physical.npy      ← A_phys: kề vật lý (khoảng cách)
    ├── adj_semantic.npy      ← A_sem: tương đồng phân phối (Non-ID)
    ├── node_meta.json        ← thông tin nút (tên, poi_type, lat/lon)
    └── granger_causality.npy ← kiểm định nhân quả (Non-I)

Dùng: python build_graph.py --input_dir tomtom_data --output_dir dataset
"""

import numpy as np
import pandas as pd
import json
import os
import argparse
from scipy.spatial.distance import euclidean
from scipy.stats import ks_2samp
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS
# ─────────────────────────────────────────────


def load_data(input_dir: str):
    flow_path = os.path.join(input_dir, "flow_data.csv")
    eta_path = os.path.join(input_dir, "eta_data.csv")

    df_flow = pd.read_csv(flow_path, parse_dates=["timestamp"])
    df_eta = pd.read_csv(eta_path, parse_dates=["timestamp"])

    print(f"✓ Loaded flow: {df_flow.shape} | eta: {df_eta.shape}")
    return df_flow, df_eta


def build_node_feature_matrix(df_flow: pd.DataFrame):
    """
    Tạo tensor X: [T, N, F]
    T = số timestep, N = số nút, F = số features
    Features: [current_speed, free_flow_speed, travel_time_ratio, confidence]
    """
    # Tính ratio = current/freeflow (congestion index)
    df_flow["congestion_ratio"] = (
        (df_flow["current_speed_kmph"] / df_flow["free_flow_speed_kmph"])
        .fillna(1.0)
        .clip(0, 1)
    )

    # Pivot: rows=timestamp, cols=node_id
    features = [
        "current_speed_kmph",
        "free_flow_speed_kmph",
        "current_travel_time_s",
        "congestion_ratio",
        "confidence",
    ]

    pivot_dict = {}
    for feat in features:
        pivot_dict[feat] = df_flow.pivot_table(
            index="timestamp", columns="node_id", values=feat
        )

    # Align tất cả theo cùng index và column
    timestamps = pivot_dict["current_speed_kmph"].index
    node_ids = sorted(pivot_dict["current_speed_kmph"].columns.tolist())

    T = len(timestamps)
    N = len(node_ids)
    F = len(features)

    X = np.zeros((T, N, F), dtype=np.float32)
    for f_idx, feat in enumerate(features):
        mat = pivot_dict[feat].reindex(columns=node_ids).values
        # Forward-fill NaN
        df_tmp = pd.DataFrame(mat).fillna(method="ffill").fillna(method="bfill")
        X[:, :, f_idx] = df_tmp.values

    print(f"✓ Node feature tensor X: {X.shape}  (T={T}, N={N}, F={F})")
    return X, timestamps, node_ids


def build_eta_labels(df_eta: pd.DataFrame, timestamps, src_dst_pairs=None):
    """
    Tạo label Y: [T, E] với E = số cặp edge
    """
    df_eta["edge_id"] = df_eta["src_node"] + "→" + df_eta["dst_node"]
    edges = sorted(df_eta["edge_id"].unique().tolist())

    pivot = df_eta.pivot_table(
        index="timestamp", columns="edge_id", values="travel_time_s"
    ).reindex(index=timestamps, columns=edges)

    Y = pivot.fillna(method="ffill").fillna(method="bfill").values.astype(np.float32)
    print(f"✓ ETA label tensor Y: {Y.shape}  (T, E={len(edges)})")
    return Y, edges


# ─────────────────────────────────────────────
# 2. XÂY DỰNG ADJACENCY MATRIX
# ─────────────────────────────────────────────


def build_physical_adjacency(node_ids: list, nodes_info: dict, threshold_km=1.0):
    """
    A_phys: kết nối nút nếu khoảng cách < threshold_km
    Dùng haversine distance
    """
    N = len(node_ids)
    A = np.zeros((N, N), dtype=np.float32)

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371  # km
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(np.radians(lat1))
            * np.cos(np.radians(lat2))
            * np.sin(dlon / 2) ** 2
        )
        return R * 2 * np.arcsin(np.sqrt(a))

    for i, ni in enumerate(node_ids):
        for j, nj in enumerate(node_ids):
            if i == j:
                continue
            if ni not in nodes_info or nj not in nodes_info:
                continue
            d = haversine(
                nodes_info[ni]["lat"],
                nodes_info[ni]["lon"],
                nodes_info[nj]["lat"],
                nodes_info[nj]["lon"],
            )
            if d < threshold_km:
                A[i, j] = np.exp(-d)  # Gaussian kernel

    print(f"✓ Physical adj: {A.sum():.0f} edges (threshold={threshold_km}km)")
    return A


def build_semantic_adjacency(X: np.ndarray, node_ids: list):
    """
    A_sem: kết nối nút có phân phối lưu lượng tương đồng
    Dùng KS-test (Kolmogorov-Smirnov) để đo Non-ID
    → Đây là đóng góp của paper (Eq.6 trong tài liệu)
    """
    N = len(node_ids)
    speed_series = X[:, :, 0]  # current_speed cho tất cả nút

    A_sem = np.zeros((N, N), dtype=np.float32)
    KL_matrix = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            xi = speed_series[:, i]
            xj = speed_series[:, j]
            # KS statistic: 0=identical, 1=totally different
            ks_stat, _ = ks_2samp(xi[~np.isnan(xi)], xj[~np.isnan(xj)])
            similarity = 1 - ks_stat  # cao = giống nhau
            KL_matrix[i, j] = ks_stat  # Non-ID measure
            if similarity > 0.7:  # threshold
                A_sem[i, j] = similarity

    print(f"✓ Semantic adj: {A_sem.sum():.0f} edges (similarity>0.7)")
    print(f"  Non-ID matrix (KS divergence) computed → phục vụ Section 2 paper")
    return A_sem, KL_matrix


def build_granger_causality(X: np.ndarray, node_ids: list, max_lag=3):
    """
    Kiểm định Granger Causality pairwise → A_causal
    Phục vụ kiểm định Non-I (Section 3 trong paper)
    Simplified: dùng linear regression F-test
    """
    from numpy.linalg import lstsq

    N = len(node_ids)
    T = X.shape[0]
    speed = X[:, :, 0]
    G = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            y = speed[max_lag:, i]
            # Restricted: chỉ dùng lags của chính i
            X_r = np.column_stack(
                [speed[max_lag - k - 1 : T - k - 1, i] for k in range(max_lag)]
            )
            # Unrestricted: thêm lags của j
            X_u = np.column_stack(
                [X_r] + [speed[max_lag - k - 1 : T - k - 1, j] for k in range(max_lag)]
            )

            if len(y) < max_lag + 5:
                continue

            _, res_r, _, _ = lstsq(X_r, y, rcond=None)
            _, res_u, _, _ = lstsq(X_u, y, rcond=None)

            rss_r = np.sum((y - X_r @ lstsq(X_r, y, rcond=None)[0]) ** 2)
            rss_u = np.sum((y - X_u @ lstsq(X_u, y, rcond=None)[0]) ** 2)

            # F-statistic simplified
            if rss_r > 0:
                F = ((rss_r - rss_u) / max_lag) / (
                    rss_u / max(len(y) - 2 * max_lag - 1, 1)
                )
                G[i, j] = 1.0 if F > 3.84 else 0.0  # p<0.05 approx

    print(f"✓ Granger causality: {G.sum():.0f} causal links found (Non-I validation)")
    return G


# ─────────────────────────────────────────────
# 3. SAVE DATASET
# ─────────────────────────────────────────────


def save_dataset(
    output_dir,
    X,
    Y,
    edges,
    timestamps,
    node_ids,
    A_phys,
    A_sem,
    KL_matrix,
    G_causal,
    nodes_info,
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "node_features.npy"), X)
    np.save(os.path.join(output_dir, "eta_labels.npy"), Y)
    np.save(os.path.join(output_dir, "adj_physical.npy"), A_phys)
    np.save(os.path.join(output_dir, "adj_semantic.npy"), A_sem)
    np.save(os.path.join(output_dir, "non_id_ks_matrix.npy"), KL_matrix)
    np.save(os.path.join(output_dir, "granger_causality.npy"), G_causal)

    meta = {
        "node_ids": node_ids,
        "edges": edges,
        "timestamps": [str(t) for t in timestamps],
        "shape_X": list(X.shape),
        "shape_Y": list(Y.shape),
        "nodes": {nid: nodes_info[nid] for nid in node_ids if nid in nodes_info},
    }
    with open(os.path.join(output_dir, "node_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Dataset saved to '{output_dir}/'")
    print(f"   node_features.npy  : {X.shape}  → GNN input X")
    print(f"   eta_labels.npy     : {Y.shape}  → ETA prediction target")
    print(f"   adj_physical.npy   : {A_phys.shape}  → baseline adjacency")
    print(f"   adj_semantic.npy   : {A_sem.shape}  → AH-GNN learned edges")
    print(f"   non_id_ks_matrix   : KS divergence (chứng minh Non-ID cho paper)")
    print(f"   granger_causality  : causal graph (chứng minh Non-I cho paper)")


# ─────────────────────────────────────────────
# 4. MAIN
# ─────────────────────────────────────────────

# Node info cần khớp với NODES trong tomtom_collector.py
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
        "name": "Lê Văn Sỹ-Nguyễn ĐC",
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir", default="tomtom_data", help="Thư mục CSV từ collector"
    )
    parser.add_argument(
        "--output_dir", default="dataset", help="Thư mục lưu dataset GNN"
    )
    parser.add_argument(
        "--dist_km", type=float, default=1.0, help="Ngưỡng khoảng cách vật lý (km)"
    )
    args = parser.parse_args()

    print("=" * 55)
    print("  Build AH-GNN Dataset từ TomTom Data")
    print("=" * 55)

    df_flow, df_eta = load_data(args.input_dir)

    print("\n[1/5] Building node feature tensor...")
    X, timestamps, node_ids = build_node_feature_matrix(df_flow)

    print("\n[2/5] Building ETA labels...")
    Y, edges = build_eta_labels(df_eta, timestamps)

    print("\n[3/5] Building physical adjacency...")
    A_phys = build_physical_adjacency(node_ids, NODES_INFO, args.dist_km)

    print("\n[4/5] Building semantic adjacency (Non-ID analysis)...")
    A_sem, KL_matrix = build_semantic_adjacency(X, node_ids)

    print("\n[5/5] Granger causality test (Non-I analysis)...")
    G_causal = build_granger_causality(X, node_ids)

    save_dataset(
        args.output_dir,
        X,
        Y,
        edges,
        timestamps,
        node_ids,
        A_phys,
        A_sem,
        KL_matrix,
        G_causal,
        NODES_INFO,
    )


if __name__ == "__main__":
    main()
