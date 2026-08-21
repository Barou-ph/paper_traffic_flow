"""
Build Node-level Congestion Labels từ eta_data.csv
===================================================
Chuyển từ edge-level ETA → node-level congestion [T, N, T_out=3]
Dùng cho spec mới: output [B, N, T_out]

Dùng: python build_node_labels.py
Output: dataset/node_labels.npy   [T, N, 3]
        dataset/zone_onehot.npy   [N, K]
"""

import numpy as np
import pandas as pd
import json
import os

INPUT_CSV = "tomtom_data/eta_data.csv"
DATASET_DIR = "dataset"

POI_TYPES = ["commercial", "arterial", "roundabout", "residential", "bridge", "mixed"]
T_OUT = 3  # predict 3 bước tiếp theo


def build():
    # Load
    X = np.load(f"{DATASET_DIR}/node_features.npy")  # [T, N, 9]
    with open(f"{DATASET_DIR}/node_meta.json", encoding="utf-8") as f:
        meta = json.load(f)

    node_ids = meta["node_ids"]
    N = len(node_ids)
    T = X.shape[0]
    node2idx = {n: i for i, n in enumerate(node_ids)}

    # Node-level label = avg_congestion_out (feature index 0)
    # Shape [T, N] — congestion tại mỗi nút mỗi timestep
    cong = X[:, :, 0]  # [T, N]

    # Build [T, N, T_out]: tại mỗi t, label là cong tại t+1, t+2, t+3
    T_valid = T - T_OUT
    Y_node = np.zeros((T_valid, N, T_OUT), dtype=np.float32)
    for i in range(T_valid):
        for k in range(T_OUT):
            Y_node[i, :, k] = cong[i + 1 + k, :]

    print(f"✓ Node labels Y_node: {Y_node.shape}  (T={T_valid}, N={N}, T_out={T_OUT})")
    print(f"  Mean congestion: {Y_node.mean():.3f}  Std: {Y_node.std():.3f}")

    # Zone one-hot [N, K]
    Z = np.zeros((N, len(POI_TYPES)), dtype=np.float32)
    for nid, info in meta["nodes"].items():
        i = node2idx[nid]
        poi = info["poi_type"]
        if poi in POI_TYPES:
            Z[i, POI_TYPES.index(poi)] = 1.0
    print(f"✓ Zone one-hot Z: {Z.shape}")

    # GPS coords [N, 2]
    gps = np.array(
        [[meta["nodes"][nid]["lat"], meta["nodes"][nid]["lon"]] for nid in node_ids],
        dtype=np.float32,
    )
    print(f"✓ GPS coords: {gps.shape}")

    # Save
    np.save(f"{DATASET_DIR}/node_labels.npy", Y_node)
    np.save(f"{DATASET_DIR}/zone_onehot.npy", Z)
    np.save(f"{DATASET_DIR}/gps_coords.npy", gps)

    # Update splits (T giảm đi T_OUT)
    T_new = T_valid
    splits = {
        "train": [0, int(T_new * 0.70)],
        "val": [int(T_new * 0.70), int(T_new * 0.85)],
        "test": [int(T_new * 0.85), T_new],
    }
    meta["splits_node"] = splits
    meta["T_node"] = T_new
    meta["T_out"] = T_OUT
    meta["n_zone_types"] = len(POI_TYPES)
    meta["poi_types"] = POI_TYPES
    with open(f"{DATASET_DIR}/node_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved:")
    print(f"   dataset/node_labels.npy  {Y_node.shape}")
    print(f"   dataset/zone_onehot.npy  {Z.shape}")
    print(f"   dataset/gps_coords.npy   {gps.shape}")
    print(
        f"   Train/Val/Test: {splits['train'][1]} / "
        f"{splits['val'][1]-splits['val'][0]} / "
        f"{splits['test'][1]-splits['test'][0]} snapshots"
    )
    print(f"\n→ Tiếp theo: python train_model.py")


if __name__ == "__main__":
    build()
