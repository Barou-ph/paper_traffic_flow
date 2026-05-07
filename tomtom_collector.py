"""
TomTom Data Collector cho Paper AH-GNN ETA
==========================================
Thu thập: Traffic Flow (lưu lượng) + Travel Time (ETA)
Khu vực mặc định: Quận 1, TP.HCM

Cách dùng:
    pip install requests pandas python-dotenv
    Tạo file .env cùng thư mục: TOMTOM_API_KEY=your_key
    python tomtom_collector.py --hours 2 --interval_min 5
"""

import requests
import pandas as pd
import json
import time
import argparse
import os
from datetime import datetime
from itertools import combinations
from dotenv import load_dotenv

load_dotenv()  # đọc file .env tự động

# ─────────────────────────────────────────────
# 1. ĐỊNH NGHĨA MẠNG NÚT (Ngã tư nghiên cứu)
#    → Thay bằng tọa độ thực của khu vực bạn chọn
# ─────────────────────────────────────────────
NODES = {
    "v01": {
        "name": "Ngã tư Bến Thành",
        "lat": 10.7729,
        "lon": 106.6980,
        "poi_type": "commercial",
    },
    "v02": {
        "name": "Ngã tư Điện Biên Phủ-Hai BT",
        "lat": 10.7875,
        "lon": 106.6923,
        "poi_type": "arterial",
    },
    "v03": {
        "name": "Vòng xoay Dân Chủ",
        "lat": 10.7802,
        "lon": 106.6958,
        "poi_type": "roundabout",
    },
    "v04": {
        "name": "Ngã tư Lê Văn Sỹ-Nguyễn Đình Chiểu",
        "lat": 10.7848,
        "lon": 106.6889,
        "poi_type": "residential",
    },
    "v05": {
        "name": "Ngã tư CMT8-Điện Biên Phủ",
        "lat": 10.7867,
        "lon": 106.6850,
        "poi_type": "arterial",
    },
    "v06": {
        "name": "Cầu Sài Gòn (đầu Q1)",
        "lat": 10.7960,
        "lon": 106.7222,
        "poi_type": "bridge",
    },
    "v07": {
        "name": "Ngã tư Nguyễn Huệ-Lê Lợi",
        "lat": 10.7740,
        "lon": 106.7030,
        "poi_type": "commercial",
    },
    "v08": {
        "name": "Ngã tư Hàm Nghi-Tôn Đức Thắng",
        "lat": 10.7723,
        "lon": 106.7040,
        "poi_type": "commercial",
    },
    "v09": {
        "name": "Ngã tư Võ Văn Tần-Nam Kỳ KN",
        "lat": 10.7798,
        "lon": 106.6910,
        "poi_type": "residential",
    },
    "v10": {
        "name": "Ngã tư Pasteur-Lê Duẩn",
        "lat": 10.7822,
        "lon": 106.6978,
        "poi_type": "mixed",
    },
    # Thêm nút nếu cần (tối thiểu 20 nút cho paper)
}

# ─────────────────────────────────────────────
# 2. TOMTOM API WRAPPERS
# ─────────────────────────────────────────────


class TomTomClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_flow = (
            "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
        )
        self.base_route = "https://api.tomtom.com/routing/1/calculateRoute"
        self.session = requests.Session()

    def get_flow(self, lat: float, lon: float) -> dict:
        """
        Lấy traffic flow tại một điểm (lat, lon).
        Trả về: currentSpeed, freeFlowSpeed, currentTravelTime, confidence
        """
        params = {
            "point": f"{lat},{lon}",
            "unit": "KMPH",
            "openLr": "false",
            "key": self.api_key,
        }
        try:
            r = self.session.get(self.base_flow, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()["flowSegmentData"]
            return {
                "current_speed_kmph": data.get("currentSpeed"),
                "free_flow_speed_kmph": data.get("freeFlowSpeed"),
                "current_travel_time_s": data.get("currentTravelTime"),
                "free_flow_travel_time_s": data.get("freeFlowTravelTime"),
                "confidence": data.get("confidence"),
                "road_closure": data.get("roadClosure", False),
            }
        except Exception as e:
            print(f"  [FLOW ERROR] ({lat},{lon}): {e}")
            return {}

    def get_eta(
        self, origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float
    ) -> dict:
        """
        Lấy ETA (travel time) từ origin → destination.
        Trả về: travel_time_s, length_m, traffic_delay_s
        """
        url = f"{self.base_route}/{origin_lat},{origin_lon}:{dest_lat},{dest_lon}/json"
        params = {
            "routeType": "fastest",
            "traffic": "true",
            "travelMode": "car",
            "key": self.api_key,
        }
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            summary = r.json()["routes"][0]["summary"]
            return {
                "travel_time_s": summary.get("travelTimeInSeconds"),
                "length_m": summary.get("lengthInMeters"),
                "traffic_delay_s": summary.get("trafficDelayInSeconds"),
                "departure_time": summary.get("departureTime"),
                "arrival_time": summary.get("arrivalTime"),
            }
        except Exception as e:
            print(
                f"  [ETA ERROR] ({origin_lat},{origin_lon})→({dest_lat},{dest_lon}): {e}"
            )
            return {}


# ─────────────────────────────────────────────
# 3. DATA COLLECTION LOOP
# ─────────────────────────────────────────────


def collect_snapshot(client: TomTomClient, output_dir: str):
    """Thu thập một snapshot tại thời điểm hiện tại."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[{timestamp}] Collecting snapshot...")

    # --- 3a. Traffic Flow tại từng nút ---
    flow_records = []
    for node_id, info in NODES.items():
        flow = client.get_flow(info["lat"], info["lon"])
        if flow:
            record = {
                "timestamp": timestamp,
                "node_id": node_id,
                "node_name": info["name"],
                "lat": info["lat"],
                "lon": info["lon"],
                "poi_type": info["poi_type"],
                **flow,
            }
            flow_records.append(record)
            print(f"  ✓ Flow [{node_id}] speed={flow.get('current_speed_kmph')} kmph")
        time.sleep(0.3)  # tránh rate limit

    # --- 3b. ETA giữa các cặp nút ---
    eta_records = []
    node_ids = list(NODES.keys())
    # Chỉ lấy cặp liền kề (theo thứ tự) để tiết kiệm API call
    # Nếu muốn all-pairs: dùng combinations(node_ids, 2)
    pairs = list(zip(node_ids, node_ids[1:]))  # n-1 cặp liên tiếp

    for src_id, dst_id in pairs:
        src = NODES[src_id]
        dst = NODES[dst_id]
        eta = client.get_eta(src["lat"], src["lon"], dst["lat"], dst["lon"])
        if eta:
            record = {
                "timestamp": timestamp,
                "src_node": src_id,
                "dst_node": dst_id,
                "src_name": src["name"],
                "dst_name": dst["name"],
                **eta,
            }
            eta_records.append(record)
            print(f"  ✓ ETA  [{src_id}→{dst_id}] {eta.get('travel_time_s')}s")
        time.sleep(0.3)

    # --- 3c. Lưu vào CSV ---
    os.makedirs(output_dir, exist_ok=True)

    flow_file = os.path.join(output_dir, "flow_data.csv")
    eta_file = os.path.join(output_dir, "eta_data.csv")

    if flow_records:
        df_flow = pd.DataFrame(flow_records)
        write_header = not os.path.exists(flow_file)
        df_flow.to_csv(flow_file, mode="a", header=write_header, index=False)

    if eta_records:
        df_eta = pd.DataFrame(eta_records)
        write_header = not os.path.exists(eta_file)
        df_eta.to_csv(eta_file, mode="a", header=write_header, index=False)

    print(f"  → Saved {len(flow_records)} flow + {len(eta_records)} ETA records")
    return len(flow_records), len(eta_records)


# ─────────────────────────────────────────────
# 4. MAIN - Chạy theo lịch (mỗi N phút)
# ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TomTom Data Collector")
    parser.add_argument(
        "--api_key", default=None, help="TomTom API Key (hoặc để trong .env)"
    )
    parser.add_argument(
        "--interval_min",
        type=int,
        default=5,
        help="Khoảng thời gian giữa các lần collect (phút), default=5",
    )
    parser.add_argument(
        "--hours", type=float, default=1.0, help="Tổng số giờ chạy, default=1"
    )
    parser.add_argument("--output_dir", default="tomtom_data", help="Thư mục lưu CSV")
    args = parser.parse_args()

    # Ưu tiên: --api_key → .env → báo lỗi
    api_key = args.api_key or os.getenv("TOMTOM_API_KEY")
    if not api_key:
        print("❌ Thiếu API key! Tạo file .env với nội dung: TOMTOM_API_KEY=your_key")
        return

    client = TomTomClient(api_key)
    total_snapshots = int((args.hours * 60) / args.interval_min)
    interval_s = args.interval_min * 60

    print("=" * 55)
    print("  TomTom Collector — AH-GNN Paper Dataset Builder")
    print("=" * 55)
    print(f"  Nodes       : {len(NODES)}")
    print(f"  Interval    : {args.interval_min} min")
    print(f"  Duration    : {args.hours} hours ({total_snapshots} snapshots)")
    print(f"  Output dir  : {args.output_dir}/")
    print(
        f"  API quota   : ~{total_snapshots * (len(NODES) + len(NODES)-1)} calls total"
    )
    print("=" * 55)

    for i in range(total_snapshots):
        print(f"\n[Snapshot {i+1}/{total_snapshots}]")
        collect_snapshot(client, args.output_dir)
        if i < total_snapshots - 1:
            print(f"  Waiting {args.interval_min} min...")
            time.sleep(interval_s)

    print("\n✅ Collection complete!")
    print(f"   flow_data.csv  → dùng làm node features X(t)")
    print(f"   eta_data.csv   → dùng làm label Y (travel time)")
    print(f"   → Tiếp theo: chạy build_graph.py để tạo adjacency matrix")


if __name__ == "__main__":
    main()
