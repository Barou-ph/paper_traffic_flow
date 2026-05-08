"""
TomTom Data Collector — chỉ dùng Routing API (Evaluation tier)
==============================================================
Thu thập: Travel Time (ETA) giữa tất cả các cặp nút Quận 1
Output : tomtom_data/eta_data.csv

Cách dùng:
    pip install requests pandas python-dotenv
    Tạo file .env: TOMTOM_API_KEY=your_key
    python tomtom_collector.py --hours 2 --interval_min 5
"""

import requests
import pandas as pd
import time
import argparse
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# MẠNG NÚT — Quận 1, TP.HCM
# ─────────────────────────────────────────────
NODES = {
    "v01": {
        "name": "Bến Thành",
        "lat": 10.7729,
        "lon": 106.6980,
        "poi_type": "commercial",
    },
    "v02": {
        "name": "Điện Biên Phủ - Hai BT",
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
        "name": "Lê Văn Sỹ - N.Đ.Chiểu",
        "lat": 10.7848,
        "lon": 106.6889,
        "poi_type": "residential",
    },
    "v05": {
        "name": "CMT8 - Điện Biên Phủ",
        "lat": 10.7867,
        "lon": 106.6850,
        "poi_type": "arterial",
    },
    "v06": {
        "name": "Cầu Sài Gòn",
        "lat": 10.7960,
        "lon": 106.7222,
        "poi_type": "bridge",
    },
    "v07": {
        "name": "Nguyễn Huệ - Lê Lợi",
        "lat": 10.7740,
        "lon": 106.7030,
        "poi_type": "commercial",
    },
    "v08": {
        "name": "Hàm Nghi - Tôn Đức Thắng",
        "lat": 10.7723,
        "lon": 106.7040,
        "poi_type": "commercial",
    },
    "v09": {
        "name": "Võ Văn Tần - Nam Kỳ KN",
        "lat": 10.7798,
        "lon": 106.6910,
        "poi_type": "residential",
    },
    "v10": {
        "name": "Pasteur - Lê Duẩn",
        "lat": 10.7822,
        "lon": 106.6978,
        "poi_type": "mixed",
    },
}

# Danh sách cặp nút liền kề (edge trong đồ thị)
EDGES = [
    ("v01", "v03"),
    ("v03", "v10"),
    ("v10", "v02"),
    ("v02", "v04"),
    ("v04", "v05"),
    ("v09", "v03"),
    ("v09", "v10"),
    ("v07", "v08"),
    ("v08", "v01"),
    ("v06", "v07"),
    ("v01", "v07"),
    ("v03", "v09"),
    ("v05", "v02"),
    ("v10", "v01"),
    ("v07", "v01"),
]


# ─────────────────────────────────────────────
# TOMTOM ROUTING CLIENT
# ─────────────────────────────────────────────
class TomTomClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://api.tomtom.com/routing/1/calculateRoute"
        self.session = requests.Session()

    def get_eta(self, src_lat, src_lon, dst_lat, dst_lon) -> dict:
        url = f"{self.base}/{src_lat},{src_lon}:{dst_lat},{dst_lon}/json"
        params = {
            "routeType": "fastest",
            "traffic": "true",
            "travelMode": "car",
            "computeTravelTimeFor": "all",
            "key": self.api_key,
        }
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            s = r.json()["routes"][0]["summary"]
            return {
                "travel_time_s": s.get("travelTimeInSeconds"),
                "free_flow_time_s": s.get("noTrafficTravelTimeInSeconds"),
                "traffic_delay_s": s.get("trafficDelayInSeconds"),
                "length_m": s.get("lengthInMeters"),
            }
        except Exception as e:
            print(f"    [ETA ERROR] {e}")
            return {}

    def test_connection(self) -> bool:
        """Kiểm tra key có hoạt động không"""
        src = NODES["v01"]
        dst = NODES["v07"]
        result = self.get_eta(src["lat"], src["lon"], dst["lat"], dst["lon"])
        return bool(result)


# ─────────────────────────────────────────────
# COLLECT MỘT SNAPSHOT
# ─────────────────────────────────────────────
def collect_snapshot(client: TomTomClient, output_dir: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    hour = datetime.now().hour

    # Gán nhãn khung giờ (dùng làm feature phân tích Non-IID)
    if 7 <= hour <= 9:
        time_label = "rush_morning"
    elif 16 <= hour <= 19:
        time_label = "rush_evening"
    elif 0 <= hour <= 5:
        time_label = "night"
    else:
        time_label = "normal"

    print(f"\n  [{timestamp}] ({time_label})")

    records = []
    for src_id, dst_id in EDGES:
        src = NODES[src_id]
        dst = NODES[dst_id]
        eta = client.get_eta(src["lat"], src["lon"], dst["lat"], dst["lon"])

        if eta:
            tt = eta.get("travel_time_s") or 0
            ff = eta.get("free_flow_time_s")
            delay = eta.get("traffic_delay_s") or 0
            # Nếu free_flow_time có giá trị thực → tính ratio
            # Nếu không (Evaluation tier) → dùng delay/travel_time làm proxy
            if ff and ff > 0:
                congestion = round(tt / ff, 3)
            elif tt > 0:
                congestion = round(1 + (delay / tt), 3)
            else:
                congestion = 1.0

            records.append(
                {
                    "timestamp": timestamp,
                    "time_label": time_label,
                    "src_node": src_id,
                    "dst_node": dst_id,
                    "src_name": src["name"],
                    "dst_name": dst["name"],
                    "src_poi": src["poi_type"],
                    "dst_poi": dst["poi_type"],
                    "travel_time_s": eta["travel_time_s"],
                    "free_flow_time_s": eta["free_flow_time_s"],
                    "traffic_delay_s": eta["traffic_delay_s"],
                    "length_m": eta["length_m"],
                    "congestion_ratio": congestion,
                }
            )
            print(
                f"    ✓ {src_id}→{dst_id}  {eta['travel_time_s']}s  "
                f"(delay={eta['traffic_delay_s']}s, ratio={congestion})"
            )
        time.sleep(0.4)  # tránh rate limit

    # Lưu CSV
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "eta_data.csv")
    if records:
        df = pd.DataFrame(records)
        write_header = not os.path.exists(out_file)
        df.to_csv(out_file, mode="a", header=write_header, index=False)
        print(f"    → {len(records)} records saved → {out_file}")

    return len(records)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--interval_min", type=int, default=5)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--output_dir", default="tomtom_data")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("TOMTOM_API_KEY")
    if not api_key:
        print("❌ Thiếu API key! Thêm vào .env: TOMTOM_API_KEY=your_key")
        return

    client = TomTomClient(api_key)

    # Test kết nối trước
    print("🔍 Kiểm tra kết nối API...")
    if not client.test_connection():
        print("❌ API key không hoạt động. Kiểm tra lại trên TomTom portal.")
        return
    print("✅ API key OK!\n")

    total = int((args.hours * 60) / args.interval_min)
    interval_s = args.interval_min * 60
    n_edges = len(EDGES)

    print("=" * 55)
    print("  TomTom ETA Collector — Quận 1 TP.HCM")
    print("=" * 55)
    print(f"  Nodes    : {len(NODES)} nút")
    print(f"  Edges    : {n_edges} cặp")
    print(f"  Interval : {args.interval_min} phút")
    print(f"  Duration : {args.hours} giờ ({total} snapshots)")
    print(f"  API calls: ~{total * n_edges} total")
    print(f"  Output   : {args.output_dir}/eta_data.csv")
    print("=" * 55)

    for i in range(total):
        print(f"\n[Snapshot {i+1}/{total}]")
        collect_snapshot(client, args.output_dir)
        if i < total - 1:
            print(f"  ⏳ Chờ {args.interval_min} phút...")
            time.sleep(interval_s)

    print("\n✅ Hoàn thành!")
    print(f"   → File: {args.output_dir}/eta_data.csv")
    print(f"   → Tiếp theo: chạy build_graph.py để tạo tensor cho GNN")


if __name__ == "__main__":
    main()
