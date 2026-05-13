"""
TomTom Fast Collector v3 — Sync + ThreadPool
=============================================
Dùng ThreadPoolExecutor để chạy song song
Ổn định hơn async, ít bị rate limit hơn
Tốc độ: ~8-12s/snapshot (90 cặp)

Cài: pip install requests pandas python-dotenv
Dùng: python tomtom_collector_fast.py --hours 2 --interval_min 3
"""

import requests
import pandas as pd
import os, time, argparse
from datetime import datetime
from itertools import permutations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

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
EDGES = list(permutations(NODES.keys(), 2))  # 90 cặp


def get_time_label(hour):
    if hour in range(0, 6):
        return "night"
    if hour in range(7, 10):
        return "rush_morning"
    if hour in range(16, 20):
        return "rush_evening"
    return "normal"


# ─────────────────────────────────────────────
# FETCH 1 CẶP (chạy trong thread riêng)
# ─────────────────────────────────────────────
def fetch_eta(api_key, src_id, dst_id):
    src, dst = NODES[src_id], NODES[dst_id]
    url = (
        f"https://api.tomtom.com/routing/1/calculateRoute/"
        f"{src['lat']},{src['lon']}:{dst['lat']},{dst['lon']}/json"
    )
    params = {
        "routeType": "fastest",
        "traffic": "true",
        "travelMode": "car",
        "key": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        s = r.json()["routes"][0]["summary"]
        tt = s.get("travelTimeInSeconds", 0)
        ff = s.get("noTrafficTravelTimeInSeconds") or 0
        dl = s.get("trafficDelayInSeconds", 0)
        lm = s.get("lengthInMeters", 0)
        ratio = (
            round(tt / ff, 3) if ff > 0 else round(1 + dl / tt, 3) if tt > 0 else 1.0
        )
        return {
            "src_node": src_id,
            "dst_node": dst_id,
            "src_name": src["name"],
            "dst_name": dst["name"],
            "src_poi": src["poi_type"],
            "dst_poi": dst["poi_type"],
            "travel_time_s": tt,
            "free_flow_time_s": ff,
            "traffic_delay_s": dl,
            "length_m": lm,
            "congestion_ratio": ratio,
        }
    except Exception:
        return None


# ─────────────────────────────────────────────
# COLLECT 1 SNAPSHOT với ThreadPool
# ─────────────────────────────────────────────
def collect_snapshot(api_key, output_dir, n_workers=8):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    time_label = get_time_label(now.hour)
    t0 = time.time()

    records = []
    errors = 0

    # Chia 90 cặp thành batch, mỗi batch 10 cặp, delay nhỏ giữa batch
    batch_size = 10
    batches = [EDGES[i : i + batch_size] for i in range(0, len(EDGES), batch_size)]

    for batch in batches:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(fetch_eta, api_key, s, d): (s, d) for s, d in batch}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    r["timestamp"] = timestamp
                    r["time_label"] = time_label
                    records.append(r)
                else:
                    errors += 1
        time.sleep(0.3)  # delay nhỏ giữa batch để tránh rate limit

    # Lưu CSV
    cols = [
        "timestamp",
        "time_label",
        "src_node",
        "dst_node",
        "src_name",
        "dst_name",
        "src_poi",
        "dst_poi",
        "travel_time_s",
        "free_flow_time_s",
        "traffic_delay_s",
        "length_m",
        "congestion_ratio",
    ]
    if records:
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "eta_data.csv")
        pd.DataFrame(records)[cols].to_csv(
            out, mode="a", header=not os.path.exists(out), index=False
        )

    elapsed = time.time() - t0
    print(
        f"  [{timestamp}] ({time_label}) "
        f"✓ {len(records)}/90  ✗ {errors} errors  ⏱ {elapsed:.1f}s"
    )
    return len(records)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--interval_min", type=int, default=3)
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--output_dir", default="tomtom_data")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("TOMTOM_API_KEY")
    if not api_key:
        print("❌ Thiếu API key!")
        return

    # Test
    print("🔍 Kiểm tra API...")
    test = fetch_eta(api_key, "v01", "v07")
    if not test:
        print("❌ API key không hoạt động!")
        return
    print("✅ OK!\n")

    total = int((args.hours * 60) / args.interval_min)
    interval_s = args.interval_min * 60

    print("=" * 55)
    print("  TomTom Collector v3 (ThreadPool) — Quận 1")
    print("=" * 55)
    print(f"  Edges    : {len(EDGES)} cặp | Workers: 8 threads")
    print(f"  Interval : {args.interval_min} phút | Duration: {args.hours}h")
    print(f"  Snapshots: {total} | Records: ~{total*90}")
    print("=" * 55)

    total_rec = 0
    for i in range(total):
        print(f"\n[Snapshot {i+1}/{total}]")
        total_rec += collect_snapshot(api_key, args.output_dir)
        if i < total - 1:
            print(f"  ⏳ Chờ {args.interval_min} phút...")
            time.sleep(interval_s)

    print(f"\n✅ Xong! Tổng: {total_rec} records")


if __name__ == "__main__":
    main()
