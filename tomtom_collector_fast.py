"""
TomTom Fast Collector — Async version
======================================
Dùng asyncio + aiohttp để query 90 cặp song song
Nhanh hơn ~5-7x so với version tuần tự

Cài thêm: pip install aiohttp
Dùng    : python tomtom_collector_fast.py --hours 2 --interval_min 3
"""

import asyncio
import aiohttp
import pandas as pd
import os
import time
import argparse
from datetime import datetime
from itertools import permutations
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# NODES — Quận 1, TP.HCM
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

EDGES = list(permutations(NODES.keys(), 2))  # 90 cặp

TIME_LABELS = {
    range(0, 6): "night",
    range(7, 10): "rush_morning",
    range(16, 20): "rush_evening",
}


def get_time_label(hour):
    for r, label in TIME_LABELS.items():
        if hour in r:
            return label
    return "normal"


# ─────────────────────────────────────────────
# ASYNC ETA QUERY
# ─────────────────────────────────────────────
async def fetch_eta(session, api_key, src_id, dst_id, semaphore):
    """Query 1 cặp nút bất đồng bộ, có semaphore để giới hạn concurrent"""
    src = NODES[src_id]
    dst = NODES[dst_id]
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
    async with semaphore:
        try:
            async with session.get(
                url, params=params, timeout=aiohttp.ClientTimeout(total=15)
            ) as r:
                if r.status != 200:
                    return None
                data = await r.json()
                s = data["routes"][0]["summary"]
                tt = s.get("travelTimeInSeconds", 0)
                ff = s.get("noTrafficTravelTimeInSeconds") or 0
                dl = s.get("trafficDelayInSeconds", 0)
                lm = s.get("lengthInMeters", 0)
                if ff and ff > 0:
                    ratio = round(tt / ff, 3)
                elif tt > 0:
                    ratio = round(1 + (dl / tt), 3)
                else:
                    ratio = 1.0
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


async def collect_snapshot_async(api_key, output_dir, max_concurrent=15):
    """Thu thập 1 snapshot — tất cả 90 cặp song song (max 15 cùng lúc)"""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    time_label = get_time_label(now.hour)

    semaphore = asyncio.Semaphore(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent)

    t0 = time.time()
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [fetch_eta(session, api_key, src, dst, semaphore) for src, dst in EDGES]
        results = await asyncio.gather(*tasks)

    elapsed = time.time() - t0

    records = []
    errors = 0
    for r in results:
        if r is None:
            errors += 1
            continue
        r["timestamp"] = timestamp
        r["time_label"] = time_label
        records.append(r)

    # Sắp xếp cột cho nhất quán với file cũ
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
        df = pd.DataFrame(records)[cols]
        os.makedirs(output_dir, exist_ok=True)
        out = os.path.join(output_dir, "eta_data.csv")
        df.to_csv(out, mode="a", header=not os.path.exists(out), index=False)

    ok = len(records)
    print(
        f"  [{timestamp}] ({time_label}) "
        f"✓ {ok}/90 records  ✗ {errors} errors  ⏱ {elapsed:.1f}s"
    )
    return ok


# ─────────────────────────────────────────────
# TEST CONNECTION
# ─────────────────────────────────────────────
async def test_connection(api_key):
    sem = asyncio.Semaphore(1)
    async with aiohttp.ClientSession() as session:
        r = await fetch_eta(session, api_key, "v01", "v07", sem)
        return r is not None


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
async def main_async(args, api_key):
    print("🔍 Kiểm tra kết nối API...")
    ok = await test_connection(api_key)
    if not ok:
        print("❌ API key không hoạt động!")
        return
    print("✅ API key OK!\n")

    total = int((args.hours * 60) / args.interval_min)
    interval_s = args.interval_min * 60

    print("=" * 58)
    print("  TomTom FAST Collector (Async) — Quận 1 TP.HCM")
    print("=" * 58)
    print(f"  Edges      : {len(EDGES)} cặp (all-pairs)")
    print(f"  Concurrent : 15 requests/lúc")
    print(f"  Interval   : {args.interval_min} phút")
    print(f"  Duration   : {args.hours} giờ ({total} snapshots)")
    print(f"  API calls  : ~{total * len(EDGES)} total")
    print(f"  Tốc độ     : ~5-8s/snapshot (vs ~36s trước)")
    print(f"  Output     : {args.output_dir}/eta_data.csv")
    print("=" * 58)

    total_records = 0
    for i in range(total):
        print(f"\n[Snapshot {i+1}/{total}]")
        n = await collect_snapshot_async(api_key, args.output_dir, max_concurrent=15)
        total_records += n

        if i < total - 1:
            print(f"  ⏳ Chờ {args.interval_min} phút...")
            await asyncio.sleep(interval_s)

    print(f"\n✅ Hoàn thành! Tổng: {total_records} records mới")
    print(f"   → {args.output_dir}/eta_data.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", default=None)
    parser.add_argument(
        "--interval_min",
        type=int,
        default=3,
        help="Khoảng cách giữa snapshots (phút), default=3",
    )
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--output_dir", default="tomtom_data")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("TOMTOM_API_KEY")
    if not api_key:
        print("❌ Thiếu API key! Thêm vào .env: TOMTOM_API_KEY=your_key")
        return

    asyncio.run(main_async(args, api_key))


if __name__ == "__main__":
    main()
