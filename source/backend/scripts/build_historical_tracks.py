# -*- coding: utf-8 -*-
"""
build_historical_tracks.py — B4: Sinh historical_tracks.json từ test set 2021–2024
-----------------------------------------------------------------------------------
Chạy:
  cd source/backend
  python -X utf8 scripts/build_historical_tracks.py

Output:
  data/historical_tracks.json — danh sách 40 bão test set, mỗi bão có full track
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent                        # source/backend/
MODEL_DIR  = BASE_DIR.parent / "model_ai"
CLEAN_CSV  = MODEL_DIR / "data/processed/bao_bien_dong_clean.csv"
OUT_FILE   = BASE_DIR / "data/historical_tracks.json"

TEST_YEARS = (2021, 2024)


def _coerce_float(val):
    try:
        v = float(val)
        return None if math.isnan(v) else round(v, 2)
    except (TypeError, ValueError):
        return None


def build(df: pd.DataFrame) -> list[dict]:
    storms = []

    for sid, grp in df.groupby("SID"):
        grp = grp.sort_values("ISO_TIME").reset_index(drop=True)

        name   = str(grp["NAME"].iloc[0]).strip()
        season = int(grp["SEASON"].iloc[0])
        basin  = str(grp["BASIN"].iloc[0]).strip()

        track = []
        for _, row in grp.iterrows():
            track.append({
                "iso_time": str(row["ISO_TIME"]),
                "lat":      round(float(row["LAT"]), 4),
                "lon":      round(float(row["LON"]), 4),
                "vmax":     _coerce_float(row["vmax"]),
                "pmin":     _coerce_float(row["pmin"]),
                "in_scs":   bool(row["in_scs"]),
            })

        # Cutoff: index đầu tiên bão vào SCS
        # Cần ít nhất 8 điểm trước cutoff làm input model (lookback=8)
        scs_indices = [i for i, p in enumerate(track) if p["in_scs"]]
        if scs_indices:
            raw_cutoff = scs_indices[0]
            cutoff_index = max(raw_cutoff, 8)  # đảm bảo có đủ 8 điểm lookback
        else:
            cutoff_index = min(8, len(track) - 1)

        storms.append({
            "sid":          sid,
            "name":         name if name not in ("", "NOT_NAMED", "UNNAMED") else f"Storm {sid[-6:]}",
            "season":       season,
            "basin":        basin,
            "track":        track,
            "cutoff_index": cutoff_index,  # index điểm bắt đầu dự đoán
        })

    # Sắp xếp theo season rồi theo thời điểm bắt đầu
    storms.sort(key=lambda s: (s["season"], s["track"][0]["iso_time"]))
    return storms


def main():
    print(f"[B4] Đọc: {CLEAN_CSV}")
    df = pd.read_csv(CLEAN_CSV, low_memory=False)
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")

    test_df = df[df["SEASON"].between(*TEST_YEARS)].copy()
    print(f"[B4] Test set: {test_df['SID'].nunique()} bão | {len(test_df):,} rows "
          f"({TEST_YEARS[0]}–{TEST_YEARS[1]})")

    storms = build(test_df)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(storms, f, ensure_ascii=False, indent=2)

    size_kb = OUT_FILE.stat().st_size / 1024
    print(f"[B4] Đã lưu: {OUT_FILE}  ({size_kb:.1f} KB, {len(storms)} bão)")

    # In mẫu
    print(f"\n  Mẫu 3 bão đầu:")
    for s in storms[:3]:
        ci = s['cutoff_index']
        print(f"    {s['sid']}  {s['name']:<12}  {s['season']}  {len(s['track'])} điểm  cutoff={ci}")


if __name__ == "__main__":
    main()
