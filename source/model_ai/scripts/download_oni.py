"""
download_oni.py — Tải ONI (Oceanic Niño Index) từ NOAA CPC
Lưu tại: data/climate/oni.csv (year, month, oni)

ONI = 3-month running average SST anomaly in Niño-3.4 region (5°N–5°S, 120°W–170°W)
Nguồn: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
"""

import pandas as pd
from pathlib import Path

_ONI_URL = "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"

_SEAS_TO_MONTH = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
    'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
    'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12,
}


def download_oni(out_path: str = None) -> str:
    base_dir = Path(__file__).parent.parent
    if out_path is None:
        out_path = base_dir / "data/climate/oni.csv"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ONI] Tải từ NOAA CPC: {_ONI_URL}")
    df = pd.read_csv(_ONI_URL, sep=r'\s+', header=0)
    df.columns = [c.strip() for c in df.columns]

    # Handle NOAA format: cols are SEAS, YR, ANOM
    if 'SEAS' in df.columns:
        df = df.rename(columns={'SEAS': 'seas', 'YR': 'year', 'ANOM': 'oni'})
    else:
        df.columns = ['seas', 'year', 'oni']

    df['month'] = df['seas'].map(_SEAS_TO_MONTH)
    df = df.dropna(subset=['month'])
    df = df[['year', 'month', 'oni']].copy()
    df['year']  = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    df['oni']   = pd.to_numeric(df['oni'], errors='coerce').fillna(0.0)
    df = df.sort_values(['year', 'month']).reset_index(drop=True)

    df.to_csv(out_path, index=False)
    print(f"[ONI] Đã lưu: {out_path}  ({len(df)} rows, {df['year'].min()}–{df['year'].max()})")
    return str(out_path)


if __name__ == "__main__":
    download_oni()
