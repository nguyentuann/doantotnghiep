"""
data_loader.py
--------------
G1: Đọc IBTrACS CSV (data_raw.csv hoặc ibtracs_WP.csv), lọc ra các cơn bão
    đã đi qua Biển Đông (SCS box), lưu full track vào bao_bien_dong_clean.csv.

Lưu ý quan trọng:
  - Lấy TOÀN BỘ track của storm có đi qua SCS (kể cả đoạn ngoài SCS)
  - Chỉ giữ TRACK_TYPE == 'main', bỏ 'spur'
  - Hàng 2 của IBTrACS CSV là units row → skiprows=[1]
  - Tất cả hyperparameter đọc từ config.yaml, không hardcode
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Đọc config.yaml, mặc định tìm ở thư mục cha của src/."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_and_filter_scs(raw_csv: str = None, config: dict = None) -> pd.DataFrame:
    """
    Đọc IBTrACS CSV và lọc ra các cơn bão đi qua Biển Đông.

    Parameters
    ----------
    raw_csv : str
        Đường dẫn tới file CSV gốc. Nếu None, dùng đường dẫn mặc định
        (data_raw.csv trong source/data/).
    config : dict
        Config đã load. Nếu None, tự load từ config.yaml.

    Returns
    -------
    pd.DataFrame
        Full track của các cơn bão đi qua SCS, đã làm sạch cơ bản.
    """
    if config is None:
        config = load_config()

    geo = config["geography"]
    clean_cfg = config["cleaning"]
    feat_cfg = config["features"]

    # --- Đường dẫn file ---
    if raw_csv is None:
        # Ưu tiên dùng data_raw.csv có sẵn trong source/data/
        default_raw = Path(__file__).parent.parent.parent / "data" / "data_raw.csv"
        raw_csv = str(default_raw)

    print(f"[G1] Đọc file: {raw_csv}")
    print(f"     Phạm vi SCS: {geo['lat_min']}–{geo['lat_max']}°N, "
          f"{geo['lon_min']}–{geo['lon_max']}°E")
    print(f"     Năm: {geo['year_start']}–{geo['year_end']}")

    # --- Đọc CSV (bỏ hàng units ở dòng 2) ---
    df = pd.read_csv(raw_csv, skiprows=clean_cfg["skiprows"], low_memory=False)
    print(f"     Raw rows: {len(df):,} | Columns: {len(df.columns)}")

    # --- Chuẩn hóa kiểu dữ liệu ---
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df["SEASON"] = pd.to_numeric(df["SEASON"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    df[feat_cfg["wind_primary"]] = pd.to_numeric(df[feat_cfg["wind_primary"]], errors="coerce")
    df[feat_cfg["wind_fallback"]] = pd.to_numeric(df[feat_cfg["wind_fallback"]], errors="coerce")
    df[feat_cfg["pres_primary"]] = pd.to_numeric(df[feat_cfg["pres_primary"]], errors="coerce")
    df[feat_cfg["pres_fallback"]] = pd.to_numeric(df[feat_cfg["pres_fallback"]], errors="coerce")

    # --- Lọc TRACK_TYPE == 'main' ---
    df = df[df["TRACK_TYPE"].str.strip() == clean_cfg["track_type"]].copy()
    print(f"     Sau lọc track_type='main': {len(df):,} rows")

    # --- Lọc năm ---
    df = df[df["SEASON"].between(geo["year_start"], geo["year_end"])].copy()
    print(f"     Sau lọc năm {geo['year_start']}–{geo['year_end']}: {len(df):,} rows")

    # --- Loại bỏ record thiếu lat/lon ---
    df = df.dropna(subset=["LAT", "LON"]).copy()

    # --- QUAN TRỌNG: Lấy FULL TRACK của storm có đi qua SCS ---
    # Bước 1: tìm SID có ít nhất 1 điểm trong SCS box
    mask_scs = (
        df["LAT"].between(geo["lat_min"], geo["lat_max"]) &
        df["LON"].between(geo["lon_min"], geo["lon_max"])
    )
    storm_ids_in_scs = df[mask_scs]["SID"].unique()
    print(f"     Số storm có điểm trong SCS: {len(storm_ids_in_scs)}")

    # Bước 2: lấy toàn bộ track (kể cả đoạn ngoài SCS — cần cho lookback)
    df_full = df[df["SID"].isin(storm_ids_in_scs)].copy()
    df_full["in_scs"] = mask_scs.reindex(df_full.index, fill_value=False)
    print(f"     Rows full track (kể cả ngoài SCS): {len(df_full):,}")

    # --- Tạo cột vmax và pmin (ưu tiên WMO, fallback USA) ---
    df_full["vmax"] = df_full[feat_cfg["wind_primary"]].fillna(
        df_full[feat_cfg["wind_fallback"]]
    )
    df_full["pmin"] = df_full[feat_cfg["pres_primary"]].fillna(
        df_full[feat_cfg["pres_fallback"]]
    )

    # --- Sắp xếp theo storm và thời gian ---
    df_full = df_full.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # --- Thống kê cuối ---
    n_storms = df_full["SID"].nunique()
    n_scs_rows = df_full["in_scs"].sum()
    nan_lat = df_full["LAT"].isna().sum()
    nan_lon = df_full["LON"].isna().sum()

    print(f"\n[G1] Kết quả:")
    print(f"     Tổng storms: {n_storms}")
    print(f"     Rows trong SCS: {n_scs_rows:,}")
    print(f"     Rows full track: {len(df_full):,}")
    print(f"     NaN lat/lon: {nan_lat}/{nan_lon}")
    print(f"     Seasons: {df_full['SEASON'].min():.0f}–{df_full['SEASON'].max():.0f}")

    # --- Checkpoint G1 ---
    min_storms = config["checkpoints"]["g1_min_storms"]
    if n_storms < min_storms:
        raise ValueError(
            f"[G1 FAIL] Chỉ có {n_storms} storms, cần >= {min_storms}. "
            f"Kiểm tra lại filter hoặc file dữ liệu."
        )
    if nan_lat > 0 or nan_lon > 0:
        raise ValueError(
            f"[G1 FAIL] Còn NaN lat/lon ({nan_lat}/{nan_lon}). "
            f"Kiểm tra lại bước làm sạch."
        )
    print(f"\n[G1] PASS: {n_storms} storms >= {min_storms}, 0 NaN lat/lon")

    return df_full


def save_clean_data(df: pd.DataFrame, config: dict = None, output_path: str = None) -> str:
    """
    Lưu DataFrame đã lọc vào file CSV.

    Parameters
    ----------
    df : pd.DataFrame
    config : dict
    output_path : str, optional
        Ghi đè đường dẫn mặc định từ config.

    Returns
    -------
    str : Đường dẫn file đã lưu.
    """
    if config is None:
        config = load_config()

    if output_path is None:
        base_dir = Path(__file__).parent.parent
        output_path = base_dir / config["data"]["clean_file"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[G1] Đã lưu: {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


def load_clean_data(config: dict = None) -> pd.DataFrame:
    """Đọc bao_bien_dong_clean.csv đã lưu trước đó."""
    if config is None:
        config = load_config()
    base_dir = Path(__file__).parent.parent
    path = base_dir / config["data"]["clean_file"]
    df = pd.read_csv(path, low_memory=False, parse_dates=["ISO_TIME"])
    print(f"[load] Đọc {path.name}: {len(df):,} rows, {df['SID'].nunique()} storms")
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = load_and_filter_scs(config=cfg)
    save_clean_data(df, config=cfg)
