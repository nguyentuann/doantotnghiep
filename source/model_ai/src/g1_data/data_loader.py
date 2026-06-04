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

import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """Đọc config.yaml, mặc định tìm ở thư mục gốc model_ai/."""
    if config_path is None:
        # src/g1_data/data_loader.py → .parent = g1_data/ → .parent = src/ → .parent = model_ai/
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
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
        default_raw = Path(__file__).parent.parent.parent.parent / "data" / "data_raw.csv"
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
    for col in [feat_cfg["wind_primary"], feat_cfg.get("wind_cma"), feat_cfg["wind_fallback"],
                feat_cfg["pres_primary"], feat_cfg.get("pres_cma"), feat_cfg["pres_fallback"],
                "DIST2LAND"]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Lọc TRACK_TYPE == 'main' ---
    df = df[df["TRACK_TYPE"].str.strip() == clean_cfg["track_type"]].copy()
    print(f"     Sau lọc track_type='main': {len(df):,} rows")

    # --- Lọc năm ---
    df = df[df["SEASON"].between(geo["year_start"], geo["year_end"])].copy()
    print(f"     Sau lọc năm {geo['year_start']}–{geo['year_end']}: {len(df):,} rows")

    # --- Loại bỏ record thiếu lat/lon ---
    df = df.dropna(subset=["LAT", "LON"]).copy()

    # --- Lấy TOÀN BỘ WP storms (không lọc SCS) ---
    # Giữ in_scs để dùng cho evaluation riêng bão Biển Đông
    mask_scs = (
        df["LAT"].between(geo["lat_min"], geo["lat_max"]) &
        df["LON"].between(geo["lon_min"], geo["lon_max"])
    )
    # ERA5 coverage bounds — sequences ngoài vùng này sẽ bị lọc ở G3
    ERA5_LAT_MIN, ERA5_LAT_MAX = -5.0, 35.0
    ERA5_LON_MIN, ERA5_LON_MAX = 95.0, 145.0
    mask_era5 = (
        df["LAT"].between(ERA5_LAT_MIN, ERA5_LAT_MAX) &
        df["LON"].between(ERA5_LON_MIN, ERA5_LON_MAX)
    )
    storm_ids_in_scs = df[mask_scs]["SID"].unique()
    print(f"     Số storm có điểm trong SCS: {len(storm_ids_in_scs)}")
    print(f"     Tổng storms WP: {df['SID'].nunique()}")

    # Lấy toàn bộ WP — không lọc SCS
    df_full = df.copy()
    df_full["in_scs"]  = mask_scs.reindex(df_full.index, fill_value=False)
    df_full["in_era5"] = mask_era5.reindex(df_full.index, fill_value=False)
    print(f"     Rows toàn WP: {len(df_full):,}")

    # --- Tạo cột vmax và pmin (ưu tiên WMO → CMA → USA) ---
    def _merge_col(primary, cma, fallback):
        s = df_full[primary].copy()
        if cma and cma in df_full.columns:
            s = s.fillna(df_full[cma])
        return s.fillna(df_full[fallback])

    df_full["vmax"] = _merge_col(feat_cfg["wind_primary"], feat_cfg.get("wind_cma"), feat_cfg["wind_fallback"])
    df_full["pmin"] = _merge_col(feat_cfg["pres_primary"], feat_cfg.get("pres_cma"), feat_cfg["pres_fallback"])

    # --- Sắp xếp theo storm và thời gian ---
    df_full = df_full.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # --- Downsample về synoptic 6h (00/06/12/18 UTC) ---
    # IBTrACS WP chứa 3h data từ JMA — giữ lại 3h sẽ làm sai dlat/dlon/speed
    rows_before = len(df_full)
    df_full = df_full[df_full["ISO_TIME"].dt.hour.isin([0, 6, 12, 18])].copy()
    df_full = df_full.reset_index(drop=True)
    print(f"     Sau 6h downsample: {len(df_full):,} rows (bỏ {rows_before - len(df_full):,} records 3h)")

    # --- Thống kê cuối ---
    n_storms     = df_full["SID"].nunique()
    n_scs_storms = df_full[df_full["in_scs"]]["SID"].nunique()
    n_scs_rows   = df_full["in_scs"].sum()
    n_era5_rows  = df_full["in_era5"].sum()
    nan_lat = df_full["LAT"].isna().sum()
    nan_lon = df_full["LON"].isna().sum()

    print(f"\n[G1] Kết quả:")
    print(f"     Tổng storms WP: {n_storms}")
    print(f"     Storms có điểm trong SCS: {n_scs_storms}")
    print(f"     Rows trong SCS: {n_scs_rows:,}")
    print(f"     Rows trong ERA5 bounds: {n_era5_rows:,}")
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
    """Lưu DataFrame đã lọc vào file CSV."""
    if config is None:
        config = load_config()

    if output_path is None:
        base_dir = Path(__file__).parent.parent.parent  # model_ai/
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
    base_dir = Path(__file__).parent.parent.parent  # model_ai/
    path = base_dir / config["data"]["clean_file"]
    df = pd.read_csv(path, low_memory=False, parse_dates=["ISO_TIME"])
    print(f"[load] Đọc {path.name}: {len(df):,} rows, {df['SID'].nunique()} storms")
    return df


if __name__ == "__main__":
    cfg = load_config()
    df = load_and_filter_scs(config=cfg)
    save_clean_data(df, config=cfg)
