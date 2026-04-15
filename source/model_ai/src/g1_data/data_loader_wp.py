"""
data_loader_wp.py
-----------------
G1 (WP-full variant): Đọc IBTrACS CSV và giữ TẤT CẢ bão WP (không filter SCS).
Dùng để train trên toàn bộ WP, val/test vẫn chỉ đánh giá trên bão SCS.

Output: data/processed/bao_bien_dong_wp_full.csv
  - Chứa full track của tất cả WP storms
  - Cột `in_scs` (bool): điểm có nằm trong SCS box
  - Cột `is_scs_storm` (bool): storm CÓ ít nhất 1 điểm trong SCS
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_all_wp(raw_csv: str = None, config: dict = None) -> pd.DataFrame:
    """
    Đọc IBTrACS CSV và giữ TẤT CẢ bão WP — không filter SCS box.
    Vẫn đánh dấu in_scs và is_scs_storm để dùng khi split val/test.

    Returns
    -------
    pd.DataFrame với full track của toàn WP, có cột in_scs + is_scs_storm.
    """
    if config is None:
        config = load_config()

    geo       = config["geography"]
    clean_cfg = config["cleaning"]
    feat_cfg  = config["features"]

    if raw_csv is None:
        default_raw = Path(__file__).parent.parent.parent.parent / "data" / "data_raw.csv"
        raw_csv = str(default_raw)

    print(f"[G1-WP] Đọc file: {raw_csv}")
    print(f"        SCS box: {geo['lat_min']}–{geo['lat_max']}°N, "
          f"{geo['lon_min']}–{geo['lon_max']}°E  (dùng để đánh dấu, không filter)")
    print(f"        Năm: {geo['year_start']}–{geo['year_end']}")

    # --- Đọc CSV ---
    df = pd.read_csv(raw_csv, skiprows=clean_cfg["skiprows"], low_memory=False)
    print(f"        Raw rows: {len(df):,} | Columns: {len(df.columns)}")

    # --- Chuẩn hóa kiểu dữ liệu ---
    df["LAT"]     = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"]     = pd.to_numeric(df["LON"], errors="coerce")
    df["SEASON"]  = pd.to_numeric(df["SEASON"], errors="coerce")
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for col in [
        feat_cfg["wind_primary"], feat_cfg.get("wind_cma"), feat_cfg["wind_fallback"],
        feat_cfg["pres_primary"], feat_cfg.get("pres_cma"), feat_cfg["pres_fallback"],
        "DIST2LAND",
    ]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Lọc TRACK_TYPE == 'main' ---
    df = df[df["TRACK_TYPE"].str.strip() == clean_cfg["track_type"]].copy()
    print(f"        Sau lọc track_type='main': {len(df):,} rows")

    # --- Lọc năm ---
    df = df[df["SEASON"].between(geo["year_start"], geo["year_end"])].copy()
    print(f"        Sau lọc năm {geo['year_start']}–{geo['year_end']}: {len(df):,} rows")

    # --- Loại bỏ record thiếu lat/lon ---
    df = df.dropna(subset=["LAT", "LON"]).copy()

    # --- Giữ TẤT CẢ WP storms (không filter) ---
    df_full = df.copy()

    # --- Đánh dấu SCS ---
    mask_scs = (
        df_full["LAT"].between(geo["lat_min"], geo["lat_max"]) &
        df_full["LON"].between(geo["lon_min"], geo["lon_max"])
    )
    df_full["in_scs"] = mask_scs.values

    # is_scs_storm: storm có ít nhất 1 điểm trong SCS
    scs_sids = df_full[mask_scs]["SID"].unique()
    df_full["is_scs_storm"] = df_full["SID"].isin(scs_sids)

    # --- Tạo vmax, pmin ---
    def _merge_col(primary, cma, fallback):
        s = df_full[primary].copy()
        if cma and cma in df_full.columns:
            s = s.fillna(df_full[cma])
        return s.fillna(df_full[fallback])

    df_full["vmax"] = _merge_col(feat_cfg["wind_primary"], feat_cfg.get("wind_cma"), feat_cfg["wind_fallback"])
    df_full["pmin"] = _merge_col(feat_cfg["pres_primary"], feat_cfg.get("pres_cma"), feat_cfg["pres_fallback"])

    # --- Sắp xếp ---
    df_full = df_full.sort_values(["SID", "ISO_TIME"]).reset_index(drop=True)

    # --- Thống kê ---
    n_storms     = df_full["SID"].nunique()
    n_scs_storms = df_full["is_scs_storm"].sum() // df_full.groupby("SID").size().mean()  # ước tính
    n_scs_storms = len(scs_sids)
    nan_lat      = df_full["LAT"].isna().sum()
    nan_lon      = df_full["LON"].isna().sum()

    print(f"\n[G1-WP] Kết quả:")
    print(f"        Tổng WP storms: {n_storms:,}")
    print(f"        Trong đó SCS storms: {n_scs_storms:,}  ({n_scs_storms/n_storms*100:.1f}%)")
    print(f"        Rows full track: {len(df_full):,}")
    print(f"        NaN lat/lon: {nan_lat}/{nan_lon}")
    print(f"        Seasons: {df_full['SEASON'].min():.0f}–{df_full['SEASON'].max():.0f}")

    if nan_lat > 0 or nan_lon > 0:
        raise ValueError(f"[G1-WP FAIL] Còn NaN lat/lon ({nan_lat}/{nan_lon}).")

    print(f"\n[G1-WP] PASS: {n_storms:,} WP storms, {n_scs_storms} SCS storms, 0 NaN lat/lon")
    return df_full


def save_wp_data(df: pd.DataFrame, config: dict = None) -> str:
    """Lưu full WP DataFrame vào data/processed/bao_bien_dong_wp_full.csv."""
    if config is None:
        config = load_config()
    base_dir = Path(__file__).parent.parent.parent
    out_path = base_dir / "data/processed/bao_bien_dong_wp_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[G1-WP] Đã lưu: {out_path} ({size_mb:.1f} MB)")
    return str(out_path)


if __name__ == "__main__":
    cfg = load_config()
    df  = load_all_wp(config=cfg)
    save_wp_data(df, config=cfg)
