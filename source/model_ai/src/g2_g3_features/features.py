"""
features.py
-----------
G2: Tính feature engineering từ bao_bien_dong_clean.csv
G3: Tạo sliding window sequences (X, y) và chuẩn hóa dữ liệu

Features (19) — Sprint 1:
  lat_norm, lon_norm      — vị trí normalize về [0,1] trong SCS box
  dlat, dlon              — thay đổi vị trí so với bước trước (degrees/6h)
  speed_kmh               — tốc độ di chuyển (Haversine / 6h)
  direction               — hướng di chuyển (0–360°, 0=Bắc)
  vmax                    — gió cực đại (kt)
  pmin                    — áp suất cực tiểu (hPa)
  sst_actual              — SST thực tế (NOAA OISST, fallback climatology)
  month_sin, month_cos    — mã hóa tuần hoàn tháng
  storm_age_h             — tuổi cơn bão (giờ kể từ điểm đầu tiên)
  dist2land               — khoảng cách đến bờ (km)
  wind_shear              — wind shear 200–850 hPa từ ERA5 (m/s)
  u500, v500              — U/V-wind 500 hPa tại tâm bão (steering flow)
  u700, v700              — U/V-wind 700 hPa tại tâm bão (steering flow)
  z500                    — Geopotential height 500 hPa (m)

Targets (16) — multi-horizon, Sprint 1:
  lat_6h,  lon_6h         — vị trí sau 6h  (bước +1)
  lat_12h, lon_12h        — vị trí sau 12h (bước +2)
  lat_18h, lon_18h        — vị trí sau 18h (bước +3)
  lat_24h, lon_24h        — vị trí sau 24h (bước +4)  ← primary metric
  lat_30h, lon_30h        — vị trí sau 30h (bước +5)
  lat_36h, lon_36h        — vị trí sau 36h (bước +6)
  lat_42h, lon_42h        — vị trí sau 42h (bước +7)
  lat_48h, lon_48h        — vị trí sau 48h (bước +8)  ← secondary metric
"""

import math
import pickle
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        # src/g2_g3_features/features.py → .parent = g2_g3_features/ → .parent = src/ → .parent = model_ai/
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Hàm tính địa lý
# ---------------------------------------------------------------------------

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Khoảng cách Haversine (km) giữa 2 điểm."""
    r = EARTH_R_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Góc phương vị (bearing) từ điểm 1 → điểm 2, 0°=Bắc, chiều kim đồng hồ."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ---------------------------------------------------------------------------
# SST climatology
# Nguồn tham chiếu: NOAA OI SST v2, vùng SCS (8–22°N, 102–120°E)
# Dùng khi không có file SST NetCDF (.nc)
# ---------------------------------------------------------------------------

# SST trung bình tháng (°C) cho vùng trung tâm Biển Đông
_SST_CLIM_MONTHLY = {
    1: 26.2, 2: 26.0, 3: 26.8, 4: 28.0, 5: 29.2, 6: 30.1,
    7: 30.3, 8: 30.2, 9: 29.5, 10: 28.4, 11: 27.5, 12: 26.8
}


def sst_climatology(month: int) -> float:
    """Trả về SST climatology (°C) theo tháng cho vùng Biển Đông."""
    return _SST_CLIM_MONTHLY.get(int(month), 28.0)


# ---------------------------------------------------------------------------
# G2: Tính features cho từng storm
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Tính 12 features cho mỗi hàng trong DataFrame.
    Input: bao_bien_dong_clean.csv (đã lọc từ G1)
    Output: DataFrame có thêm các cột feature
    """
    geo = config["geography"]
    feat_cfg = config["features"]
    dt_h = config["data"]["time_step_hours"]  # 6h

    lat_range = geo["lat_max"] - geo["lat_min"]  # 14
    lon_range = geo["lon_max"] - geo["lon_min"]  # 18

    print("[G2] Tính features...")
    records = []

    for sid, grp in df.groupby("SID"):
        grp = grp.sort_values("ISO_TIME").reset_index(drop=True)
        n = len(grp)

        # Cache arrays để tránh gọi iloc trong vòng lặp
        lats_arr   = grp["LAT"].values
        lons_arr   = grp["LON"].values
        times_arr  = grp["ISO_TIME"].values
        first_time = grp["ISO_TIME"].iloc[0]
        vmax_arr   = grp.get("vmax", pd.Series([np.nan]*n)).values
        pmin_arr   = grp.get("pmin", pd.Series([np.nan]*n)).values
        dist_arr   = pd.to_numeric(grp.get("DIST2LAND", pd.Series([np.nan]*n)), errors="coerce").values

        for i in range(n):
            lat = float(lats_arr[i])
            lon = float(lons_arr[i])
            t   = pd.Timestamp(times_arr[i])
            month = t.month if pd.notna(t) else 8

            # --- Normalize vị trí ---
            lat_norm = (lat - geo["lat_min"]) / lat_range
            lon_norm = (lon - geo["lon_min"]) / lon_range

            # --- Tốc độ di chuyển & hướng (so với bước TRƯỚC) ---
            if i == 0:
                dlat      = 0.0
                dlon      = 0.0
                speed_kmh = 0.0
                direction = 0.0
            else:
                prev_lat = float(lats_arr[i - 1])
                prev_lon = float(lons_arr[i - 1])
                dlat = lat - prev_lat
                dlon = lon - prev_lon
                try:
                    dist_km   = haversine_km(prev_lat, prev_lon, lat, lon)
                    speed_kmh = dist_km / dt_h
                    direction = bearing_deg(prev_lat, prev_lon, lat, lon)
                except Exception:
                    speed_kmh = 0.0
                    direction = 0.0

            # --- Gió & Áp suất ---
            vmax = float(vmax_arr[i]) if not np.isnan(vmax_arr[i]) else np.nan
            pmin = float(pmin_arr[i]) if not np.isnan(pmin_arr[i]) else np.nan

            # --- SST climatology ---
            sst_c = sst_climatology(month)

            # --- Mã hóa tuần hoàn tháng ---
            month_sin = math.sin(2 * math.pi * month / 12)
            month_cos = math.cos(2 * math.pi * month / 12)

            # --- Tuổi cơn bão (giờ) ---
            if pd.notna(t) and pd.notna(first_time):
                storm_age_h = (t - pd.Timestamp(first_time)).total_seconds() / 3600
            else:
                storm_age_h = i * dt_h

            # --- Khoảng cách đến bờ (km) ---
            dist2land = float(dist_arr[i]) if not np.isnan(dist_arr[i]) else np.nan

            # --- Beta drift (Holland 1983) — physics prior ---
            # V_beta ≈ 2.5 m/s × cos(lat) (mạnh hơn ở vĩ độ thấp)
            # Hướng ~320° từ Bắc (NW), đặc trưng cho TC ở Bắc Bán Cầu
            beta_speed_ms = 2.5 * math.cos(math.radians(lat))   # m/s
            beta_dist_km  = beta_speed_ms * 3.6 * dt_h          # km / 6h
            beta_dir_rad  = math.radians(320.0)
            beta_lat_drift = beta_dist_km * math.cos(beta_dir_rad) / 111.0
            beta_lon_drift = (beta_dist_km * math.sin(beta_dir_rad)
                              / (111.0 * max(math.cos(math.radians(lat)), 0.1)))

            season_val = grp["SEASON"].iloc[0] if "SEASON" in grp.columns else np.nan
            in_scs_val = bool(grp["in_scs"].iloc[i]) if "in_scs" in grp.columns else False

            # Buffer box: ±3° quanh SCS để training mở rộng (scs_v10+)
            # SCS: 8-22°N, 102-120°E  →  Buffer: 5-25°N, 99-123°E
            in_buffer_val = (5.0 <= lat <= 25.0) and (99.0 <= lon <= 123.0)

            records.append({
                "SID":       sid,
                "SEASON":    season_val,
                "ISO_TIME":  t,
                "LAT":       lat,
                "LON":       lon,
                "in_scs":    in_scs_val,
                "in_buffer": in_buffer_val,
                # 14 features (sẽ được bổ sung thêm 5 steering bởi extract_multi_level_features)
                "lat_norm":    lat_norm,
                "lon_norm":    lon_norm,
                "dlat":        dlat,
                "dlon":        dlon,
                "speed_kmh":   speed_kmh,
                "direction":   direction,
                "vmax":        vmax,
                "pmin":        pmin,
                "sst_c":       sst_c,   # giữ lại để era5_extractor dùng làm fallback
                "sst_actual":  sst_c,   # sẽ được overwrite bởi extract_era5_features()
                "month_sin":   month_sin,
                "month_cos":   month_cos,
                "storm_age_h": storm_age_h,
                "dist2land":   dist2land,
                # Beta drift physics prior (Holland 1983, Carr & Elsberry 1990)
                "beta_lat_drift": beta_lat_drift,   # deg/6h, poleward
                "beta_lon_drift": beta_lon_drift,   # deg/6h, westward
                # Placeholders cho steering features (Sprint 1/2)
                # sẽ được overwrite bởi extract_multi_level_features()
                "u500": 0.0,
                "v500": 0.0,
                "u700": 0.0,
                "v700": 0.0,
                "z500": 0.0,
                "u200": 0.0,
                "v200": 0.0,
                "u850": 0.0,
                "v850": 0.0,
                "vort500": 0.0,   # relative vorticity 500 hPa (×10⁵ s⁻¹)
                "vort850": 0.0,   # relative vorticity 850 hPa
                # Placeholder cho ENSO — sẽ được overwrite bởi add_enso_feature()
                "enso_oni": 0.0,
            })

    feat_df = pd.DataFrame(records)

    # --- Xử lý NaN vmax/pmin: interpolate trong từng storm ---
    for col in ["vmax", "pmin"]:
        feat_df[col] = feat_df.groupby("SID")[col].transform(
            lambda s: s.interpolate(method="linear").ffill().bfill()
        )

    # --- Nếu vẫn còn NaN → fill median (chỉ các cột đã có trong DataFrame) ---
    existing_cols = [c for c in feat_cfg["names"] if c in feat_df.columns]
    for col in existing_cols:
        if feat_df[col].isna().any():
            median_val = feat_df[col].median()
            feat_df[col] = feat_df[col].fillna(median_val)
            print(f"     [warn] {col}: còn NaN → fill median ({median_val:.2f})")

    print(f"[G2] Tổng rows feature: {len(feat_df):,} | Storms: {feat_df['SID'].nunique()}")
    print(f"[G2] NaN còn lại: {feat_df[existing_cols].isna().sum().sum()}")
    return feat_df


# ---------------------------------------------------------------------------
# ENSO / ONI feature
# ---------------------------------------------------------------------------

_SEAS_TO_MONTH = {
    'DJF': 1, 'JFM': 2, 'FMA': 3, 'MAM': 4,
    'AMJ': 5, 'MJJ': 6, 'JJA': 7, 'JAS': 8,
    'ASO': 9, 'SON': 10, 'OND': 11, 'NDJ': 12,
}


def _load_oni(base_dir: Path) -> pd.DataFrame | None:
    """Đọc oni.csv nếu tồn tại, trả về DataFrame (year, month, oni)."""
    oni_path = base_dir / "data/climate/oni.csv"
    if not oni_path.exists():
        return None
    df = pd.read_csv(oni_path)
    df['year']  = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    return df


def add_enso_feature(feat_df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """
    Thêm cột enso_oni vào DataFrame.
    - Nguồn: data/climate/oni.csv (year, month, oni)
    - Fallback: 0.0 (neutral) nếu không có file hoặc năm ngoài phạm vi
    """
    feat_df = feat_df.copy()
    oni_df  = _load_oni(base_dir)

    if oni_df is None:
        print("[ENSO] Không tìm thấy data/climate/oni.csv — enso_oni=0.0. "
              "Chạy: python scripts/download_oni.py")
        feat_df["enso_oni"] = 0.0
        return feat_df

    # Build lookup: (year, month) → oni
    oni_map = {(int(r.year), int(r.month)): float(r.oni)
               for r in oni_df.itertuples(index=False)}

    times   = pd.to_datetime(feat_df["ISO_TIME"])
    years   = times.dt.year.values
    months  = times.dt.month.values

    oni_vals = np.zeros(len(feat_df), dtype=np.float32)
    for i, (y, m) in enumerate(zip(years, months)):
        oni_vals[i] = oni_map.get((int(y), int(m)), 0.0)

    feat_df["enso_oni"] = oni_vals
    covered = int(np.sum(oni_vals != 0.0))
    print(f"[ENSO] enso_oni: {covered:,}/{len(feat_df):,} rows có giá trị "
          f"({covered/len(feat_df)*100:.1f}%), ONI range [{oni_vals.min():.2f}, {oni_vals.max():.2f}]")
    return feat_df


# ---------------------------------------------------------------------------
# G3: Tạo sliding window sequences
# ---------------------------------------------------------------------------

_ERA5_LAT_MIN, _ERA5_LAT_MAX = -5.0, 35.0
_ERA5_LON_MIN, _ERA5_LON_MAX = 95.0, 145.0


def _all_in_era5(lats, lons, indices) -> bool:
    """Kiểm tra tất cả các điểm có nằm trong ERA5 coverage không."""
    for idx in indices:
        if not (_ERA5_LAT_MIN <= lats[idx] <= _ERA5_LAT_MAX and
                _ERA5_LON_MIN <= lons[idx] <= _ERA5_LON_MAX):
            return False
    return True


def make_sequences(feat_df: pd.DataFrame, config: dict):
    """
    Tạo sequences (X, y, meta) từ DataFrame features.

    Parameters
    ----------
    feat_df : pd.DataFrame  — output của build_features()
    config  : dict

    Returns
    -------
    X    : np.ndarray shape (N, lookback, n_features)
    y    : np.ndarray shape (N, n_steps*2)  — multi-horizon lat/lon
           Sprint 1: (N, 16) = [lat_6h, lon_6h, ..., lat_48h, lon_48h]
    meta : pd.DataFrame  — SID, SEASON, init_time, in_scs tương ứng với mỗi sequence
           in_scs: True nếu vị trí tâm bão tại init_time (last lookback step) nằm trong SCS box
    """
    feat_names   = config["features"]["names"]
    lookback     = config["model"]["lookback"]       # 8
    anchor_steps = config["model"]["anchor_steps"]   # [1,2,3,4,5,6,7,8]
    max_step     = max(anchor_steps)                 # 8

    X_list, y_list, meta_list = [], [], []
    skipped_era5 = 0

    for sid, grp in feat_df.groupby("SID"):
        grp = grp.sort_values("ISO_TIME").reset_index(drop=True)
        n = len(grp)

        # Cần ít nhất lookback + max_step bước
        min_len = lookback + max_step
        if n < min_len:
            continue

        # Chỉ dùng features có trong DataFrame (hỗ trợ backward compat)
        available = [f for f in feat_names if f in grp.columns]
        vals   = grp[available].values.astype(np.float32)
        lats   = grp["LAT"].values
        lons   = grp["LON"].values
        season = grp["SEASON"].iloc[0]
        times  = grp["ISO_TIME"].values
        in_scs_arr = (grp["in_scs"].values if "in_scs" in grp.columns
                      else np.zeros(n, dtype=bool))
        in_buffer_arr = (grp["in_buffer"].values if "in_buffer" in grp.columns
                         else np.zeros(n, dtype=bool))

        # Sliding window
        for i in range(n - min_len + 1):
            # Tất cả lookback + horizon points phải nằm trong ERA5 coverage
            all_indices = list(range(i, i + lookback + max_step))
            if not _all_in_era5(lats, lons, all_indices):
                skipped_era5 += 1
                continue

            x_seq = vals[i : i + lookback]   # (lookback, n_features)

            # Multi-horizon targets: [lat_step1, lon_step1, lat_step2, lon_step2, ...]
            y_row = []
            for step in anchor_steps:
                target_idx = i + lookback + step - 1
                y_row.append(lats[target_idx])
                y_row.append(lons[target_idx])

            X_list.append(x_seq)
            y_list.append(y_row)
            init_idx = i + lookback - 1
            meta_list.append({
                "SID":       sid,
                "SEASON":    season,
                "init_time": times[init_idx],
                "in_scs":    bool(in_scs_arr[init_idx]),
                "in_buffer": bool(in_buffer_arr[init_idx]),
            })

    if skipped_era5 > 0:
        print(f"[G3] Bỏ qua {skipped_era5:,} sequences ngoài ERA5 bounds (lat [{_ERA5_LAT_MIN},{_ERA5_LAT_MAX}], lon [{_ERA5_LON_MIN},{_ERA5_LON_MAX}])")

    X    = np.array(X_list,  dtype=np.float32)
    y    = np.array(y_list,  dtype=np.float32)
    meta = pd.DataFrame(meta_list)

    print(f"[G3] Sequences: X{X.shape}, y{y.shape}")
    return X, y, meta


# ---------------------------------------------------------------------------
# G3: Chuẩn hóa & lưu
# ---------------------------------------------------------------------------

def split_and_scale(X, y, meta, config, tag: str = ""):
    """
    Chia train/val/test theo năm, fit StandardScaler trên train, transform tất cả.
    Lưu scaler.pkl (hoặc scaler_{tag}.pkl nếu tag được chỉ định).

    Parameters
    ----------
    tag : str  — hậu tố phân biệt phiên bản features, ví dụ "14feat"

    Returns
    -------
    dict với keys: X_train, X_val, X_test, y_train, y_val, y_test,
                   meta_train, meta_val, meta_test, scaler
    """
    split_cfg = config["split"]
    base_dir  = Path(__file__).parent.parent.parent  # model_ai/
    suffix    = f"_{tag}" if tag else ""
    scaler_dir = (base_dir / config["output"]["scaler_path"]).parent
    scaler_path = scaler_dir / f"scaler{suffix}.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    season = meta["SEASON"].values

    tr_mask  = (season >= split_cfg["train"][0]) & (season <= split_cfg["train"][1])
    val_mask = (season >= split_cfg["val"][0])   & (season <= split_cfg["val"][1])
    te_mask  = (season >= split_cfg["test"][0])  & (season <= split_cfg["test"][1])

    X_train, y_train = X[tr_mask],  y[tr_mask]
    X_val,   y_val   = X[val_mask], y[val_mask]
    X_test,  y_test  = X[te_mask],  y[te_mask]

    print(f"[G3] Split — train: {tr_mask.sum():,} | val: {val_mask.sum():,} | test: {te_mask.sum():,}")

    # StandardScaler: fit CHỈ trên train, reshape 3D → 2D → scale → reshape lại
    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    CLIP_SIGMA = 5.0  # clip outliers sau StandardScaler — phòng steering features có đuôi nặng
    X_train = np.clip(scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape), -CLIP_SIGMA, CLIP_SIGMA)
    X_val   = np.clip(scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape),   -CLIP_SIGMA, CLIP_SIGMA)
    X_test  = np.clip(scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape), -CLIP_SIGMA, CLIP_SIGMA)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[G3] Scaler lưu tại: {scaler_path}  (clip ±{CLIP_SIGMA}σ)")

    # Checkpoint G3
    min_seq = config["checkpoints"]["g3_min_sequences"]
    total   = tr_mask.sum() + val_mask.sum() + te_mask.sum()
    if total < min_seq:
        raise ValueError(
            f"[G3 FAIL] Chỉ có {total} sequences, cần >= {min_seq}."
        )
    print(f"[G3] PASS: {total:,} sequences >= {min_seq}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "meta_train": meta[tr_mask].reset_index(drop=True),
        "meta_val":   meta[val_mask].reset_index(drop=True),
        "meta_test":  meta[te_mask].reset_index(drop=True),
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from src.g1_data.data_loader import load_clean_data, load_and_filter_scs, save_clean_data
    from src.g2_g3_features.era5_extractor import extract_era5_features, extract_multi_level_features

    parser = argparse.ArgumentParser(description="G2/G3: Feature engineering + sequences")
    parser.add_argument(
        "--tag", default="14feat",
        help="Hậu tố phân biệt phiên bản output (default: 14feat). "
             "Tag chứa 'wp' sẽ load toàn bộ WP basin thay vì chỉ SCS.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bắt buộc re-extract ERA5 dù feature_matrix_{tag}.csv đã tồn tại.",
    )
    parser.add_argument(
        "--g3-only", action="store_true",
        help="Chỉ chạy G3 (sequences+scaler), bỏ qua G2 hoàn toàn. "
             "Yêu cầu feature_matrix_{tag}.csv đã tồn tại.",
    )
    parser.add_argument(
        "--lookback", type=int, default=None,
        help="Override lookback (config mặc định = 8). Dùng cho ablation: --lookback 4",
    )
    parser.add_argument(
        "--lite", action="store_true",
        help="LITE mode: chỉ dùng 15 features storm-state + beta drift "
             "(không cần ERA5). Cho real-time deployment.",
    )
    parser.add_argument(
        "--westward-only", action="store_true",
        help="Filter training+val sequences về westward only (test KHÔNG đổi). "
             "Verify hypothesis: bão bất thường có gây nhiễu model không?",
    )
    parser.add_argument(
        "--lean", action="store_true",
        help="LEAN mode: drop 7 features có importance < 1 km (24 features). "
             "Verify: model có giữ skill khi loại features ít đóng góp?",
    )
    args = parser.parse_args()
    tag  = args.tag

    cfg      = load_config()
    if args.lookback is not None:
        cfg["model"]["lookback"] = args.lookback
        print(f"[CFG] Override lookback = {args.lookback}")
    if args.lite:
        # LITE mode: chỉ giữ storm-state (13) + beta drift (2) = 15 features
        # Tất cả available real-time từ JTWC/JMA bulletin (không cần ERA5)
        cfg["features"]["names"] = [
            "lat_norm", "lon_norm", "dlat", "dlon",
            "speed_kmh", "direction", "vmax", "pmin",
            "sst_actual", "month_sin", "month_cos",
            "storm_age_h", "dist2land",
            "beta_lat_drift", "beta_lon_drift",
        ]
        cfg["features"]["n_features"] = 15
        print(f"[CFG] LITE mode: 15 features (storm-state + beta drift, no ERA5)")
    if args.lean:
        # LEAN mode: drop 7 features có permutation importance < 1 km
        # (đo trên champion scs_v12_lb6 với baseline MAE 24h = 101.3 km)
        # Dropped:
        #   - direction       (+0.46 km, redundant với dlat/dlon)
        #   - sst_actual      (+0.08 km, redundant với vmax)
        #   - month_sin       (+0.13 km)
        #   - storm_age_h     (+0.78 km)
        #   - dist2land       (+0.24 km)
        #   - wind_shear      (+0.55 km, surprisingly low)
        #   - beta_lon_drift  (+0.00 km, analytical zero info)
        # Còn lại: 24 features
        cfg["features"]["names"] = [
            "lat_norm", "lon_norm", "dlat", "dlon", "speed_kmh",
            "vmax", "pmin", "month_cos",
            "u500", "v500", "u700", "v700", "z500",
            "u200", "v200", "u850", "v850",
            "vort500", "vort850",
            "beta_lat_drift",
            "asteer_u", "asteer_v",
            "asteer_u850", "asteer_v850",
        ]
        cfg["features"]["n_features"] = 24
        print(f"[CFG] LEAN mode: 24 features (drop 7 importance < 1 km)")
    base_dir = Path(__file__).parent.parent.parent  # model_ai/
    suffix   = f"_{tag}" if tag else ""

    feat_stem = Path(cfg["data"]["feature_file"])
    feat_path = base_dir / feat_stem.parent / f"{feat_stem.stem}{suffix}{feat_stem.suffix}"

    # ── G2: Feature extraction ──────────────────────────────────────────────
    # Nếu feature_matrix_{tag}.csv đã tồn tại → skip hoàn toàn G2 (tiết kiệm ~30-40 phút)
    # Dùng --force để bắt buộc re-extract, --g3-only để chỉ tạo lại sequences/scaler
    if feat_path.exists() and not args.force:
        size_mb = feat_path.stat().st_size / 1024 / 1024
        print(f"[G2] SKIP — feature_matrix đã có: {feat_path.name} ({size_mb:.1f} MB)")
        print("[G2] Load từ cache... (dùng --force để re-extract ERA5)")
        feat_df = pd.read_csv(feat_path, low_memory=False, parse_dates=["ISO_TIME"])
        print(f"[G2] Loaded: {len(feat_df):,} rows, {feat_df['SID'].nunique()} storms")

        # Auto-add analytical features từ config nếu thiếu trong cache
        # (chỉ áp dụng cho features không cần ERA5: enso_oni, beta_drift)
        feat_names_chk = cfg["features"]["names"]
        if "enso_oni" in feat_names_chk and "enso_oni" not in feat_df.columns:
            print("[G2] Cache thiếu 'enso_oni' — auto-add...")
            feat_df = add_enso_feature(feat_df, base_dir)
            # Save back để lần sau không phải thêm lại
            feat_df.to_csv(feat_path, index=False)
            print(f"[G2] Đã cập nhật cache: {feat_path.name}")
    else:
        if args.g3_only:
            raise FileNotFoundError(
                f"--g3-only yêu cầu {feat_path} phải tồn tại. Chạy không có --g3-only trước."
            )

        # G1: Load data
        # scs_v10+ dùng buffer training → cần toàn bộ WP basin
        use_full_wp = ("wp" in tag.lower()
                       or tag.startswith("scs_v10")
                       or tag.startswith("scs_buf"))
        if use_full_wp:
            print(f"[G2] Tag '{tag}' — load toàn bộ WP basin từ raw CSV")
            df_clean = load_and_filter_scs(config=cfg)
            wp_clean_path = base_dir / f"data/processed/ibtracs{suffix}.csv"
            save_clean_data(df_clean, output_path=str(wp_clean_path))
        else:
            df_clean = load_clean_data(cfg)

        feat_df    = build_features(df_clean, cfg)
        feat_names = cfg["features"]["names"]

        # G2+: wind_shear + sst_actual
        if "wind_shear" in feat_names or "sst_actual" in feat_names:
            feat_df = extract_era5_features(feat_df, cfg)
        else:
            print("[G2] Bỏ qua ERA5/SST — không có trong feature list")

        # G2+: steering + vorticity features
        steering_cols = ["u500", "v500", "u700", "v700", "z500", "u200", "v200", "u850", "v850",
                         "vort500", "vort850"]
        if any(f in feat_names for f in steering_cols):
            feat_df = extract_multi_level_features(feat_df, cfg)
        else:
            print("[G2] Bỏ qua steering features — không có trong feature list")

        # G2+: ENSO ONI
        if "enso_oni" in feat_names:
            feat_df = add_enso_feature(feat_df, base_dir)
        else:
            print("[G2] Bỏ qua enso_oni — không có trong feature list")

        # Lưu feature_matrix_{tag}.csv
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        feat_df.to_csv(feat_path, index=False)
        size_mb = feat_path.stat().st_size / 1024 / 1024
        print(f"[G2] Đã lưu: {feat_path} ({size_mb:.1f} MB)")

        # Xóa ERA5 cache tạm
        for cache_name in ["_wind_shear_cache.npy", "_era5ml_cache.npz"]:
            cache_p = base_dir / "data/features" / cache_name
            if cache_p.exists():
                cache_p.unlink()
                print(f"[ERA5] Đã xóa cache tạm {cache_name}")

    # ── G3: Sequences + Scaler ──────────────────────────────────────────────
    X, y, meta = make_sequences(feat_df, cfg)

    # scs_v10+: filter to BUFFER box (5-25°N, 99-123°E) cho training
    # — sẽ tăng số lượng sequences gần biên SCS, fix worst cases ở biên đông
    if (tag.startswith("scs_v10") or tag.startswith("scs_buf")) and "in_buffer" in meta.columns:
        n_before = len(X)
        buffer_mask = meta["in_buffer"].astype(bool).values
        X    = X[buffer_mask]
        y    = y[buffer_mask]
        meta = meta[buffer_mask].reset_index(drop=True)
        n_scs = int(meta["in_scs"].astype(bool).sum()) if "in_scs" in meta.columns else 0
        print(f"[G3] Buffer filter ({tag}): {len(X):,}/{n_before:,} sequences "
              f"(SCS-only: {n_scs:,})")

    # --westward-only: filter training+val sequences về westward only
    # Keep test FULL (apples-to-apples comparison with champion)
    # Westward = trajectory với mean(dlon) < -0.05 AND |dlon| > |dlat| ở 4 bước cuối
    if args.westward_only:
        n_before = len(X)
        # Classify trajectory (matching evaluate.py classify_trajectories logic)
        # dlat/dlon là features index 2, 3 trong X (raw, chưa scale)
        n_avg = min(4, X.shape[1])  # last 4 steps hoặc lookback nếu ngắn hơn
        dlat_avg = X[:, -n_avg:, 2].mean(axis=1)
        dlon_avg = X[:, -n_avg:, 3].mean(axis=1)
        westward_mask = (dlon_avg < -0.05) & (np.abs(dlon_avg) > np.abs(dlat_avg))

        # Identify test sequences theo SEASON (test = 2021-2024)
        split_cfg = cfg["split"]
        season = meta["SEASON"].values
        test_mask = (season >= split_cfg["test"][0]) & (season <= split_cfg["test"][1])

        # Keep: TẤT CẢ test + chỉ WESTWARD train/val
        keep_mask = westward_mask | test_mask
        X = X[keep_mask]; y = y[keep_mask]; meta = meta[keep_mask].reset_index(drop=True)

        n_test_kept = int(test_mask[keep_mask if False else slice(None)].sum())  # informational
        n_train_val_west = int((westward_mask & ~test_mask).sum())
        print(f"[G3] Westward filter: kept {len(X):,}/{n_before:,} sequences "
              f"(train+val filtered: {n_train_val_west:,} westward, "
              f"test KEPT FULL)")

    data = split_and_scale(X, y, meta, cfg, tag=tag)

    seq_path = base_dir / f"data/features/sequences{suffix}.npz"
    seq_path.parent.mkdir(parents=True, exist_ok=True)

    # in_scs flags để G7 filter test set theo Biển Đông
    in_scs_train = data["meta_train"].get("in_scs", pd.Series([False] * len(data["X_train"]))).values.astype(bool)
    in_scs_val   = data["meta_val"].get("in_scs",   pd.Series([False] * len(data["X_val"]))).values.astype(bool)
    in_scs_test  = data["meta_test"].get("in_scs",  pd.Series([False] * len(data["X_test"]))).values.astype(bool)
    # in_buffer flags để verify scope training
    in_buf_train = data["meta_train"].get("in_buffer", pd.Series([False] * len(data["X_train"]))).values.astype(bool)
    in_buf_val   = data["meta_val"].get("in_buffer",   pd.Series([False] * len(data["X_val"]))).values.astype(bool)
    in_buf_test  = data["meta_test"].get("in_buffer",  pd.Series([False] * len(data["X_test"]))).values.astype(bool)

    # SID + init_time để G7 phân tích per-storm
    # IMPORTANT: ép kiểu string cố định (<U...) thay vì object để np.load không cần allow_pickle
    def _astr(s, w=32):
        return np.asarray(s.astype(str).tolist(), dtype=f"<U{w}")
    sid_train       = _astr(data["meta_train"]["SID"], 16)
    sid_val         = _astr(data["meta_val"]["SID"],   16)
    sid_test        = _astr(data["meta_test"]["SID"],  16)
    init_time_test  = _astr(data["meta_test"]["init_time"], 32)
    season_test     = data["meta_test"]["SEASON"].astype(int).values

    np.savez(
        seq_path,
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"],     y_val=data["y_val"],
        X_test=data["X_test"],   y_test=data["y_test"],
        in_scs_train=in_scs_train,
        in_scs_val=in_scs_val,
        in_scs_test=in_scs_test,
        in_buffer_train=in_buf_train,
        in_buffer_val=in_buf_val,
        in_buffer_test=in_buf_test,
        sid_train=sid_train,
        sid_val=sid_val,
        sid_test=sid_test,
        init_time_test=init_time_test,
        season_test=season_test,
    )
    n_scs_test = int(in_scs_test.sum())
    print(f"[G3] Sequences lưu tại: {seq_path}")
    print(f"[G3] Test SCS-only: {n_scs_test:,}/{len(in_scs_test):,} ({n_scs_test/max(len(in_scs_test),1)*100:.1f}%)")

    print(f"\n=== G2/G3 HOÀN THÀNH (tag='{tag}') ===")
    print(f"    X_train: {data['X_train'].shape}")
    print(f"    X_val:   {data['X_val'].shape}")
    print(f"    X_test:  {data['X_test'].shape}")
