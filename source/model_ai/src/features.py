"""
features.py
-----------
G2: Tính feature engineering từ bao_bien_dong_clean.csv
G3: Tạo sliding window sequences (X, y) và chuẩn hóa dữ liệu

Features (12):
  lat_norm, lon_norm      — vị trí normalize về [0,1] trong SCS box
  dlat, dlon              — thay đổi vị trí so với bước trước (degrees/6h)
  speed_kmh               — tốc độ di chuyển (Haversine / 6h)
  direction               — hướng di chuyển (0–360°, 0=Bắc)
  vmax                    — gió cực đại (kt)
  pmin                    — áp suất cực tiểu (hPa)
  sst_c                   — nhiệt độ bề mặt biển (climatology)
  month_sin, month_cos    — mã hóa tuần hoàn tháng
  storm_age_h             — tuổi cơn bão (giờ kể từ điểm đầu tiên)

Targets (4):
  lat_24h, lon_24h        — vị trí sau 24h (bước +4)
  lat_48h, lon_48h        — vị trí sau 48h (bước +8)
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
        config_path = Path(__file__).parent.parent / "config.yaml"
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

        for i in range(n):
            row = grp.iloc[i]

            lat = row["LAT"]
            lon = row["LON"]
            t   = row["ISO_TIME"]
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
                prev = grp.iloc[i - 1]
                dlat = lat - prev["LAT"]
                dlon = lon - prev["LON"]
                try:
                    dist_km   = haversine_km(prev["LAT"], prev["LON"], lat, lon)
                    speed_kmh = dist_km / dt_h
                    direction = bearing_deg(prev["LAT"], prev["LON"], lat, lon)
                except Exception:
                    speed_kmh = 0.0
                    direction = 0.0

            # --- Gió & Áp suất ---
            vmax = row.get("vmax", np.nan)
            pmin = row.get("pmin", np.nan)

            # --- SST climatology ---
            sst_c = sst_climatology(month)

            # --- Mã hóa tuần hoàn tháng ---
            month_sin = math.sin(2 * math.pi * month / 12)
            month_cos = math.cos(2 * math.pi * month / 12)

            # --- Tuổi cơn bão (giờ) ---
            first_time = grp.iloc[0]["ISO_TIME"]
            if pd.notna(t) and pd.notna(first_time):
                storm_age_h = (t - first_time).total_seconds() / 3600
            else:
                storm_age_h = i * dt_h

            records.append({
                "SID":       sid,
                "SEASON":    row.get("SEASON", np.nan),
                "ISO_TIME":  t,
                "LAT":       lat,
                "LON":       lon,
                "in_scs":    row.get("in_scs", False),
                # 12 features
                "lat_norm":    lat_norm,
                "lon_norm":    lon_norm,
                "dlat":        dlat,
                "dlon":        dlon,
                "speed_kmh":   speed_kmh,
                "direction":   direction,
                "vmax":        vmax,
                "pmin":        pmin,
                "sst_c":       sst_c,
                "month_sin":   month_sin,
                "month_cos":   month_cos,
                "storm_age_h": storm_age_h,
            })

    feat_df = pd.DataFrame(records)

    # --- Xử lý NaN vmax/pmin: interpolate trong từng storm ---
    for col in ["vmax", "pmin"]:
        feat_df[col] = feat_df.groupby("SID")[col].transform(
            lambda s: s.interpolate(method="linear").ffill().bfill()
        )

    # --- Nếu vẫn còn NaN → fill median toàn bộ ---
    for col in feat_cfg["names"]:
        if feat_df[col].isna().any():
            median_val = feat_df[col].median()
            feat_df[col] = feat_df[col].fillna(median_val)
            print(f"     [warn] {col}: còn NaN → fill median ({median_val:.2f})")

    print(f"[G2] Tổng rows feature: {len(feat_df):,} | Storms: {feat_df['SID'].nunique()}")
    print(f"[G2] NaN còn lại: {feat_df[feat_cfg['names']].isna().sum().sum()}")
    return feat_df


# ---------------------------------------------------------------------------
# G3: Tạo sliding window sequences
# ---------------------------------------------------------------------------

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
    y    : np.ndarray shape (N, 4)  — [lat_24h, lon_24h, lat_48h, lon_48h]
    meta : pd.DataFrame  — SID, SEASON, init_time tương ứng với mỗi sequence
    """
    feat_names  = config["features"]["names"]
    lookback    = config["model"]["lookback"]   # 8
    anchor      = config["model"]["anchor_steps"]  # [4, 8]
    step_24h    = anchor[0]  # 4
    step_48h    = anchor[1]  # 8

    X_list, y_list, meta_list = [], [], []

    for sid, grp in feat_df.groupby("SID"):
        grp = grp.sort_values("ISO_TIME").reset_index(drop=True)
        n = len(grp)

        # Cần ít nhất lookback + step_48h bước
        min_len = lookback + step_48h
        if n < min_len:
            continue

        vals   = grp[feat_names].values.astype(np.float32)  # (n, 12)
        lats   = grp["LAT"].values
        lons   = grp["LON"].values
        season = grp["SEASON"].iloc[0]
        times  = grp["ISO_TIME"].values

        # Sliding window
        for i in range(n - min_len + 1):
            x_seq = vals[i : i + lookback]                  # (8, 12)
            lat_24h = lats[i + lookback + step_24h - 1]
            lon_24h = lons[i + lookback + step_24h - 1]
            lat_48h = lats[i + lookback + step_48h - 1]
            lon_48h = lons[i + lookback + step_48h - 1]

            X_list.append(x_seq)
            y_list.append([lat_24h, lon_24h, lat_48h, lon_48h])
            meta_list.append({
                "SID":       sid,
                "SEASON":    season,
                "init_time": times[i + lookback - 1],
            })

    X    = np.array(X_list,  dtype=np.float32)   # (N, 8, 12)
    y    = np.array(y_list,  dtype=np.float32)   # (N, 4)
    meta = pd.DataFrame(meta_list)

    print(f"[G3] Sequences: X{X.shape}, y{y.shape}")
    return X, y, meta


# ---------------------------------------------------------------------------
# G3: Chuẩn hóa & lưu
# ---------------------------------------------------------------------------

def split_and_scale(X, y, meta, config):
    """
    Chia train/val/test theo năm, fit StandardScaler trên train, transform tất cả.
    Lưu scaler.pkl.

    Returns
    -------
    dict với keys: X_train, X_val, X_test, y_train, y_val, y_test,
                   meta_train, meta_val, meta_test, scaler
    """
    split_cfg = config["split"]
    base_dir  = Path(__file__).parent.parent
    scaler_path = base_dir / config["output"]["scaler_path"]
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

    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[G3] Scaler lưu tại: {scaler_path}")

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
    from data_loader import load_clean_data

    cfg = load_config()
    base_dir = Path(__file__).parent.parent

    # G2: Tính features
    df_clean = load_clean_data(cfg)
    feat_df  = build_features(df_clean, cfg)

    # Lưu feature_matrix.csv
    feat_path = base_dir / cfg["data"]["feature_file"]
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(feat_path, index=False)
    size_mb = feat_path.stat().st_size / 1024 / 1024
    print(f"[G2] Đã lưu: {feat_path} ({size_mb:.1f} MB)")

    # G3: Tạo sequences
    X, y, meta = make_sequences(feat_df, cfg)

    # G3: Split + Scale + lưu scaler
    data = split_and_scale(X, y, meta, cfg)

    # Lưu sequences (numpy .npz)
    seq_path = base_dir / "data/features/sequences.npz"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        seq_path,
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"],     y_val=data["y_val"],
        X_test=data["X_test"],   y_test=data["y_test"],
    )
    print(f"[G3] Sequences lưu tại: {seq_path}")
    print("\n=== G2/G3 HOÀN THÀNH ===")
    print(f"    X_train: {data['X_train'].shape}")
    print(f"    X_val:   {data['X_val'].shape}")
    print(f"    X_test:  {data['X_test'].shape}")
