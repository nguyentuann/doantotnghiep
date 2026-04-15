"""
preprocessor.py — Tính 14 features từ raw track points, scale bằng scaler_14feat.pkl
--------------------------------------------------------------------------------------
Input : list[TrackPoint] (tối thiểu 8 điểm)
Output: np.ndarray shape (1, 8, 14) — sẵn sàng đưa vào ONNX model

14 features (theo thứ tự khớp với sequences_14feat.npz):
  0  lat_norm      — latitude normalized to SCS box
  1  lon_norm      — longitude normalized to SCS box
  2  dlat          — lat displacement from previous step
  3  dlon          — lon displacement from previous step
  4  speed_kmh     — translation speed (km/h)
  5  direction     — bearing 0–360°
  6  vmax          — max wind (kt)
  7  pmin          — min pressure (hPa)
  8  sst_actual    — SST (°C) from NOAA OISST; fallback to climatology
  9  month_sin     — seasonal encoding sin
  10 month_cos     — seasonal encoding cos
  11 storm_age_h   — hours since storm first point in window
  12 dist2land     — distance to nearest land (km); default 252 km if unknown
  13 wind_shear    — ERA5 |V200 - V850| (m/s); fallback to monthly climatology
"""

import math
import pickle
import numpy as np
from pathlib import Path

_SCALER_PATH = Path(__file__).parent.parent.parent / "model_ai/models/scaler_14feat.pkl"

# SCS box để normalize lat/lon (khớp với config.yaml)
_LAT_MIN, _LAT_MAX = 8.0, 22.0
_LON_MIN, _LON_MAX = 102.0, 120.0

# SST climatology fallback (tháng 1–12)
_SST_CLIM = {
    1: 26.2, 2: 26.0, 3: 26.8, 4: 28.0, 5: 29.2, 6: 30.1,
    7: 30.3, 8: 30.2, 9: 29.5, 10: 28.4, 11: 27.5, 12: 26.8,
}

# Wind shear climatology (ERA5 median per month, computed from feature_matrix_14feat.csv)
_WIND_SHEAR_CLIM = {
    1: 9.91, 2: 9.07, 3: 9.42,  4: 9.82,  5: 9.07,  6: 10.13,
    7: 10.22, 8: 9.62, 9: 9.22, 10: 9.95, 11: 10.34, 12: 11.18,
}

# dist2land default khi không có giá trị (median trên toàn dataset)
_DIST2LAND_DEFAULT = 252.0

_scaler = None


def _load_scaler():
    global _scaler
    if _scaler is not None:
        return _scaler
    if not _SCALER_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy scaler: {_SCALER_PATH}")
    with open(_SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)
    return _scaler


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def build_feature_matrix(points: list[dict]) -> np.ndarray:
    """
    Tính 14 features cho danh sách điểm track.

    Parameters
    ----------
    points : list of dict với keys:
        lat, lon          — bắt buộc
        vmax              — optional, default 35 kt
        pmin              — optional, default 1000 hPa
        iso_time          — optional, dùng để tính month/storm_age
        dist2land         — optional (km), default _DIST2LAND_DEFAULT
        wind_shear        — optional (m/s), default monthly climatology

    Returns
    -------
    np.ndarray shape (N, 14)
    """
    rows = []

    for i, pt in enumerate(points):
        lat  = pt["lat"]
        lon  = pt["lon"]
        vmax = pt.get("vmax") or 35.0
        pmin = pt.get("pmin") or 1000.0

        # Thời gian
        iso_time = pt.get("iso_time")
        if iso_time:
            import pandas as pd
            t = pd.Timestamp(iso_time)
            month = t.month
        else:
            month = 9  # peak typhoon season
        storm_age_h = i * 6.0

        # Position features
        lat_norm = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
        lon_norm = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)

        # Displacement
        if i > 0:
            prev = points[i - 1]
            dlat      = lat - prev["lat"]
            dlon      = lon - prev["lon"]
            speed_kmh = _haversine_km(prev["lat"], prev["lon"], lat, lon) / 6.0
            direction = _bearing(prev["lat"], prev["lon"], lat, lon)
        else:
            dlat, dlon, speed_kmh, direction = 0.0, 0.0, 0.0, 0.0

        sst_actual  = pt.get("sst_actual") or _SST_CLIM.get(month, 28.0)
        month_sin   = math.sin(2 * math.pi * month / 12)
        month_cos   = math.cos(2 * math.pi * month / 12)
        dist2land   = pt.get("dist2land") or _DIST2LAND_DEFAULT
        wind_shear  = pt.get("wind_shear") or _WIND_SHEAR_CLIM.get(month, 9.8)

        rows.append([
            lat_norm, lon_norm,
            dlat, dlon,
            speed_kmh, direction,
            vmax, pmin,
            sst_actual,
            month_sin, month_cos,
            storm_age_h,
            dist2land,
            wind_shear,
        ])

    return np.array(rows, dtype=np.float32)


def prepare_input(points: list[dict], lookback: int = 8) -> np.ndarray:
    """
    Lấy lookback điểm cuối, tính features, scale, trả về (1, lookback, 14).
    """
    scaler = _load_scaler()

    # Lấy lookback điểm cuối
    window = points[-lookback:]
    feat_matrix = build_feature_matrix(window)   # (lookback, 14)

    # Scale (scaler fit trên 14 features)
    feat_scaled = scaler.transform(feat_matrix)   # (lookback, 14)

    return feat_scaled[np.newaxis].astype(np.float32)  # (1, lookback, 14)


def unscale_output(pred: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse-transform 4 target values [lat_24h, lon_24h, lat_48h, lon_48h].
    Scaler được fit trên 14 features; lat_norm và lon_norm là features 0 và 1.
    """
    lat_mean = scaler.mean_[0]
    lat_std  = scaler.scale_[0]
    lon_mean = scaler.mean_[1]
    lon_std  = scaler.scale_[1]

    lat_24h = pred[0, 0] * lat_std + lat_mean
    lon_24h = pred[0, 1] * lon_std + lon_mean
    lat_48h = pred[0, 2] * lat_std + lat_mean
    lon_48h = pred[0, 3] * lon_std + lon_mean

    # Denormalize từ [0,1] về lat/lon thực
    lat_24h = lat_24h * (_LAT_MAX - _LAT_MIN) + _LAT_MIN
    lon_24h = lon_24h * (_LON_MAX - _LON_MIN) + _LON_MIN
    lat_48h = lat_48h * (_LAT_MAX - _LAT_MIN) + _LAT_MIN
    lon_48h = lon_48h * (_LON_MAX - _LON_MIN) + _LON_MIN

    return np.array([lat_24h, lon_24h, lat_48h, lon_48h])
