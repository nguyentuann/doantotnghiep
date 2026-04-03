"""
preprocessor.py — Tính 12 features từ raw track points, scale bằng scaler.pkl
------------------------------------------------------------------------------
Input : list[TrackPoint] (tối thiểu 8 điểm)
Output: np.ndarray shape (1, 8, 12) — sẵn sàng đưa vào ONNX model
"""

import math
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

_SCALER_PATH = Path(__file__).parent.parent.parent / "model_ai/models/scaler.pkl"

# SCS box để normalize lat/lon (khớp với config.yaml)
_LAT_MIN, _LAT_MAX = 8.0, 22.0
_LON_MIN, _LON_MAX = 102.0, 120.0

# SST climatology fallback
_SST_CLIM = {
    1: 26.2, 2: 26.0, 3: 26.8, 4: 28.0, 5: 29.2, 6: 30.1,
    7: 30.3, 8: 30.2, 9: 29.5, 10: 28.4, 11: 27.5, 12: 26.8,
}

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
    Tính 12 features cho danh sách điểm track.

    Parameters
    ----------
    points : list of dict với keys: lat, lon, vmax (optional), pmin (optional),
             iso_time (optional, dùng để tính month/storm_age)

    Returns
    -------
    np.ndarray shape (N, 12)
    """
    n = len(points)
    rows = []

    for i, pt in enumerate(points):
        lat = pt["lat"]
        lon = pt["lon"]
        vmax = pt.get("vmax") or 35.0
        pmin = pt.get("pmin") or 1000.0

        # Thời gian
        iso_time = pt.get("iso_time")
        if iso_time:
            import pandas as pd
            t = pd.Timestamp(iso_time)
            month = t.month
            storm_age_h = i * 6.0
        else:
            month = 9  # peak typhoon season
            storm_age_h = i * 6.0

        # Position features
        lat_norm = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
        lon_norm = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)

        # Displacement
        if i > 0:
            prev = points[i - 1]
            dlat = lat - prev["lat"]
            dlon = lon - prev["lon"]
            speed_kmh = _haversine_km(prev["lat"], prev["lon"], lat, lon) / 6.0
            direction = _bearing(prev["lat"], prev["lon"], lat, lon)
        else:
            dlat, dlon, speed_kmh, direction = 0.0, 0.0, 0.0, 0.0

        sst_c = _SST_CLIM.get(month, 28.0)
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        rows.append([
            lat_norm, lon_norm,
            dlat, dlon,
            speed_kmh, direction,
            vmax, pmin,
            sst_c,
            month_sin, month_cos,
            storm_age_h,
        ])

    return np.array(rows, dtype=np.float32)


def prepare_input(points: list[dict], lookback: int = 8) -> np.ndarray:
    """
    Lấy lookback điểm cuối, tính features, scale, trả về (1, lookback, 12).
    """
    scaler = _load_scaler()

    # Lấy lookback điểm cuối
    window = points[-lookback:]
    feat_matrix = build_feature_matrix(window)   # (lookback, 12)

    # Scale (scaler fit trên 12 features)
    feat_scaled = scaler.transform(feat_matrix)   # (lookback, 12)

    return feat_scaled[np.newaxis].astype(np.float32)  # (1, lookback, 12)


def unscale_output(pred: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse-transform 4 target values [lat_24h, lon_24h, lat_48h, lon_48h].
    Scaler được fit trên 12 features; lat và lon là features 0 và 1.
    """
    # lat_norm → lat, lon_norm → lon
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
