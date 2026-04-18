"""
preprocessor.py — Tính features từ raw track points, scale bằng scaler.pkl
---------------------------------------------------------------------------
Tự động detect 12 hoặc 14 features dựa trên scaler được load.

12 features (scaler_wp_full.pkl):
  lat_norm, lon_norm, dlat, dlon, speed_kmh, direction,
  vmax, pmin, month_sin, month_cos, storm_age_h, dist2land

14 features (scaler_wp_steering.pkl) — thêm:
  12  steering_u  — mean U wind (850+200 hPa) ±5° box (m/s)
  13  steering_v  — mean V wind (850+200 hPa) ±5° box (m/s)
"""

import math
import pickle
import numpy as np
from pathlib import Path

# Thử load scaler wp_steering trước (14 feat), fallback về wp_full (12 feat)
_BASE = Path(__file__).parent.parent.parent / "model_ai/models"
_SCALER_STEERING = _BASE / "scaler_wp_steering.pkl"
_SCALER_WP_FULL  = _BASE / "scaler_wp_full.pkl"

_SCALER_PATH = _SCALER_STEERING if _SCALER_STEERING.exists() else _SCALER_WP_FULL

# SCS box để normalize lat/lon
_LAT_MIN, _LAT_MAX = 8.0, 22.0
_LON_MIN, _LON_MAX = 102.0, 120.0
_DIST2LAND_DEFAULT = 252.0

_scaler    = None
_n_features = None


def _load_scaler():
    global _scaler, _n_features, _SCALER_PATH
    if _scaler is not None:
        return _scaler
    if not _SCALER_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy scaler: {_SCALER_PATH}")
    with open(_SCALER_PATH, "rb") as f:
        _scaler = pickle.load(f)
    _n_features = _scaler.n_features_in_
    print(f"[preprocessor] Scaler loaded: {_SCALER_PATH.name} ({_n_features} features)")
    return _scaler


def get_n_features() -> int:
    _load_scaler()
    return _n_features


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


# ─── ERA5 Steering Flow Extractor (lazy, cache 1 year) ────────────────────────

class _ERA5SteeringExtractor:
    """Trích xuất steering_u, steering_v từ ERA5 tại điểm (lat, lon, time)."""

    _ERA5_DIR = Path(__file__).parent.parent.parent / "model_ai/data/era5/pressure"
    _RADIUS   = 5.0  # degrees box bán kính

    def __init__(self):
        self._year  = None
        self._ds    = None
        self._u_full = None   # (n_vt, n_pl, nlat, nlon)
        self._v_full = None
        self._times  = None
        self._lats   = None
        self._lons   = None
        self._idx200 = None
        self._idx850 = None

    def _load_year(self, year: int):
        if self._year == year:
            return self._ds is not None
        # Đóng file cũ
        if self._ds is not None:
            try:
                self._ds.close()
            except Exception:
                pass
            self._ds = self._u_full = self._v_full = None

        path = self._ERA5_DIR / f"era5_pressure_{year}.nc"
        if not path.exists():
            self._year = year
            return False

        try:
            import xarray as xr
            import pandas as pd
            ds = xr.open_dataset(str(path))

            tc = "valid_time" if "valid_time" in ds.coords else "time"
            pl = ds["pressure_level"].values

            if "time" in ds.dims:
                u = ds["u"].mean(dim="time", skipna=True).values.astype(np.float32)
                v = ds["v"].mean(dim="time", skipna=True).values.astype(np.float32)
            else:
                u = ds["u"].values.astype(np.float32)
                v = ds["v"].values.astype(np.float32)

            self._ds     = ds
            self._times  = pd.DatetimeIndex(ds[tc].values)
            self._lats   = ds["latitude"].values
            self._lons   = ds["longitude"].values
            self._idx200 = int(np.argmin(np.abs(pl - 200.0)))
            self._idx850 = int(np.argmin(np.abs(pl - 850.0)))
            self._u_full = u
            self._v_full = v
            self._year   = year
            return True
        except Exception as e:
            print(f"[ERA5Steering] Lỗi load {path.name}: {e}")
            self._year = year
            return False

    def get(self, lat: float, lon: float, iso_time) -> tuple[float, float]:
        """Trả về (steering_u, steering_v) tại (lat, lon, time). Fallback (0, 0)."""
        try:
            import pandas as pd
            ts = pd.Timestamp(iso_time)
            if not self._load_year(ts.year):
                return 0.0, 0.0

            ti = self._times.get_indexer([np.datetime64(ts)], method="nearest")[0]
            if ti < 0:
                return 0.0, 0.0

            u200 = self._u_full[ti, self._idx200]
            u850 = self._u_full[ti, self._idx850]
            v200 = self._v_full[ti, self._idx200]
            v850 = self._v_full[ti, self._idx850]

            lats, lons = self._lats, self._lons
            lat_mask = (lats >= lat - self._RADIUS) & (lats <= lat + self._RADIUS)
            lon_mask = (lons >= lon - self._RADIUS) & (lons <= lon + self._RADIUS)

            if lat_mask.sum() == 0 or lon_mask.sum() == 0:
                li = int(np.argmin(np.abs(lats - lat)))
                lo = int(np.argmin(np.abs(lons - lon)))
                su = float((u200[li, lo] + u850[li, lo]) / 2)
                sv = float((v200[li, lo] + v850[li, lo]) / 2)
            else:
                su_field = (u200 + u850) / 2
                sv_field = (v200 + v850) / 2
                su = float(np.nanmean(su_field[np.ix_(lat_mask, lon_mask)]))
                sv = float(np.nanmean(sv_field[np.ix_(lat_mask, lon_mask)]))

            return su, sv
        except Exception:
            return 0.0, 0.0

    def close(self):
        if self._ds is not None:
            try:
                self._ds.close()
            except Exception:
                pass


_steering_extractor = _ERA5SteeringExtractor()


# ─── Feature matrix ───────────────────────────────────────────────────────────

def build_feature_matrix(points: list[dict],
                         steering_flows: list[tuple] | None = None) -> np.ndarray:
    """
    Tính features cho danh sách điểm track.

    Parameters
    ----------
    points        : list of dict (lat, lon, vmax, pmin, iso_time, dist2land)
    steering_flows: list of (steering_u, steering_v) per point, hoặc None

    Returns
    -------
    np.ndarray shape (N, 12) hoặc (N, 14) tuỳ theo steering_flows
    """
    rows = []
    use_steering = steering_flows is not None

    for i, pt in enumerate(points):
        lat  = pt["lat"]
        lon  = pt["lon"]
        vmax = pt.get("vmax") or 35.0
        pmin = pt.get("pmin") or 1000.0

        iso_time = pt.get("iso_time")
        if iso_time:
            import pandas as pd
            t = pd.Timestamp(iso_time)
            month = t.month
        else:
            month = 9
        storm_age_h = i * 6.0

        lat_norm = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
        lon_norm = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)

        if i > 0:
            prev = points[i - 1]
            dlat      = lat - prev["lat"]
            dlon      = lon - prev["lon"]
            speed_kmh = _haversine_km(prev["lat"], prev["lon"], lat, lon) / 6.0
            direction = _bearing(prev["lat"], prev["lon"], lat, lon)
        else:
            dlat, dlon, speed_kmh, direction = 0.0, 0.0, 0.0, 0.0

        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)
        dist2land = pt.get("dist2land") or _DIST2LAND_DEFAULT

        row = [
            lat_norm, lon_norm,
            dlat, dlon,
            speed_kmh, direction,
            vmax, pmin,
            month_sin, month_cos,
            storm_age_h, dist2land,
        ]

        if use_steering:
            su, sv = steering_flows[i]
            row += [float(su), float(sv)]

        rows.append(row)

    return np.array(rows, dtype=np.float32)


def prepare_input(points: list[dict], lookback: int = 8) -> np.ndarray:
    """
    Lấy lookback điểm cuối, tính features, scale.

    Tự động detect 12 hoặc 14 features từ scaler.
    Nếu 14 features, trích xuất steering_u/v từ ERA5.

    Returns
    -------
    np.ndarray shape (1, lookback, n_features)
    """
    scaler    = _load_scaler()
    n_feat    = _n_features
    window    = points[-lookback:]

    steering_flows = None
    if n_feat == 14:
        # Trích xuất steering flow từ ERA5 cho từng điểm trong window
        # Điểm không có iso_time hoặc ngoài ERA5 range → fallback (0, 0)
        steering_flows = []
        last_valid_su, last_valid_sv = 0.0, 0.0
        for pt in window:
            iso_time = pt.get("iso_time")
            if iso_time:
                su, sv = _steering_extractor.get(pt["lat"], pt["lon"], iso_time)
                if su != 0.0 or sv != 0.0:
                    last_valid_su, last_valid_sv = su, sv
                else:
                    # Dùng giá trị hợp lệ cuối cùng (persistence)
                    su, sv = last_valid_su, last_valid_sv
            else:
                su, sv = last_valid_su, last_valid_sv
            steering_flows.append((su, sv))

    feat_matrix = build_feature_matrix(window, steering_flows)   # (lookback, n_feat)
    feat_scaled = scaler.transform(feat_matrix)                   # (lookback, n_feat)
    return feat_scaled[np.newaxis].astype(np.float32)             # (1, lookback, n_feat)


def compute_cliper(x_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    Tính CLIPER prediction từ input đã scaled.

    Parameters
    ----------
    x_scaled : (1, lookback, n_feat)
    scaler   : fitted StandardScaler

    Returns
    -------
    np.ndarray shape (1, 4) — [lat_24h, lon_24h, lat_48h, lon_48h] in degrees
    """
    last_step_scaled = x_scaled[0, -1, :]
    last_step_orig   = scaler.inverse_transform(last_step_scaled.reshape(1, -1))[0]

    lat_current = last_step_orig[0] * (_LAT_MAX - _LAT_MIN) + _LAT_MIN
    lon_current = last_step_orig[1] * (_LON_MAX - _LON_MIN) + _LON_MIN
    dlat = last_step_orig[2]
    dlon = last_step_orig[3]

    return np.array([[
        lat_current + 4 * dlat,
        lon_current + 4 * dlon,
        lat_current + 8 * dlat,
        lon_current + 8 * dlon,
    ]], dtype=np.float32)


def decode_output(pred: np.ndarray, x_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    Model output (delta) + CLIPER → lat/lon thực (degrees).

    Returns
    -------
    np.ndarray shape (4,) — [lat_24h, lon_24h, lat_48h, lon_48h]
    """
    cliper = compute_cliper(x_scaled, scaler)
    return (cliper + pred)[0]
