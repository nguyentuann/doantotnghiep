"""
preprocessor.py — Tính features từ raw track points, scale với scaler.pkl

Feature order scs_v11_lb6 (29) — BEST hiện tại:
  base(13) + 12 ERA5 (wind_shear..vort850) + 2 beta drift + 2 asteer (annulus)
  Cụ thể:
    0-12 : lat_norm, lon_norm, dlat, dlon, speed_kmh, direction, vmax, pmin,
           sst_actual, month_sin, month_cos, storm_age_h, dist2land
   13-18 : wind_shear, u500, v500, u700, v700, z500
   19-22 : u200, v200, u850, v850
   23-24 : vort500, vort850
   25-26 : beta_lat_drift, beta_lon_drift (analytical — Holland 1983)
   27-28 : asteer_u, asteer_v (annulus 3-7° DLM steering — Chan & Gray 1982)

Feature order sprint1+ENSO (20):
  base(13) + 6 ERA5 + enso_oni

Feature order sprint1 (19):
  base(13) + 6 ERA5

Feature order legacy (12):
  lat_norm, lon_norm, dlat, dlon, speed_kmh, direction, vmax, pmin,
  month_sin, month_cos, storm_age_h, dist2land
"""

import math
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime


def _parse_dt(iso_time):
    """Parse iso_time -> datetime (stdlib, thay cho pandas). Fallback 2021-09."""
    if iso_time:
        try:
            return datetime.fromisoformat(str(iso_time).replace("Z", "").replace("z", ""))
        except Exception:
            pass
    return datetime(2021, 9, 1)


def _month_of(iso_time) -> int:
    return _parse_dt(iso_time).month if iso_time else 9


class _ScalerShim:
    """Thay StandardScaler của sklearn — chỉ cần mean/scale/transform (numpy)."""
    def __init__(self, mean, scale, n):
        self.mean_ = np.asarray(mean, dtype=np.float64)
        self.scale_ = np.asarray(scale, dtype=np.float64)
        self.n_features_in_ = int(n)

    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def inverse_transform(self, X):
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.mean_

# Priority: scs_v12_lb6 (BEST) → scs_v11_lb6 → scs_v9_lb6 → ...
_BASE = Path(__file__).parent.parent / "model_ai/models"
_SCALER_CANDIDATES = [
    _BASE / "scaler_scs_v12_lb6.pkl",   # 31 feat, BEST (42.0% skill)
    _BASE / "scaler_scs_v11_lb6.pkl",   # 29 feat (41.1%)
    _BASE / "scaler_scs_v9_lb6.pkl",    # 27 feat
    _BASE / "scaler_scs_v9.pkl",        # 27 feat (lb8)
    _BASE / "scaler_scs_v7.pkl",        # 25 feat
    _BASE / "scaler_wp_6h_v7.pkl",
    _BASE / "scaler_wp_6h_v6.pkl",
    _BASE / "scaler_wp_6h_v5.pkl",
    _BASE / "scaler_wp_6h_ext.pkl",
    _BASE / "scaler_wp_6h_enso.pkl",
    _BASE / "scaler_wp_6h_v3.pkl",
    _BASE / "scaler_wp_6h_v2.pkl",
    _BASE / "scaler_wp_6h.pkl",
    _BASE / "scaler_sprint1_6h.pkl",
    _BASE / "scaler_sprint1.pkl",
    _BASE / "scaler_14feat.pkl",
    _BASE / "scaler_wp_full.pkl",
    _BASE / "scaler.pkl",
]
_SCALER_PATH = next((p for p in _SCALER_CANDIDATES if p.exists()), _SCALER_CANDIDATES[-1])

_LAT_MIN, _LAT_MAX = 8.0, 22.0
_LON_MIN, _LON_MAX = 102.0, 120.0
_DIST2LAND_DEFAULT = 252.0
_CLIP_SIGMA = 5.0

_scaler = None
_n_features = None

# ── ENSO / ONI lookup ────────────────────────────────────────────────────────
_ONI_PATH = Path(__file__).parent.parent / "model_ai/data/climate/oni.csv"
_oni_lookup: dict | None = None   # (year, month) → float


def _load_oni_lookup() -> dict:
    global _oni_lookup
    if _oni_lookup is not None:
        return _oni_lookup
    if not _ONI_PATH.exists():
        _oni_lookup = {}
        return _oni_lookup
    try:
        import pandas as pd
        df = pd.read_csv(_ONI_PATH)
        _oni_lookup = {(int(r.year), int(r.month)): float(r.oni)
                       for r in df.itertuples(index=False)}
        print(f"[preprocessor] ONI loaded: {len(_oni_lookup)} entries")
    except Exception as e:
        print(f"[preprocessor] ONI load error: {e}")
        _oni_lookup = {}
    return _oni_lookup


def _get_oni(year: int, month: int) -> float:
    """Tra cứu ONI theo year/month. Fallback: lấy tháng gần nhất có data."""
    lookup = _load_oni_lookup()
    if not lookup:
        return 0.0
    val = lookup.get((year, month))
    if val is not None:
        return val
    # Recent months chưa có trong ONI → lấy entry cuối cùng
    for y in range(year, year - 2, -1):
        for m in range(month, 0, -1):
            v = lookup.get((y, m))
            if v is not None:
                return v
    return 0.0


# SST climatology — khớp với training (NOAA OISST SCS mean)
_SST_CLIM = {1: 26.2, 2: 26.0, 3: 26.8, 4: 28.0, 5: 29.2, 6: 30.1,
             7: 30.3, 8: 30.2, 9: 29.5, 10: 28.4, 11: 27.5, 12: 26.8}


def _sst_clim(month: int) -> float:
    return _SST_CLIM.get(int(month), 28.0)


def _load_scaler():
    global _scaler, _n_features, _SCALER_PATH
    if _scaler is not None:
        return _scaler
    # Ưu tiên .npz (numpy, không cần sklearn). Fallback .pkl nếu có sklearn.
    npz = _SCALER_PATH.with_suffix(".npz")
    if npz.exists():
        d = np.load(npz)
        _scaler = _ScalerShim(d["mean"], d["scale"], int(d["n_features"]))
        _SCALER_PATH = npz
    else:
        if not _SCALER_PATH.exists():
            raise FileNotFoundError(f"Scaler không tồn tại: {_SCALER_PATH}")
        with open(_SCALER_PATH, "rb") as f:
            _scaler = pickle.load(f)
    _n_features = _scaler.n_features_in_
    print(f"[preprocessor] Scaler: {_SCALER_PATH.name} ({_n_features} features)")
    return _scaler


def get_n_features() -> int:
    _load_scaler()
    return _n_features


def _era5_climatology_fallback() -> tuple:
    """
    Trả về giá trị trung bình training (scaler.mean_) cho ERA5 features
    thay vì 0 khi không có file ERA5 — tránh distribution shift tại inference.

    Luôn trả về 12 giá trị: wind_shear, u500, v500, u700, v700, z500,
                            u200, v200, u850, v850, vort500, vort850.
    """
    try:
        scaler = _load_scaler()
        n = getattr(scaler, 'n_features_in_', 0)
        # n >= 25: layout có đủ 12 ERA5 features từ index 13 → 24
        if n >= 25:
            return tuple(float(v) for v in scaler.mean_[13:25])
        elif n >= 23:
            base = tuple(float(v) for v in scaler.mean_[13:23])
            return base + (0.0, 0.0)
        elif n >= 19:
            base = tuple(float(v) for v in scaler.mean_[13:19])
            return base + (5.0, 2.0, -3.0, 2.0, 0.0, 0.0)
    except Exception:
        pass
    return (9.0, 2.0, -3.0, 1.5, -1.0, 5880.0, 5.0, 2.0, -3.0, 2.0, 0.0, 0.0)


def _asteer_climatology_fallback() -> tuple:
    """
    Annulus steering DLM (asteer_u, asteer_v) climatology từ scaler.mean_.
    Layout scs_v11_lb6 (n=29) / scs_v12_lb6 (n=31): index 27–28.
    Fallback hardcode khi không có scaler.
    """
    try:
        scaler = _load_scaler()
        n = getattr(scaler, 'n_features_in_', 0)
        if n >= 29:
            return float(scaler.mean_[27]), float(scaler.mean_[28])
    except Exception:
        pass
    return -3.0, 1.5  # rough easterly mean over SCS


def _asteer850_climatology_fallback() -> tuple:
    """
    Annulus steering 850 hPa (asteer_u850, asteer_v850) climatology.
    Layout scs_v12_lb6 (n=31): index 29–30.
    """
    try:
        scaler = _load_scaler()
        n = getattr(scaler, 'n_features_in_', 0)
        if n >= 31:
            return float(scaler.mean_[29]), float(scaler.mean_[30])
    except Exception:
        pass
    return -3.5, 1.0  # low-level easterly mean over SCS


def _beta_drift(lat: float, dt_h: float = 6.0) -> tuple:
    """
    Beta drift physics prior (Holland 1983, Carr & Elsberry 1990).
    Trùng với features.py — KHÔNG đổi công thức để khớp training distribution.

    Returns (beta_lat_drift, beta_lon_drift) — degrees per dt_h step.
    """
    beta_speed_ms = 2.5 * math.cos(math.radians(lat))   # m/s
    beta_dist_km  = beta_speed_ms * 3.6 * dt_h          # km / dt_h
    beta_dir_rad  = math.radians(320.0)                 # NW direction
    beta_lat = beta_dist_km * math.cos(beta_dir_rad) / 111.0
    beta_lon = (beta_dist_km * math.sin(beta_dir_rad)
                / (111.0 * max(math.cos(math.radians(lat)), 0.1)))
    return beta_lat, beta_lon


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


# ─── ERA5 Multi-Level Extractor ────────────────────────────────────────────────

class _ERA5MultiLevelExtractor:
    """
    Trích xuất 6 ERA5 features từ pressure_4lev files dùng box avg ±5°
    (khớp với training):  wind_shear, u500, v500, u700, v700, z500
    """
    _ERA5_DIR = Path(__file__).parent.parent.parent / "model_ai/data/era5/pressure_4lev"
    _RADIUS = 10.0

    def __init__(self):
        self._year = None
        self._u_full = None   # (n_vt, n_pl, nlat, nlon)
        self._v_full = None
        self._z_full = None   # geopotential height (m) = raw/g
        self._times = None
        self._lats = None
        self._lons = None
        self._idx200 = self._idx500 = self._idx700 = self._idx850 = None
        self._ds = None
        # _n_era5: số ERA5 features mà model hiện tại cần (6 hoặc 10)
        self._n_era5 = None

    def _load_year(self, year: int) -> bool:
        if self._year == year:
            return self._u_full is not None
        if self._ds is not None:
            try:
                self._ds.close()
            except Exception:
                pass
            self._ds = self._u_full = self._v_full = self._z_full = None

        path = self._ERA5_DIR / f"era5_pressure_{year}.nc"
        if not path.exists():
            self._year = year
            return False

        try:
            import xarray as xr
            import pandas as pd
            ds = xr.open_dataset(str(path))
            pl = ds["pressure_level"].values
            self._idx200 = int(np.argmin(np.abs(pl - 200)))
            self._idx500 = int(np.argmin(np.abs(pl - 500)))
            self._idx700 = int(np.argmin(np.abs(pl - 700)))
            self._idx850 = int(np.argmin(np.abs(pl - 850)))

            tc = "valid_time" if "valid_time" in ds.coords else "time"
            self._times = pd.DatetimeIndex(ds[tc].values)
            self._lats = ds["latitude"].values
            self._lons = ds["longitude"].values
            self._u_full = ds["u"].values.astype(np.float32)
            self._v_full = ds["v"].values.astype(np.float32)
            self._z_full = (ds["z"].values / 9.80665).astype(np.float32) if "z" in ds else None
            self._ds = ds
            self._year = year
            return True
        except Exception as e:
            print(f"[ERA5] Lỗi load {path.name}: {e}")
            self._year = year
            return False

    def get(self, lat: float, lon: float, iso_time) -> tuple:
        """
        Returns 16 values:
          (wind_shear, u500, v500, u700, v700, z500,
           u200, v200, u850, v850, vort500, vort850,
           asteer_u, asteer_v,         ← DLM 500+700 annulus
           asteer_u850, asteer_v850)   ← 850 hPa annulus

        - Box (±10°) average cho u/v ở mỗi level
        - Annulus (3-7°) DLM (500+700) cho asteer (Chan & Gray 1982)
        - Annulus (3-7°) tại 850 hPa cho asteer_u850/v850 (low-level steering)
        - Vorticity ζ = ∂v/∂x − ∂u/∂y, scaled ×10⁵ (s⁻¹)
        - Fallback: training means (12), asteer fallback từ scaler hoặc constant
        """
        try:
            ts = _parse_dt(iso_time)
            if not self._load_year(ts.year):
                return (_era5_climatology_fallback()
                        + _asteer_climatology_fallback()
                        + _asteer850_climatology_fallback())

            ti = self._times.get_indexer([np.datetime64(ts)], method="nearest")[0]
            if ti < 0:
                return (_era5_climatology_fallback()
                        + _asteer_climatology_fallback()
                        + _asteer850_climatology_fallback())

            lats, lons = self._lats, self._lons
            lat_mask = (lats >= lat - self._RADIUS) & (lats <= lat + self._RADIUS)
            lon_mask = (lons >= lon - self._RADIUS) & (lons <= lon + self._RADIUS)

            if lat_mask.sum() == 0:
                li = int(np.argmin(np.abs(lats - lat)))
                lat_mask = np.zeros(len(lats), dtype=bool)
                lat_mask[li] = True
            if lon_mask.sum() == 0:
                lo = int(np.argmin(np.abs(lons - lon)))
                lon_mask = np.zeros(len(lons), dtype=bool)
                lon_mask[lo] = True

            def box_mean(arr2d):
                return float(np.nanmean(arr2d[np.ix_(lat_mask, lon_mask)]))

            u200 = box_mean(self._u_full[ti, self._idx200])
            v200 = box_mean(self._v_full[ti, self._idx200])
            u500 = box_mean(self._u_full[ti, self._idx500])
            v500 = box_mean(self._v_full[ti, self._idx500])
            u700 = box_mean(self._u_full[ti, self._idx700])
            v700 = box_mean(self._v_full[ti, self._idx700])
            u850 = box_mean(self._u_full[ti, self._idx850])
            v850 = box_mean(self._v_full[ti, self._idx850])
            wind_shear = float(np.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2))
            z500 = box_mean(self._z_full[ti, self._idx500]) if self._z_full is not None else 0.

            # Relative vorticity ζ = ∂v/∂x − ∂u/∂y at 500 và 850 hPa (×10⁵ s⁻¹)
            R = 6371e3
            lat_step_rad = np.deg2rad(lats[1] - lats[0])
            lon_step_rad = np.deg2rad(lons[1] - lons[0])
            cos_lat = np.maximum(np.cos(np.deg2rad(lats)), 0.01)[:, None]

            def vort_box(idx_lvl):
                u2d = self._u_full[ti, idx_lvl]
                v2d = self._v_full[ti, idx_lvl]
                dudy = np.gradient(u2d, axis=0) / (R * lat_step_rad)
                dvdx = np.gradient(v2d, axis=1) / (R * cos_lat * lon_step_rad)
                vort_field = (dvdx - dudy) * 1e5
                return float(np.nanmean(vort_field[np.ix_(lat_mask, lon_mask)]))

            vort500 = vort_box(self._idx500)
            vort850 = vort_box(self._idx850)

            # ── Annulus deep-layer-mean steering (3-7° từ tâm) ──────────────
            # Trùng logic với era5_extractor.py (training). Loại bỏ vortex bão.
            try:
                lat_sub = lats[lat_mask]
                lon_sub = lons[lon_mask]
                LATG, LONG = np.meshgrid(lat_sub, lon_sub, indexing="ij")
                dlat_d = LATG - lat
                dlon_d = (LONG - lon) * math.cos(math.radians(lat))
                dist_d = np.sqrt(dlat_d ** 2 + dlon_d ** 2)
                ann = (dist_d >= 3.0) & (dist_d <= 7.0)
                if ann.any():
                    u500_box = self._u_full[ti, self._idx500][np.ix_(lat_mask, lon_mask)]
                    v500_box = self._v_full[ti, self._idx500][np.ix_(lat_mask, lon_mask)]
                    u700_box = self._u_full[ti, self._idx700][np.ix_(lat_mask, lon_mask)]
                    v700_box = self._v_full[ti, self._idx700][np.ix_(lat_mask, lon_mask)]
                    dlm_u = (u500_box + u700_box) / 2.0
                    dlm_v = (v500_box + v700_box) / 2.0
                    asteer_u = float(np.nanmean(dlm_u[ann]))
                    asteer_v = float(np.nanmean(dlm_v[ann]))
                    # 850 hPa annulus
                    u850_box = self._u_full[ti, self._idx850][np.ix_(lat_mask, lon_mask)]
                    v850_box = self._v_full[ti, self._idx850][np.ix_(lat_mask, lon_mask)]
                    asteer_u850 = float(np.nanmean(u850_box[ann]))
                    asteer_v850 = float(np.nanmean(v850_box[ann]))
                else:
                    asteer_u, asteer_v = _asteer_climatology_fallback()
                    asteer_u850, asteer_v850 = _asteer850_climatology_fallback()
            except Exception:
                asteer_u, asteer_v = _asteer_climatology_fallback()
                asteer_u850, asteer_v850 = _asteer850_climatology_fallback()

            return (wind_shear, u500, v500, u700, v700, z500,
                    u200, v200, u850, v850, vort500, vort850,
                    asteer_u, asteer_v,
                    asteer_u850, asteer_v850)
        except Exception:
            return (_era5_climatology_fallback()
                        + _asteer_climatology_fallback()
                        + _asteer850_climatology_fallback())


_era5_extractor = _ERA5MultiLevelExtractor()


# ─── Feature matrix ────────────────────────────────────────────────────────────

def _build_base_row(pt, prev, storm_age_h):
    """Tính 13 features chung (lat_norm..dist2land) trả về tuple."""
    lat, lon = pt["lat"], pt["lon"]
    vmax      = pt.get("vmax") or 35.0
    pmin      = pt.get("pmin") or 1000.0
    dist2land = pt.get("dist2land") or _DIST2LAND_DEFAULT
    iso_time  = pt.get("iso_time")
    month     = _month_of(iso_time)

    lat_norm = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
    lon_norm = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)

    if prev is not None:
        dlat      = lat - prev["lat"]
        dlon      = lon - prev["lon"]
        speed_kmh = _haversine_km(prev["lat"], prev["lon"], lat, lon) / 6.0
        direction = _bearing(prev["lat"], prev["lon"], lat, lon)
    else:
        dlat = dlon = speed_kmh = direction = 0.0

    sst       = _sst_clim(month)
    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    return [lat_norm, lon_norm, dlat, dlon, speed_kmh, direction, vmax, pmin,
            sst, month_sin, month_cos, storm_age_h, dist2land]


def _build_row_sprint1(pt, prev, storm_age_h, era5_feat):
    """Build 19-feature row (sprint1): base(13) + wind_shear,u500,v500,u700,v700,z500."""
    wind_shear, u500, v500, u700, v700, z500 = era5_feat[:6]
    return _build_base_row(pt, prev, storm_age_h) + [
        wind_shear, u500, v500, u700, v700, z500,
    ]


def _build_row_sprint1_v5(pt, prev, storm_age_h, era5_feat):
    """Build 23-feature row (sprint2/v5): base(13) + 10 ERA5 features."""
    wind_shear, u500, v500, u700, v700, z500, u200, v200, u850, v850 = era5_feat[:10]
    return _build_base_row(pt, prev, storm_age_h) + [
        wind_shear, u500, v500, u700, v700, z500, u200, v200, u850, v850,
    ]


def _build_row_sprint1_v7(pt, prev, storm_age_h, era5_feat):
    """Build 25-feature row (v7): base(13) + 10 ERA5 + vort500 + vort850."""
    wind_shear, u500, v500, u700, v700, z500, u200, v200, u850, v850, vort500, vort850 = era5_feat[:12]
    return _build_base_row(pt, prev, storm_age_h) + [
        wind_shear, u500, v500, u700, v700, z500, u200, v200, u850, v850,
        vort500, vort850,
    ]


def _build_row_v9(pt, prev, storm_age_h, era5_feat):
    """
    Build 27-feature row (scs_v9 layout): v7 (25) + 2 beta drift (analytical).
    Index 25-26 = beta_lat_drift, beta_lon_drift.
    """
    base = _build_row_sprint1_v7(pt, prev, storm_age_h, era5_feat)
    beta_lat, beta_lon = _beta_drift(pt["lat"])
    return base + [beta_lat, beta_lon]


def _build_row_v11(pt, prev, storm_age_h, era5_feat, asteer):
    """
    Build 29-feature row (scs_v11_lb6 layout): v9 (27) + 2 annulus steering DLM.
    Index 27-28 = asteer_u, asteer_v.
    """
    base = _build_row_v9(pt, prev, storm_age_h, era5_feat)
    return base + [asteer[0], asteer[1]]


def _build_row_v12(pt, prev, storm_age_h, era5_feat, asteer, asteer850):
    """
    Build 31-feature row (scs_v12_lb6 layout): v11 (29) + 2 annulus 850 hPa.
    Index 29-30 = asteer_u850, asteer_v850.
    """
    base = _build_row_v11(pt, prev, storm_age_h, era5_feat, asteer)
    return base + [asteer850[0], asteer850[1]]


def _build_row_legacy(pt, prev, storm_age_h):
    """Build 12-feature row (legacy wp_full order)."""
    lat, lon = pt["lat"], pt["lon"]
    vmax = pt.get("vmax") or 35.0
    pmin = pt.get("pmin") or 1000.0
    dist2land = pt.get("dist2land") or _DIST2LAND_DEFAULT

    iso_time = pt.get("iso_time")
    month = _month_of(iso_time)

    lat_norm = (lat - _LAT_MIN) / (_LAT_MAX - _LAT_MIN)
    lon_norm = (lon - _LON_MIN) / (_LON_MAX - _LON_MIN)

    if prev is not None:
        dlat      = lat - prev["lat"]
        dlon      = lon - prev["lon"]
        speed_kmh = _haversine_km(prev["lat"], prev["lon"], lat, lon) / 6.0
        direction = _bearing(prev["lat"], prev["lon"], lat, lon)
    else:
        dlat = dlon = speed_kmh = direction = 0.0

    month_sin = math.sin(2 * math.pi * month / 12)
    month_cos = math.cos(2 * math.pi * month / 12)

    return [
        lat_norm, lon_norm, dlat, dlon, speed_kmh, direction, vmax, pmin,
        month_sin, month_cos, storm_age_h, dist2land,
    ]


def prepare_input(points: list[dict], lookback: int = 8) -> np.ndarray:
    """
    Lấy lookback điểm cuối, tính features, scale + clip ±5σ.
    Tự động detect 12/19/20/23/25/27/29 features từ scaler.

    Returns
    -------
    np.ndarray shape (1, lookback, n_features)
    """
    scaler = _load_scaler()
    n_feat = _n_features
    window = points[-lookback:]
    rows = []

    if n_feat in (19, 20, 23, 25, 27, 29, 31):
        # Sprint1/v7/v9/v11/v12: extract ERA5, use persistence fallback
        last_era5 = _era5_climatology_fallback()
        era5_feats = []
        for pt in window:
            iso_time = pt.get("iso_time")
            if iso_time:
                feats = _era5_extractor.get(pt["lat"], pt["lon"], iso_time)
                if any(f != 0. for f in feats):
                    last_era5 = feats
                else:
                    feats = last_era5
            else:
                feats = last_era5
            era5_feats.append(feats)

        # asteer climatology fallback
        asteer_clim    = _asteer_climatology_fallback()    if n_feat >= 29 else None
        asteer850_clim = _asteer850_climatology_fallback() if n_feat >= 31 else None

        for i, pt in enumerate(window):
            prev = window[i - 1] if i > 0 else None
            storm_age_h = i * 6.0
            if n_feat == 31:
                feats = era5_feats[i]
                asteer = (feats[12], feats[13]) if len(feats) >= 14 else asteer_clim
                asteer850 = (feats[14], feats[15]) if len(feats) >= 16 else asteer850_clim
                row = _build_row_v12(pt, prev, storm_age_h, feats, asteer, asteer850)
            elif n_feat == 29:
                feats = era5_feats[i]
                asteer = (feats[12], feats[13]) if len(feats) >= 14 else asteer_clim
                row = _build_row_v11(pt, prev, storm_age_h, feats, asteer)
            elif n_feat == 27:
                row = _build_row_v9(pt, prev, storm_age_h, era5_feats[i])
            elif n_feat == 25:
                row = _build_row_sprint1_v7(pt, prev, storm_age_h, era5_feats[i])
            elif n_feat == 23:
                row = _build_row_sprint1_v5(pt, prev, storm_age_h, era5_feats[i])
            else:
                row = _build_row_sprint1(pt, prev, storm_age_h, era5_feats[i])
            if n_feat == 20:
                iso_time = pt.get("iso_time")
                ts = _parse_dt(iso_time) if iso_time else None
                oni = _get_oni(ts.year, ts.month) if ts else 0.0
                row.append(oni)
            rows.append(row)

    else:
        for i, pt in enumerate(window):
            prev = window[i - 1] if i > 0 else None
            storm_age_h = i * 6.0
            rows.append(_build_row_legacy(pt, prev, storm_age_h))

    feat_matrix = np.array(rows, dtype=np.float32)       # (lookback, n_feat)
    feat_scaled = scaler.transform(feat_matrix)           # (lookback, n_feat)

    # Clip ±5σ cho mọi tag từ sprint1 trở đi (steering features có thể heavy-tail)
    if n_feat >= 19:
        feat_scaled = np.clip(feat_scaled, -_CLIP_SIGMA, _CLIP_SIGMA)

    return feat_scaled[np.newaxis].astype(np.float32)     # (1, lookback, n_feat)


def compute_cliper(x_scaled: np.ndarray, scaler, anchor_steps: list) -> np.ndarray:
    """
    CLIPER: constant velocity extrapolation dùng velocity trung bình 3 bước cuối.

    Parameters
    ----------
    anchor_steps : list of int — bước dự đoán, ví dụ [1,2,...,8] hoặc [4,8]

    Returns
    -------
    np.ndarray shape (1, len(anchor_steps)*2) — [lat_s1, lon_s1, lat_s2, lon_s2, ...]
    """
    lookback = x_scaled.shape[1]
    n_feat   = x_scaled.shape[2]

    last_scaled = x_scaled[0, -1, :]
    last_orig   = scaler.inverse_transform(last_scaled.reshape(1, -1))[0]
    lat_cur = last_orig[0] * (_LAT_MAX - _LAT_MIN) + _LAT_MIN
    lon_cur = last_orig[1] * (_LON_MAX - _LON_MIN) + _LON_MIN

    # Velocity trung bình 3 bước cuối — giảm bias khi bão tăng/giảm tốc đột ngột
    n_avg     = min(3, lookback)
    last_n    = scaler.inverse_transform(x_scaled[0, -n_avg:, :])  # (n_avg, n_feat)
    dlat      = float(last_n[:, 2].mean())
    dlon      = float(last_n[:, 3].mean())

    cliper = []
    for k in anchor_steps:
        cliper.extend([lat_cur + k * dlat, lon_cur + k * dlon])

    return np.array([cliper], dtype=np.float32)


def decode_output(pred: np.ndarray, x_scaled: np.ndarray, scaler) -> np.ndarray:
    """
    model output (delta) + CLIPER → lat/lon thực (degrees).

    Returns
    -------
    np.ndarray shape (n_output,) — lat/lon pairs for each anchor step
    """
    n_out = pred.shape[1]
    if n_out == 16:
        anchor_steps = list(range(1, 9))   # sprint1: steps 1–8 (+6h to +48h)
    else:
        anchor_steps = [4, 8]              # legacy: +24h, +48h

    cliper = compute_cliper(x_scaled, scaler, anchor_steps)
    return (cliper + pred)[0]              # (n_output,)
