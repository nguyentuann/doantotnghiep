"""
era5_extractor.py
-----------------
Interpolate dữ liệu khí quyển tới từng điểm track bão.

Nguồn dữ liệu:
  - ERA5 pressure:  data/era5/pressure/era5_pressure_{year}_{month:02d}.nc  (u/v wind 200+850 hPa)
  - NOAA OISST v2:  data/noaa_sst/sst.mnmean.nc  (SST monthly mean 1°, 1 file duy nhất)

Output cho mỗi điểm track:
  wind_shear  — |V200 - V850| (m/s) = sqrt((u200-u850)² + (v200-v850)²)
  sst_actual  — SST thực tế (°C) theo tháng, fallback về climatology
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import xarray as xr
    HAS_XARRAY = True
except ImportError:
    HAS_XARRAY = False


# SST climatology fallback (dùng khi cả NOAA lẫn ERA5 đều không có)
_SST_CLIM = {
    1: 26.2, 2: 26.0, 3: 26.8, 4: 28.0, 5: 29.2, 6: 30.1,
    7: 30.3, 8: 30.2, 9: 29.5, 10: 28.4, 11: 27.5, 12: 26.8,
}


# =====================================================================
# ERA5 Wind Shear Extractor
# =====================================================================

class ERA5WindShear:
    """Load ERA5 yearly pressure files, tính wind shear tại storm point.

    Mỗi file năm (era5_pressure_{year}.nc) được cache trong bộ nhớ.
    Fallback: nếu không có file năm, thử load từ file tháng lẻ (era5_pressure_{year}_{month:02d}.nc).
    """

    def __init__(self, era5_dir: Path):
        self.pres_dir = Path(era5_dir) / "pressure"
        self._cache: dict = {}   # key: year → xr.Dataset hoặc None

    def _load(self, year: int):
        if year in self._cache:
            return self._cache[year]

        # Ưu tiên file năm
        yearly_path = self.pres_dir / f"era5_pressure_{year}.nc"
        if yearly_path.exists():
            ds = xr.open_dataset(yearly_path)
            ds = self._normalize_coords(ds)
            self._cache[year] = ds
            return ds

        # Fallback: ghép từ các file tháng còn sót (không cần dask)
        monthly = sorted(self.pres_dir.glob(f"era5_pressure_{year}_*.nc"))
        if monthly:
            datasets = [xr.open_dataset(str(f)) for f in monthly]
            ds = xr.concat(datasets, dim="time")
            for d in datasets:
                d.close()
            ds = self._normalize_coords(ds)
            self._cache[year] = ds
            return ds

        self._cache[year] = None
        return None

    @staticmethod
    def _normalize_coords(ds):
        rename = {}
        if "lat" in ds.coords and "latitude" not in ds.coords:
            rename["lat"] = "latitude"
        if "lon" in ds.coords and "longitude" not in ds.coords:
            rename["lon"] = "longitude"
        if "level" in ds.coords and "pressure_level" not in ds.coords:
            rename["level"] = "pressure_level"
        return ds.rename(rename) if rename else ds

    def get(self, lat: float, lon: float, time: pd.Timestamp) -> float:
        ds = self._load(time.year)
        if ds is None:
            return np.nan
        try:
            # ERA5 files dùng 'valid_time' làm time coordinate (không phải 'time')
            time_coord = "valid_time" if "valid_time" in ds.coords else "time"
            pt = ds.sel({time_coord: np.datetime64(time)}, method="nearest")
            if "time" in pt.dims:
                pt = pt.mean(dim="time", skipna=True)
            pt = pt.squeeze()
            pt = pt.interp(latitude=lat, longitude=lon, method="linear")
            u200 = float(pt["u"].sel(pressure_level=200.0, method="nearest"))
            v200 = float(pt["v"].sel(pressure_level=200.0, method="nearest"))
            u850 = float(pt["u"].sel(pressure_level=850.0, method="nearest"))
            v850 = float(pt["v"].sel(pressure_level=850.0, method="nearest"))
            return math.sqrt((u200 - u850) ** 2 + (v200 - v850) ** 2)
        except Exception:
            return np.nan

    def close(self):
        for ds in self._cache.values():
            if ds is not None:
                ds.close()
        self._cache.clear()


# =====================================================================
# NOAA OISST Extractor (1 file duy nhất, monthly mean, 1°)
# =====================================================================

class NOAAsstExtractor:
    """Load sst.mnmean.nc (NOAA OISST v2), interpolate SST tại storm point."""

    def __init__(self, sst_path: Path):
        self._ds = None
        self._path = Path(sst_path)

    def _load(self):
        if self._ds is not None:
            return self._ds
        if not self._path.exists():
            return None
        ds = xr.open_dataset(self._path)
        # Chuẩn hóa coordinate names
        rename = {}
        if "lat" in ds.coords and "latitude" not in ds.coords:
            rename["lat"] = "latitude"
        if "lon" in ds.coords and "longitude" not in ds.coords:
            rename["lon"] = "longitude"
        if rename:
            ds = ds.rename(rename)
        self._ds = ds
        return ds

    def get(self, lat: float, lon: float, time: pd.Timestamp) -> float:
        ds = self._load()
        if ds is None:
            return _SST_CLIM.get(time.month, 28.0)
        try:
            pt = ds.sel(time=np.datetime64(time), method="nearest") \
                   .interp(latitude=lat, longitude=lon, method="linear")
            sst_v = float(pt["sst"])
            if np.isnan(sst_v):
                return _SST_CLIM.get(time.month, 28.0)
            # NOAA OISST v2 thường ở °C, nhưng kiểm tra Kelvin
            return sst_v - 273.15 if sst_v > 100 else sst_v
        except Exception:
            return _SST_CLIM.get(time.month, 28.0)

    def close(self):
        if self._ds is not None:
            self._ds.close()
            self._ds = None


# =====================================================================
# Batch extraction: Steering Flow (steering_u, steering_v)
# =====================================================================

def extract_steering_features(feat_df: pd.DataFrame, config: dict,
                               radius_deg: float = 5.0) -> pd.DataFrame:
    """
    Thêm cột steering_u và steering_v vào DataFrame features.

    Steering flow = trung bình U/V tại (850+200 hPa) trong vùng ±radius_deg
    xung quanh tâm bão. Đây là luồng gió môi trường "cuốn" bão di chuyển.

    Parameters
    ----------
    feat_df    : DataFrame output của build_features()
    config     : dict từ config.yaml
    radius_deg : bán kính box trung bình (degrees), mặc định 5°

    Returns
    -------
    DataFrame với 2 cột mới: steering_u (m/s), steering_v (m/s)
    """
    base_dir = Path(__file__).parent.parent.parent
    feat_df  = feat_df.copy()

    if not HAS_XARRAY:
        print("[ERA5-Steering] xarray chưa cài — fallback về 0.0")
        feat_df["steering_u"] = 0.0
        feat_df["steering_v"] = 0.0
        return feat_df

    era5_dir = base_dir / config["data"]["era5_dir"]
    wind_ext = ERA5WindShear(era5_dir)

    n = len(feat_df)
    su_vals = np.full(n, np.nan, dtype=np.float32)
    sv_vals = np.full(n, np.nan, dtype=np.float32)

    times = pd.to_datetime(feat_df["ISO_TIME"])
    lats  = feat_df["LAT"].values
    lons  = feat_df["LON"].values
    years = times.dt.year.values
    unique_years = np.unique(years)

    for yi, year in enumerate(unique_years):
        year_mask = years == year
        year_idxs = np.where(year_mask)[0]

        print(f"  [Steering] {year} ({yi+1}/{len(unique_years)}): mở file...", flush=True)
        ds = wind_ext._load(year)
        if ds is None:
            print(f"  [Steering] {year}: không có file ERA5 — bỏ qua", flush=True)
            continue

        try:
            time_coord = "valid_time" if "valid_time" in ds.coords else "time"
            era5_times = pd.DatetimeIndex(ds[time_coord].values)
            era5_lats  = ds["latitude"].values
            era5_lons  = ds["longitude"].values
            pl         = ds["pressure_level"].values

            idx_200 = int(np.argmin(np.abs(pl - 200.0)))
            idx_850 = int(np.argmin(np.abs(pl - 850.0)))

            print(f"    đọc u/v vào RAM...", flush=True)
            if "time" in ds.dims:
                u_full = ds["u"].mean(dim="time", skipna=True).values.astype(np.float32)
                v_full = ds["v"].mean(dim="time", skipna=True).values.astype(np.float32)
            else:
                u_full = ds["u"].values.astype(np.float32)  # (n_vt, n_pl, nlat, nlon)
                v_full = ds["v"].values.astype(np.float32)

            # Steering = mean of 850 and 200 hPa
            su_full = (u_full[:, idx_200] + u_full[:, idx_850]) / 2.0  # (n_vt, nlat, nlon)
            sv_full = (v_full[:, idx_200] + v_full[:, idx_850]) / 2.0

            storm_times = times.iloc[year_idxs]
            nearest_idx = era5_times.get_indexer(storm_times, method="nearest")

            # Lat tăng dần cho slicing nhất quán
            lat_asc = era5_lats[0] < era5_lats[-1]

            n_done = 0
            for local_i, global_i in enumerate(year_idxs):
                ti   = nearest_idx[local_i]
                slat = lats[global_i]
                slon = lons[global_i]

                # Box ±radius_deg
                lat_lo = slat - radius_deg
                lat_hi = slat + radius_deg
                lon_lo = slon - radius_deg
                lon_hi = slon + radius_deg

                lat_mask = (era5_lats >= min(lat_lo, lat_hi)) & (era5_lats <= max(lat_lo, lat_hi))
                lon_mask = (era5_lons >= lon_lo) & (era5_lons <= lon_hi)

                if lat_mask.sum() == 0 or lon_mask.sum() == 0:
                    # Điểm nằm ngoài ERA5 domain — fallback về tại điểm
                    lat_idx = int(np.argmin(np.abs(era5_lats - slat)))
                    lon_idx = int(np.argmin(np.abs(era5_lons - slon)))
                    su_vals[global_i] = su_full[ti, lat_idx, lon_idx]
                    sv_vals[global_i] = sv_full[ti, lat_idx, lon_idx]
                else:
                    box_su = su_full[ti][np.ix_(lat_mask, lon_mask)]
                    box_sv = sv_full[ti][np.ix_(lat_mask, lon_mask)]
                    su_vals[global_i] = float(np.nanmean(box_su))
                    sv_vals[global_i] = float(np.nanmean(box_sv))
                n_done += 1

            print(f"  [Steering] {year}: {n_done:,} rows ✓", flush=True)

        except Exception as e:
            print(f"  [Steering] {year}: lỗi — {e}", flush=True)
        finally:
            if year in wind_ext._cache and wind_ext._cache[year] is not None:
                wind_ext._cache[year].close()
                del wind_ext._cache[year]

    wind_ext.close()

    feat_df["steering_u"] = su_vals
    feat_df["steering_v"] = sv_vals

    # Fill NaN: median cùng tháng → global median → 0
    month_col = times.dt.month
    for col in ["steering_u", "steering_v"]:
        feat_df[col] = feat_df.groupby(month_col)[col].transform(
            lambda s: s.fillna(s.median())
        )
        gmed = feat_df[col].median()
        feat_df[col] = feat_df[col].fillna(0.0 if np.isnan(gmed) else gmed)

    covered = int((~np.isnan(su_vals)).sum())
    print(f"[Steering] coverage: {covered:,}/{n:,} rows ({covered/n*100:.1f}%)")
    return feat_df


# =====================================================================
# Batch extraction cho toàn bộ feature DataFrame
# =====================================================================

def extract_era5_features(feat_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Thêm cột wind_shear và sst_actual vào DataFrame features.

    - wind_shear: từ ERA5 pressure (u/v 200+850 hPa). NaN nếu chưa tải ERA5.
    - sst_actual: từ NOAA OISST monthly mean. Fallback về sst_c climatology.
    - NaN wind_shear được fill bằng median cùng tháng.

    Tối ưu tốc độ: batch theo (year, valid_time) — mỗi time-slice ERA5 chỉ load 1 lần,
    interpolate toàn bộ điểm cùng timestamp cùng lúc bằng scipy RegularGridInterpolator.

    Parameters
    ----------
    feat_df : DataFrame output của build_features()
    config  : dict từ config.yaml

    Returns
    -------
    DataFrame với 2 cột mới: wind_shear, sst_actual
    """
    base_dir = Path(__file__).parent.parent.parent  # model_ai/
    feat_df  = feat_df.copy()

    if not HAS_XARRAY:
        print("[ERA5] xarray chưa cài (pip install xarray netcdf4) — dùng fallback.")
        feat_df["wind_shear"] = 0.0
        feat_df["sst_actual"] = feat_df["sst_c"]
        return feat_df

    try:
        from scipy.interpolate import RegularGridInterpolator
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False

    era5_dir  = base_dir / config["data"]["era5_dir"]
    noaa_path = base_dir / config["data"]["noaa_sst"]

    sst_ext = NOAAsstExtractor(noaa_path)
    wind_ext = ERA5WindShear(era5_dir)

    n = len(feat_df)
    wind_shear_vals = np.full(n, np.nan, dtype=np.float32)
    sst_actual_vals = np.full(n, np.nan, dtype=np.float32)

    times = pd.to_datetime(feat_df["ISO_TIME"])
    lats  = feat_df["LAT"].values
    lons  = feat_df["LON"].values

    covered_ws  = 0
    covered_sst = 0

    # Checkpoint: load cache nếu đã chạy dở
    cache_path = base_dir / "data/features/_wind_shear_cache.npy"
    if cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape == wind_shear_vals.shape:
            wind_shear_vals = cached.copy()
            covered_ws = int(np.sum(~np.isnan(wind_shear_vals)))
            print(f"[ERA5] Tìm thấy cache: {covered_ws:,}/{n:,} rows đã có wind_shear")

    # ── SST: vẫn dùng extractor cũ (NOAA file nhỏ, nhanh) ─────────────────────
    print(f"[NOAA] Đang extract SST cho {n:,} rows...")
    for pos in range(n):
        st = sst_ext.get(lats[pos], lons[pos], times.iloc[pos])
        sst_actual_vals[pos] = st
        if not np.isnan(st):
            covered_sst += 1
        if (pos + 1) % 5000 == 0:
            print(f"  [NOAA] {pos+1:,}/{n:,} rows done")
    sst_ext.close()

    # ── Wind shear: load toàn bộ timestamps cần thiết mỗi năm 1 lần ─────────────
    years = times.dt.year.values
    unique_years = np.unique(years)
    total_years  = len(unique_years)

    for yi, year in enumerate(unique_years):
        year_mask = years == year
        year_idxs = np.where(year_mask)[0]

        # Skip năm đã có đủ cache
        if not np.isnan(wind_shear_vals[year_idxs]).any():
            print(f"  [ERA5] {year} ({yi+1}/{total_years}): skip (đã có cache)", flush=True)
            covered_ws += int(np.sum(~np.isnan(wind_shear_vals[year_idxs])))
            continue

        print(f"  [ERA5] {year} ({yi+1}/{total_years}): mở file...", flush=True)
        ds = wind_ext._load(year)
        if ds is None:
            print(f"  [ERA5] {year}: không có file — bỏ qua", flush=True)
            continue

        try:
            time_coord = "valid_time" if "valid_time" in ds.coords else "time"
            era5_times = pd.DatetimeIndex(ds[time_coord].values)
            era5_lats  = ds["latitude"].values
            era5_lons  = ds["longitude"].values
            pl         = ds["pressure_level"].values

            idx_200 = int(np.argmin(np.abs(pl - 200.0)))
            idx_850 = int(np.argmin(np.abs(pl - 850.0)))

            # Load TOÀN BỘ u/v cả năm vào numpy 1 lần (đọc tuần tự → nhanh)
            # shape: (valid_time, pressure_level, lat, lon) sau khi fix
            print(f"    đọc u/v vào RAM...", flush=True)
            if "time" in ds.dims:
                # file cũ chưa fix: collapse time artifact trước
                u_full = ds["u"].mean(dim="time", skipna=True).values.astype(np.float32)
                v_full = ds["v"].mean(dim="time", skipna=True).values.astype(np.float32)
            else:
                u_full = ds["u"].values.astype(np.float32)  # (n_vt, n_pl, nlat, nlon)
                v_full = ds["v"].values.astype(np.float32)
            print(f"    u/v loaded: {u_full.shape}", flush=True)

            # Tìm nearest ERA5 timestamp index cho storm rows trong năm
            storm_times  = times.iloc[year_idxs]
            nearest_idx  = era5_times.get_indexer(storm_times, method="nearest")
            unique_t_idx = np.unique(nearest_idx)

            # Wind shear field toàn bộ timestamps cần: (n_ts, nlat, nlon)
            u200 = u_full[unique_t_idx, idx_200]   # numpy fancy indexing — fast
            v200 = v_full[unique_t_idx, idx_200]
            u850 = u_full[unique_t_idx, idx_850]
            v850 = v_full[unique_t_idx, idx_850]

            ws_all = np.sqrt((u200 - u850)**2 + (v200 - v850)**2)  # (n_ts, nlat, nlon)

            # Lat tăng dần cho RegularGridInterpolator
            if era5_lats[0] > era5_lats[-1]:
                ws_all      = ws_all[:, ::-1, :]
                interp_lats = era5_lats[::-1]
            else:
                interp_lats = era5_lats

            # Map từ nearest_idx → vị trí trong unique_t_idx
            t_pos_map = {ti: k for k, ti in enumerate(unique_t_idx)}

            n_done = 0
            for ti in unique_t_idx:
                rows_at_t = year_idxs[nearest_idx == ti]
                k = t_pos_map[ti]
                ws_field = ws_all[k]   # (nlat, nlon)

                interp = RegularGridInterpolator(
                    (interp_lats, era5_lons), ws_field,
                    method="linear", bounds_error=False, fill_value=np.nan,
                )
                pts     = np.stack([lats[rows_at_t], lons[rows_at_t]], axis=1)
                ws_vals = interp(pts).astype(np.float32)

                for kk, pos in enumerate(rows_at_t):
                    wind_shear_vals[pos] = ws_vals[kk]
                    if not np.isnan(ws_vals[kk]):
                        covered_ws += 1
                n_done += len(rows_at_t)

            pct = n_done / year_mask.sum() * 100
            print(f"  [ERA5] {year} ({yi+1}/{total_years}): {n_done:,} rows, {pct:.0f}% ✓", flush=True)

        except Exception as e:
            print(f"  [ERA5] {year}: lỗi — {e}", flush=True)

        finally:
            # Lưu checkpoint sau mỗi năm
            np.save(cache_path, wind_shear_vals)
            # Giải phóng RAM ngay sau mỗi năm
            if year in wind_ext._cache and wind_ext._cache[year] is not None:
                wind_ext._cache[year].close()
                del wind_ext._cache[year]

    wind_ext.close()
    # Cache _wind_shear_cache.npy được giữ lại đến khi features.py lưu xong
    # sequences_14feat.npz, sau đó features.py sẽ tự xóa.

    feat_df["wind_shear"] = wind_shear_vals
    feat_df["sst_actual"] = sst_actual_vals

    # --- Fill NaN wind_shear: median cùng tháng, sau đó global median ---
    month_col = times.dt.month
    feat_df["wind_shear"] = feat_df.groupby(month_col)["wind_shear"].transform(
        lambda s: s.fillna(s.median())
    )
    global_median = feat_df["wind_shear"].median()
    # Nếu tất cả NaN (chưa tải ERA5) → fill 0
    if np.isnan(global_median):
        global_median = 0.0
    feat_df["wind_shear"] = feat_df["wind_shear"].fillna(global_median)

    # --- Fill NaN sst_actual: dùng sst_c climatology ---
    sst_nan = feat_df["sst_actual"].isna()
    feat_df.loc[sst_nan, "sst_actual"] = feat_df.loc[sst_nan, "sst_c"]

    pct_ws  = covered_ws  / n * 100
    pct_sst = covered_sst / n * 100
    print(f"[ERA5] wind_shear coverage: {pct_ws:.1f}%  ({covered_ws:,}/{n:,} rows)")
    print(f"[NOAA] sst_actual coverage: {pct_sst:.1f}%  ({covered_sst:,}/{n:,} rows)")

    return feat_df
