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
            pt   = ds.sel(time=np.datetime64(time), method="nearest") \
                     .interp(latitude=lat, longitude=lon, method="linear")
            u200 = float(pt["u"].sel(pressure_level=200))
            v200 = float(pt["v"].sel(pressure_level=200))
            u850 = float(pt["u"].sel(pressure_level=850))
            v850 = float(pt["v"].sel(pressure_level=850))
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
# Batch extraction cho toàn bộ feature DataFrame
# =====================================================================

def extract_era5_features(feat_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Thêm cột wind_shear và sst_actual vào DataFrame features.

    - wind_shear: từ ERA5 pressure (u/v 200+850 hPa). NaN nếu chưa tải ERA5.
    - sst_actual: từ NOAA OISST monthly mean. Fallback về sst_c climatology.
    - NaN wind_shear được fill bằng median cùng tháng.

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

    era5_dir  = base_dir / config["data"]["era5_dir"]
    noaa_path = base_dir / config["data"]["noaa_sst"]

    wind_ext = ERA5WindShear(era5_dir)
    sst_ext  = NOAAsstExtractor(noaa_path)

    n = len(feat_df)
    wind_shear_vals = np.full(n, np.nan, dtype=np.float32)
    sst_actual_vals = np.full(n, np.nan, dtype=np.float32)

    times = pd.to_datetime(feat_df["ISO_TIME"])

    covered_ws  = 0
    covered_sst = 0

    # Nhóm theo year-month → mở mỗi file ERA5 đúng 1 lần
    ym_col = times.dt.to_period("M")

    for ym, idx_list in feat_df.groupby(ym_col).groups.items():
        for pos in idx_list:
            row = feat_df.iloc[pos]
            t   = times.iloc[pos]

            ws = wind_ext.get(row["LAT"], row["LON"], t)
            st = sst_ext.get(row["LAT"], row["LON"], t)

            wind_shear_vals[pos] = ws
            sst_actual_vals[pos] = st

            if not np.isnan(ws):
                covered_ws += 1
            if not np.isnan(st):
                covered_sst += 1

    wind_ext.close()
    sst_ext.close()

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
