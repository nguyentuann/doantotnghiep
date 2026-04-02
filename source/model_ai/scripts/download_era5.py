"""
download_era5.py — Tải dữ liệu ERA5 từ Copernicus Climate Data Store (CDS)
---------------------------------------------------------------------------
Yêu cầu:
  1. Đăng ký tài khoản tại https://cds.climate.copernicus.eu
  2. Tạo file ~/.cdsapirc với nội dung:
         url: https://cds.climate.copernicus.eu/api
         key: <UID>:<API_KEY>
  3. pip install cdsapi xarray netcdf4

Chạy:
  cd source/model_ai
  python scripts/download_era5.py                     # tải tất cả
  python scripts/download_era5.py --years 2020 2021   # tải năm cụ thể
  python scripts/download_era5.py --check             # kiểm tra file đã tải

Output:
  data/era5/pressure/era5_pressure_<year>.nc   — wind shear, geopotential, humidity
  data/era5/sst/era5_sst_<year>.nc             — sea surface temperature
"""

import argparse
import sys
import time
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent          # source/model_ai/
ERA5_DIR   = BASE_DIR / "data/era5"
PRES_DIR   = ERA5_DIR / "pressure"
SST_DIR    = ERA5_DIR / "sst"

YEAR_START = 1980
YEAR_END   = 2024

# Vùng bao phủ Biển Đông + vùng đệm (North, West, South, East)
AREA = [35, 95, -5, 145]

# Các giờ UTC khớp với IBTrACS (6h resolution)
TIMES = ["00:00", "06:00", "12:00", "18:00"]

# Pressure levels cần thiết cho wind shear và steering flow
PRESSURE_LEVELS = ["200", "500", "700", "850"]

# Biến tải từ pressure levels
PRESSURE_VARS = [
    "u_component_of_wind",   # U-wind → wind shear (200–850 hPa)
    "v_component_of_wind",   # V-wind → wind shear
    "geopotential",          # Geopotential height → steering flow (500 hPa)
    "relative_humidity",     # Relative humidity → intensity (700 hPa)
]

# Biến tải từ single levels
SINGLE_VARS = [
    "sea_surface_temperature",   # SST thực tế thay climatology
]

MAX_RETRIES = 3
RETRY_WAIT  = 30   # giây chờ giữa mỗi lần retry


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _all_days():
    return [f"{d:02d}" for d in range(1, 32)]


def _all_months():
    return [f"{m:02d}" for m in range(1, 13)]


def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def _download_with_retry(client, dataset: str, request: dict, output: Path):
    """Tải 1 file với retry tự động khi lỗi mạng."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"    [attempt {attempt}/{MAX_RETRIES}] → {output.name}")
            client.retrieve(dataset, request, str(output))
            print(f"    [ok] {output.name}  ({_size_mb(output):.1f} MB)")
            return True
        except Exception as e:
            print(f"    [error] {e}")
            if attempt < MAX_RETRIES:
                print(f"    Chờ {RETRY_WAIT}s rồi thử lại...")
                time.sleep(RETRY_WAIT)
            else:
                print(f"    [fail] Bỏ qua {output.name} sau {MAX_RETRIES} lần thử.")
                return False


# ─── Download functions ───────────────────────────────────────────────────────

def download_pressure_year(client, year: int) -> bool:
    """Tải ERA5 pressure levels cho 1 năm."""
    output = PRES_DIR / f"era5_pressure_{year}.nc"
    if output.exists():
        print(f"  [skip] {output.name} đã tồn tại ({_size_mb(output):.1f} MB)")
        return True

    request = {
        "product_type": "reanalysis",
        "variable":       PRESSURE_VARS,
        "pressure_level": PRESSURE_LEVELS,
        "year":           [str(year)],
        "month":          _all_months(),
        "day":            _all_days(),
        "time":           TIMES,
        "area":           AREA,
        "format":         "netcdf",
    }
    return _download_with_retry(client, "reanalysis-era5-pressure-levels", request, output)


def download_sst_year(client, year: int) -> bool:
    """Tải ERA5 SST cho 1 năm."""
    output = SST_DIR / f"era5_sst_{year}.nc"
    if output.exists():
        print(f"  [skip] {output.name} đã tồn tại ({_size_mb(output):.1f} MB)")
        return True

    request = {
        "product_type": "reanalysis",
        "variable":     SINGLE_VARS,
        "year":         [str(year)],
        "month":        _all_months(),
        "day":          _all_days(),
        "time":         TIMES,
        "area":         AREA,
        "format":       "netcdf",
    }
    return _download_with_retry(client, "reanalysis-era5-single-levels", request, output)


# ─── Check mode ───────────────────────────────────────────────────────────────

def check_downloads(years: list[int]):
    """In trạng thái các file đã tải."""
    print(f"\n{'─'*55}")
    print(f"  {'Year':<6} {'Pressure':>15} {'SST':>12}")
    print(f"  {'─'*46}")

    total_mb = 0.0
    missing  = []

    for year in years:
        pres = PRES_DIR / f"era5_pressure_{year}.nc"
        sst  = SST_DIR  / f"era5_sst_{year}.nc"

        pres_str = f"{_size_mb(pres):8.1f} MB" if pres.exists() else "   MISSING"
        sst_str  = f"{_size_mb(sst):8.1f} MB"  if sst.exists()  else "   MISSING"

        if pres.exists():
            total_mb += _size_mb(pres)
        else:
            missing.append(f"pressure/{year}")

        if sst.exists():
            total_mb += _size_mb(sst)
        else:
            missing.append(f"sst/{year}")

        print(f"  {year:<6} {pres_str:>15} {sst_str:>12}")

    print(f"{'─'*55}")
    print(f"  Tổng: {total_mb:.1f} MB  ({total_mb/1024:.2f} GB)")
    if missing:
        print(f"  Còn thiếu {len(missing)} file: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
    else:
        print("  Tất cả file đã tải xong.")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tải ERA5 từ CDS")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=list(range(YEAR_START, YEAR_END + 1)),
        help=f"Danh sách năm cần tải (mặc định: {YEAR_START}–{YEAR_END})",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Chỉ kiểm tra file đã tải, không tải thêm",
    )
    parser.add_argument(
        "--no-pressure", action="store_true",
        help="Bỏ qua tải pressure levels",
    )
    parser.add_argument(
        "--no-sst", action="store_true",
        help="Bỏ qua tải SST",
    )
    args = parser.parse_args()

    years = sorted(args.years)

    # Tạo thư mục output
    PRES_DIR.mkdir(parents=True, exist_ok=True)
    SST_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_downloads(years)
        return

    # Import cdsapi (chỉ khi cần tải)
    try:
        import cdsapi
    except ImportError:
        print("[error] cdsapi chưa cài. Chạy: pip install cdsapi")
        sys.exit(1)

    client = cdsapi.Client()

    print(f"\n{'='*55}")
    print(f"  ERA5 Download  |  {len(years)} năm: {years[0]}–{years[-1]}")
    print(f"  Vùng: {AREA}  (N W S E)")
    print(f"  Output: {ERA5_DIR}")
    print(f"{'='*55}\n")

    ok_count   = 0
    fail_count = 0

    for year in years:
        print(f"\n[{year}]")

        if not args.no_pressure:
            success = download_pressure_year(client, year)
            if success:
                ok_count += 1
            else:
                fail_count += 1

        if not args.no_sst:
            success = download_sst_year(client, year)
            if success:
                ok_count += 1
            else:
                fail_count += 1

    print(f"\n{'='*55}")
    print(f"  Hoàn thành: {ok_count} file OK, {fail_count} file lỗi")
    check_downloads(years)


if __name__ == "__main__":
    main()
