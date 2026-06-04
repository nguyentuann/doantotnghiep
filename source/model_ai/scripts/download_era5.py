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
PRES_DIR   = ERA5_DIR / "pressure_4lev"   # Sprint 1: 4 levels (200/500/700/850) + geopotential
SST_DIR    = ERA5_DIR / "sst"

YEAR_START = 1979
YEAR_END   = 2024

# Vùng bao phủ Biển Đông + vùng đệm (North, West, South, East)
AREA = [35, 95, -5, 145]

# Các giờ UTC khớp với IBTrACS (6h resolution)
TIMES = ["00:00", "06:00", "12:00", "18:00"]

# Sprint 1: 4 pressure levels — wind shear (200/850) + steering flow (500/700)
PRESSURE_LEVELS = ["200", "500", "700", "850"]

# u/v wind + geopotential (z500 → ridge/trough tín hiệu recurve)
PRESSURE_VARS = [
    "u_component_of_wind",   # U-wind → wind shear + steering flow
    "v_component_of_wind",   # V-wind → wind shear + steering flow
    "geopotential",          # z500 → subtropical high position, recurve signal
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

def _merge_monthly_to_yearly(monthly_files: "list[Path]", yearly_path: Path) -> bool:
    """Merge 12 file tháng thành 1 file năm với cấu trúc sạch (valid_time làm dim chính)."""
    try:
        import xarray as xr
    except ImportError:
        print("  [warn] xarray chưa cài — bỏ qua merge, giữ file tháng")
        return False

    try:
        print(f"  [merge] Đang gộp {len(monthly_files)} tháng → {yearly_path.name} ...")
        datasets = [xr.open_dataset(str(f)) for f in sorted(monthly_files)]

        # Dùng valid_time làm dim concat → cấu trúc sạch (valid_time, pl, lat, lon)
        # Tránh tạo artifact dim time=12 gây đọc dữ liệu chậm 12x
        processed = []
        for d in datasets:
            if "valid_time" in d.coords and "time" in d.dims:
                d = d.swap_dims({"time": "valid_time"}).drop_vars("time", errors="ignore")
            processed.append(d)

        ds = xr.concat(processed, dim="valid_time")
        enc = {v: {"zlib": True, "complevel": 4}
               for v in ds.data_vars}
        ds.to_netcdf(str(yearly_path), encoding=enc)
        ds.close()
        for d in processed:
            d.close()

        size_mb = _size_mb(yearly_path)
        print(f"  [merge] OK — {yearly_path.name} ({size_mb:.1f} MB)")

        for f in monthly_files:
            f.unlink()
        print(f"  [merge] Đã xóa {len(monthly_files)} file tháng")
        return True
    except Exception as e:
        print(f"  [merge] Lỗi: {e} — giữ nguyên file tháng")
        return False


def fix_era5_files(years: "list[int]"):
    """Fix các file năm đã merge sai (time=12 artifact) → cấu trúc sạch (valid_time, pl, lat, lon).
    Chạy 1 lần duy nhất. Sau đó ERA5 extraction nhanh hơn ~10x.
    """
    try:
        import xarray as xr
    except ImportError:
        print("[error] xarray chưa cài")
        return

    PRES_DIR.mkdir(parents=True, exist_ok=True)
    fixed = 0

    for year in years:
        path = PRES_DIR / f"era5_pressure_{year}.nc"
        if not path.exists():
            print(f"  [fix] {year}: không có file — bỏ qua")
            continue

        # Kiểm tra có artifact time dim không
        ds = xr.open_dataset(str(path))
        has_artifact = ("time" in ds.dims and "time" not in ds.coords
                        and "valid_time" in ds.coords)
        ds.close()

        if not has_artifact:
            print(f"  [fix] {year}: đã OK — bỏ qua")
            continue

        print(f"  [fix] {year}: đang fix (time=12 → valid_time)...", flush=True)
        tmp_path = path.with_suffix(".fix.nc")
        try:
            ds = xr.open_dataset(str(path))
            # Collapse time artifact bằng mean(skipna) rồi swap dim
            ds_fix = ds.mean(dim="time", skipna=True)
            # Khôi phục valid_time làm coordinate dimension
            if "valid_time" in ds.coords:
                vt = ds["valid_time"].values
                if vt.ndim > 1:
                    vt = vt[0]  # lấy hàng đầu nếu 2D
                ds_fix = ds_fix.assign_coords(valid_time=("valid_time", vt))
            enc = {v: {"zlib": True, "complevel": 4} for v in ds_fix.data_vars}
            ds_fix.to_netcdf(str(tmp_path), encoding=enc)
            ds_fix.close()
            ds.close()

            tmp_path.replace(path)
            size_mb = _size_mb(path)
            print(f"  [fix] {year}: OK — {path.name} ({size_mb:.1f} MB)", flush=True)
            fixed += 1
        except Exception as e:
            print(f"  [fix] {year}: lỗi — {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    print(f"\n[fix] Hoàn thành: {fixed}/{len(years)} file đã fix.")


def download_pressure_year(client, year: int) -> bool:
    """Tải ERA5 pressure levels cho 1 năm theo tháng, sau đó merge thành 1 file năm."""
    yearly_path = PRES_DIR / f"era5_pressure_{year}.nc"
    if yearly_path.exists():
        print(f"  [skip] {yearly_path.name} ({_size_mb(yearly_path):.1f} MB)")
        return True

    monthly_files = []
    ok = True

    for month in range(1, 13):
        output = PRES_DIR / f"era5_pressure_{year}_{month:02d}.nc"
        if output.exists():
            print(f"  [skip] {output.name} ({_size_mb(output):.1f} MB)")
            monthly_files.append(output)
            continue

        request = {
            "product_type": "reanalysis",
            "variable":       PRESSURE_VARS,
            "pressure_level": PRESSURE_LEVELS,
            "year":           [str(year)],
            "month":          [f"{month:02d}"],
            "day":            _all_days(),
            "time":           TIMES,
            "area":           AREA,
            "format":         "netcdf",
        }
        success = _download_with_retry(client, "reanalysis-era5-pressure-levels", request, output)
        if success:
            monthly_files.append(output)
        else:
            ok = False

    # Merge nếu đủ 12 tháng
    if len(monthly_files) == 12:
        _merge_monthly_to_yearly(monthly_files, yearly_path)
    elif monthly_files:
        print(f"  [warn] Chỉ có {len(monthly_files)}/12 tháng — chưa merge")

    return ok


def download_sst_year(client, year: int) -> bool:
    """Tải ERA5 SST cho 1 năm, tách từng tháng để tránh giới hạn CDS."""
    ok = True

    for month in range(1, 13):
        output = SST_DIR / f"era5_sst_{year}_{month:02d}.nc"
        if output.exists():
            print(f"  [skip] {output.name} ({_size_mb(output):.1f} MB)")
            continue

        request = {
            "product_type": "reanalysis",
            "variable":     SINGLE_VARS,
            "year":         [str(year)],
            "month":        [f"{month:02d}"],
            "day":          _all_days(),
            "time":         TIMES,
            "area":         AREA,
            "format":       "netcdf",
        }
        success = _download_with_retry(client, "reanalysis-era5-single-levels", request, output)
        if not success:
            ok = False

    return ok


# ─── Check mode ───────────────────────────────────────────────────────────────

def check_downloads(years: "list[int]"):
    """In trạng thái các file đã tải (ưu tiên file năm, fallback file tháng)."""
    print(f"\n{'─'*60}")
    print(f"  {'Year':<6} {'Pressure':>30}")
    print(f"  {'─'*50}")

    total_mb   = 0.0
    missing    = []

    for year in years:
        yearly = PRES_DIR / f"era5_pressure_{year}.nc"
        if yearly.exists():
            mb = _size_mb(yearly)
            pres_str = f"yearly ({mb:.1f} MB)"
            total_mb += mb
        else:
            monthly = sorted(PRES_DIR.glob(f"era5_pressure_{year}_*.nc"))
            mb = sum(_size_mb(f) for f in monthly)
            pres_str = f"{len(monthly):2d}/12 monthly ({mb:.1f} MB)" if monthly else "MISSING"
            total_mb += mb
            if not yearly.exists() and len(monthly) < 12:
                missing.append(str(year))

        print(f"  {year:<6} {pres_str:>30}")

    print(f"{'─'*60}")
    print(f"  Tổng: {total_mb:.1f} MB  ({total_mb/1024:.2f} GB)")
    if missing:
        print(f"  Còn thiếu: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
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
        "--merge", action="store_true",
        help="Chỉ merge file tháng thành file năm (không tải thêm)",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Fix file năm bị merge sai (time=12 artifact) → cấu trúc sạch. Chạy 1 lần.",
    )
    args = parser.parse_args()

    years = sorted(args.years)

    # Tạo thư mục output
    PRES_DIR.mkdir(parents=True, exist_ok=True)
    SST_DIR.mkdir(parents=True, exist_ok=True)

    if args.check:
        check_downloads(years)
        return

    if args.fix:
        fix_era5_files(years)
        return

    if args.merge:
        PRES_DIR.mkdir(parents=True, exist_ok=True)
        for year in years:
            yearly_path = PRES_DIR / f"era5_pressure_{year}.nc"
            if yearly_path.exists():
                print(f"  [skip] {yearly_path.name} đã tồn tại")
                continue
            monthly_files = sorted(PRES_DIR.glob(f"era5_pressure_{year}_*.nc"))
            if len(monthly_files) == 12:
                _merge_monthly_to_yearly(monthly_files, yearly_path)
            elif monthly_files:
                print(f"  [warn] {year}: chỉ có {len(monthly_files)}/12 tháng — bỏ qua merge")
            else:
                print(f"  [skip] {year}: không có file tháng nào")
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


    print(f"\n{'='*55}")
    print(f"  Hoàn thành: {ok_count} file OK, {fail_count} file lỗi")
    check_downloads(years)


if __name__ == "__main__":
    main()
