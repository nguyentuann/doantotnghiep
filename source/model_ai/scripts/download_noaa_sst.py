"""
download_noaa_sst.py — Tải NOAA OISST v2.1 monthly mean từ PSL
----------------------------------------------------------------
Không cần đăng ký, không cần API key.
Tải 1 file NetCDF duy nhất (~20 MB) chứa SST monthly mean toàn cầu 1981–nay.

Chạy:
  cd source/model_ai
  python scripts/download_noaa_sst.py            # tải file
  python scripts/download_noaa_sst.py --check    # kiểm tra file đã tải

Output:
  data/noaa_sst/sst.mon.mean.nc  — SST monthly mean 1°, toàn cầu
"""

import argparse
import time
from pathlib import Path

try:
    import requests
except ImportError:
    raise SystemExit("[error] requests chưa cài. Chạy: pip install requests")


# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent          # source/model_ai/
OUT_DIR  = BASE_DIR / "data/noaa_sst"
OUT_FILE = OUT_DIR / "sst.mnmean.nc"

# Nguồn tải — bản 1° resolution, ~61 MB
URLS = [
    "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2/sst.mnmean.nc",
]

MAX_RETRIES = 3
RETRY_WAIT  = 30


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def download() -> bool:
    """Tải file SST monthly mean. Thử lần lượt các URL."""
    if OUT_FILE.exists():
        print(f"[skip] {OUT_FILE.name} đã tồn tại ({_size_mb(OUT_FILE):.1f} MB)")
        return True

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for url in URLS:
        host = url.split("/")[2]
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"  [attempt {attempt}/{MAX_RETRIES}] {host}")
                resp = requests.get(url, timeout=180, stream=True)
                resp.raise_for_status()

                total = int(resp.headers.get("content-length", 0))
                downloaded = 0

                with open(OUT_FILE, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=256 * 1024):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"\r  Downloading... {downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB ({pct:.0f}%)", end="", flush=True)

                print(f"\n  [ok] {OUT_FILE.name}  ({_size_mb(OUT_FILE):.1f} MB)")
                return True

            except Exception as e:
                if OUT_FILE.exists():
                    OUT_FILE.unlink()
                print(f"\n  [error] {e}")
                if attempt < MAX_RETRIES:
                    print(f"  Chờ {RETRY_WAIT}s rồi thử lại...")
                    time.sleep(RETRY_WAIT)

        print(f"  [skip] {host} không kết nối được, thử URL tiếp...")

    print("[fail] Tất cả URL đều lỗi.")
    return False


def check():
    """Kiểm tra file và in thông tin."""
    print(f"\n{'─'*50}")
    if not OUT_FILE.exists():
        print(f"  File: MISSING")
        print(f"  Chạy: python scripts/download_noaa_sst.py")
    else:
        mb = _size_mb(OUT_FILE)
        print(f"  File: {OUT_FILE}")
        print(f"  Size: {mb:.1f} MB")

        try:
            import xarray as xr
            ds = xr.open_dataset(OUT_FILE)
            t_min = str(ds.time.values[0])[:10]
            t_max = str(ds.time.values[-1])[:10]
            lat_shape = ds.dims.get("lat", ds.dims.get("latitude", "?"))
            lon_shape = ds.dims.get("lon", ds.dims.get("longitude", "?"))
            print(f"  Time: {t_min} → {t_max}  ({len(ds.time)} months)")
            print(f"  Grid: {lat_shape} × {lon_shape}")
            ds.close()
        except ImportError:
            print("  (cài xarray để xem chi tiết: pip install xarray netcdf4)")
        except Exception as e:
            print(f"  [warn] Không đọc được file: {e}")

    print(f"{'─'*50}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Tải NOAA OISST v2.1 monthly mean")
    parser.add_argument("--check", action="store_true", help="Kiểm tra file đã tải")
    args = parser.parse_args()

    if args.check:
        check()
    else:
        ok = download()
        if ok:
            check()


if __name__ == "__main__":
    main()
