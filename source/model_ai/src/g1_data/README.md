# G1 — Thu thập & Làm sạch dữ liệu

**File:** `data_loader.py`
**Input:** `data/raw/ibtracs_WP.csv` (IBTrACS v4r01, ~109 MB)
**Output:** `data/processed/bao_bien_dong_clean.csv` (16.4 MB)
**Checkpoint:** ≥ 300 storms, 0 NaN lat/lon ✅ PASS (561 storms, 35,432 rows)

---

## Chạy

```bash
cd source/model_ai
python -m src.g1_data.data_loader
```

---

## Quy trình xử lý

```
IBTrACS CSV (246,679 rows)
  → bỏ units row (skiprows=[1])
  → lọc TRACK_TYPE == 'main'
  → lọc năm 1980–2024
  → tìm SID có ít nhất 1 điểm trong SCS box (8–22°N, 102–120°E)
  → lấy TOÀN BỘ track của SID đó (kể cả đoạn ngoài SCS)
  → tạo cột vmax (WMO_WIND → USA_WIND) và pmin (WMO_PRES → USA_PRES)
  → lưu bao_bien_dong_clean.csv
```

## Quyết định thiết kế quan trọng

**Tại sao lấy full track (kể cả ngoài SCS)?**
Lookback window 48h (8 bước) cần có đủ lịch sử. Nếu bão vừa vào SCS, các bước lookback sẽ là đoạn ngoài SCS. Cắt bỏ đoạn ngoài → mất ngữ cảnh → dự đoán sai.

```python
# ĐÚNG
storm_ids_in_scs = df[mask_scs]["SID"].unique()
df_full = df[df["SID"].isin(storm_ids_in_scs)]  # full track

# SAI — mất lookback history
df_scs = df[mask_scs]
```

**Tại sao skiprows=[1]?**
Hàng 2 trong IBTrACS CSV là units row (ví dụ: "deg", "kt", "mb") — không phải dữ liệu.

**WMO_WIND vs USA_WIND?**
WMO_WIND là gió 10 phút (chuẩn quốc tế). USA_WIND là gió 1 phút (JTWC, cao hơn ~12%). Ưu tiên WMO_WIND để nhất quán với tiêu chuẩn WMO.

---

## API

```python
from src.g1_data import load_and_filter_scs, save_clean_data, load_clean_data

cfg = load_config()
df  = load_and_filter_scs(config=cfg)   # lọc SCS
save_clean_data(df, config=cfg)          # lưu CSV
df  = load_clean_data(config=cfg)        # đọc lại CSV đã lưu
```
