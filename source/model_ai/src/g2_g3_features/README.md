# G2/G3 — Feature Engineering & Sequences

**File:** `features.py`
**Input:** `data/processed/bao_bien_dong_clean.csv`
**Output:** `data/features/feature_matrix.csv`, `data/features/sequences.npz`, `models/scaler.pkl`
**Checkpoint:** ≥ 5,000 sequences ✅ PASS (27,055 sequences)

---

## Chạy

```bash
cd source/model_ai
python -m src.g2_g3_features.features
```

---

## 12 Features (G2)

| # | Tên | Công thức | Ý nghĩa vật lý |
|---|-----|-----------|----------------|
| 1 | `lat_norm` | (LAT − 8) / 14 | Vị trí vĩ độ [0,1] → lực Coriolis |
| 2 | `lon_norm` | (LON − 102) / 18 | Vị trí kinh độ [0,1] → khoảng cách đất liền |
| 3 | `dlat` | LAT[t] − LAT[t−1] | Quán tính bắc-nam ★★★ |
| 4 | `dlon` | LON[t] − LON[t−1] | Quán tính đông-tây ★★★ |
| 5 | `speed_kmh` | Haversine(t−1, t) / 6h | Tốc độ tịnh tiến (km/h) ★★★ |
| 6 | `direction` | atan2(dlat, dlon) → [0°,360°] | Hướng di chuyển (0°=Bắc) ★★★ |
| 7 | `vmax` | WMO_WIND → USA_WIND | Cường độ bão (kt) |
| 8 | `pmin` | WMO_PRES → USA_PRES | Áp suất tâm bão (hPa) |
| 9 | `sst_c` | Climatology theo tháng | SST > 26°C → bão mạnh lên |
| 10 | `month_sin` | sin(2π × month / 12) | Mã hóa tuần hoàn tháng |
| 11 | `month_cos` | cos(2π × month / 12) | Tháng 12 và tháng 1 gần nhau |
| 12 | `storm_age_h` | giờ kể từ điểm đầu tiên | Bão non vs bão già |

**Tại sao month_sin/cos thay vì số nguyên?**
Nếu dùng số nguyên: tháng 12 (=12) và tháng 1 (=1) cách xa 11 đơn vị. Nhưng thực tế chúng liên tiếp nhau. Sin/cos giải quyết bằng cách mã hóa tháng trên vòng tròn đơn vị.

---

## Sliding Window — Sequences (G3)

```
Mỗi sequence:
  X : [8, 12]   — 8 bước × 6h = 48h lịch sử, 12 features
  y : [4]       — [lat_24h, lon_24h, lat_48h, lon_48h]

Tổng: 27,055 sequences
  X_train : (22,567, 8, 12)  — năm 1980–2016
  X_val   : (2,557, 8, 12)   — năm 2017–2020
  X_test  : (1,931, 8, 12)   — năm 2021–2024
```

**Chia theo năm, không random** → tránh data leakage giữa các cơn bão cùng năm.

**Sliding window trong từng storm riêng biệt** → không ghép cuối storm này với đầu storm khác.

---

## Chuẩn hóa (G3)

`StandardScaler` fit **chỉ trên train** → transform val + test. Lưu `models/scaler.pkl`.

Lý do: nếu fit trên toàn bộ dữ liệu, mean/std sẽ bị "rò" thông tin từ val/test vào train → data leakage.

---

## API

```python
from src.g2_g3_features import build_features, make_sequences, split_and_scale

feat_df       = build_features(df_clean, cfg)
X, y, meta    = make_sequences(feat_df, cfg)
data          = split_and_scale(X, y, meta, cfg)

# Sau đó:
data["X_train"]  # (22567, 8, 12) — đã scale
data["y_train"]  # (22567, 4)     — lat/lon thực (chưa scale)
data["scaler"]   # StandardScaler đã fit
```
