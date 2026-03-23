# DỰ ÁN: DỰ ĐOÁN ĐƯỜNG ĐI BÃO BIỂN ĐÔNG — TỔNG QUAN HỆ THỐNG

> **Agent README** — Đọc file này trước khi thực hiện bất kỳ tác vụ nào trong dự án.
>
> Cập nhật: 2025 | Trường ĐHBK Đà Nẵng — Khoa CNTT

---

## 1. MỤC TIÊU DỰ ÁN

Xây dựng mô hình học sâu (Deep Learning) dự đoán đường đi của bão nhiệt đới trên vùng  **Biển Đông Việt Nam** , đạt:

| Chỉ số              | Mục tiêu tối thiểu | Mục tiêu chính |
| --------------------- | ---------------------- | ----------------- |
| MAE 24h               | < 200 km               | < 150 km          |
| MAE 48h               | < 350 km               | < 250 km          |
| Skill Score vs CLIPER | > 20%                  | > 30%             |

**Phạm vi địa lý:** 8°–22°N, 102°–120°E (Biển Đông Việt Nam)

**Phạm vi thời gian dữ liệu:** 1980–2024

**Horizon dự báo:** 24h và 48h (bước 6h)

**Lookback window:** 48h (8 bước × 6h)

**Bão đặc biệt (~5%):** Phát hiện riêng, báo cáo cone of uncertainty — không tính vào Skill Score tổng.

---

## 2. CẤU TRÚC THƯ MỤC

```
typhoon_prediction/
├── PROJECT_OVERVIEW.md          ← File này — đọc trước
├── config.yaml                  ← TẤT CẢ hyperparameter — không hardcode
├── data/
│   ├── raw/                     ← Dữ liệu gốc, KHÔNG sửa
│   │   ├── ibtracs_WP.csv       ← IBTrACS v04r01 West Pacific
│   │   ├── sst_YYYY.nc          ← NOAA OI SST theo từng năm
│   │   └── era5_steering.nc     ← ERA5 gió 500hPa (nếu có)
│   ├── processed/
│   │   └── bao_bien_dong_clean.csv  ← Sau G2: track đã làm sạch 6h
│   └── features/
│       └── feature_matrix.csv   ← Sau G3: 11 features + targets
├── models/
│   ├── checkpoints/             ← .pt lưu theo epoch
│   │   └── best.pt              ← Best validation checkpoint
│   ├── final/
│   │   └── model_best.pt        ← Model tốt nhất cuối cùng
│   └── scaler.pkl               ← StandardScaler — fit trên train only
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_features.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── data_loader.py           ← Tải, lọc, làm sạch IBTrACS
│   ├── features.py              ← Tính features, sliding window
│   ├── models.py                ← LSTM, BiLSTM, Transformer
│   ├── train.py                 ← Training loop, HaversineLoss
│   └── evaluate.py              ← MAE, RMSE, Skill Score, CLIPER
└── results/
    ├── results_log.json         ← Log TẤT CẢ experiments
    └── figures/                 ← PNG biểu đồ, HTML bản đồ Folium
```

---

## 3. CONFIG.YAML — NGUỒN SỰ THẬT DUY NHẤT

```yaml
geography:
  lat_min: 8.0    lat_max: 22.0
  lon_min: 102.0  lon_max: 120.0
  year_start: 1980  year_end: 2024

data:
  ibtracs_file: 'data/raw/ibtracs_WP.csv'
  clean_file:   'data/processed/bao_bien_dong_clean.csv'
  feature_file: 'data/features/feature_matrix.csv'
  time_step_hours: 6

model:
  lookback: 8      # 8 bước × 6h = 48h nhìn về quá khứ
  horizon:  4      # 4 bước × 6h = 24h dự đoán tới
  hidden_size: 128
  num_layers: 2
  dropout: 0.3
  batch_size: 32
  lr: 0.001
  max_epochs: 150
  patience: 20     # early stopping

split:             # Chia theo NĂM — không random
  train: [1980, 2016]
  val:   [2017, 2020]
  test:  [2021, 2024]
```

**AGENT RULE:** Không được hardcode số vào code. Luôn đọc từ `config.yaml`.

---

## 4. PIPELINE 5 GIAI ĐOẠN

```
[G1] IBTrACS raw CSV
       ↓ load_and_filter_scs()  — lọc SCS box, lấy FULL TRACK của storm
[G2] bao_bien_dong_clean.csv
       ↓ build_features()       — tính 11 features vật lý
[G3] feature_matrix.csv
       ↓ make_sequences()       — sliding window [N, 8, 11] → y [N, 4]
[G4] X_train, X_val, X_test     — normalize với StandardScaler
       ↓ train_model()          — LSTM → BiLSTM → Transformer
[G5] model_best.pt
       ↓ evaluate_model()       — MAE, Skill Score, Folium maps
```

### Lưu ý quan trọng về G1 (KHÔNG LÀM SAI):

```python
# ĐÚNG: Lấy TOÀN BỘ track của storm có đi qua SCS
storm_ids_in_scs = df[mask_scs]['SID'].unique()
df_full = df[df['SID'].isin(storm_ids_in_scs)]  # kể cả đoạn ngoài SCS
df_full['in_scs'] = mask_scs.reindex(df_full.index, fill_value=False)

# SAI: Chỉ lấy đoạn trong SCS (mất ngữ cảnh lookback)
# df_scs = df[mask_scs]  ← KHÔNG làm thế này
```

---

## 5. 11 FEATURES — CƠ SỞ VẬT LÝ

| #  | Tên              | Công thức                  | Lý do                                           |
| -- | ----------------- | ---------------------------- | ------------------------------------------------ |
| 1  | `lat_norm`      | (LAT-8)/(22-8)               | Vị trí → lực Coriolis, xu hướng quặt bắc |
| 2  | `lon_norm`      | (LON-102)/(120-102)          | Vị trí → khoảng cách đất liền, WPSH      |
| 3  | `dlat`          | LAT[t]-LAT[t-1]              | Quán tính bắc-nam ★★★                      |
| 4  | `dlon`          | LON[t]-LON[t-1]              | Quán tính đông-tây ★★★                   |
| 5  | `speed_kmh`     | Haversine(t-1,t)/6h          | Tốc độ tịnh tiến ★★★                     |
| 6  | `direction`     | atan2(dlat,dlon)×180/π     | Hướng di chuyển ★★★                        |
| 7  | `vmax`          | WMO_WIND (fallback USA_WIND) | Cường độ bão                                |
| 8  | `pmin`          | WMO_PRES (fallback USA_PRES) | Áp suất tâm                                   |
| 9  | `sst_c`         | NOAA OI SST tại (lat,lon)   | SST>26°C → bão mạnh lên                     |
| 10 | `month_sin/cos` | sin/cos(2π×tháng/12)      | Tính mùa vụ (vòng tròn)                     |
| 11 | `storm_age_h`   | giờ kể từ hình thành    | Bão non vs bão già                            |

**Chuẩn hóa:** StandardScaler fit trên train → transform val+test. Lưu `scaler.pkl`.

**Target:** `y = [lat_t4, lon_t4, lat_t8, lon_t8]` (vị trí sau 24h và 48h)

---

## 6. KIẾN TRÚC 3 MÔ HÌNH (thứ tự train)

### Model 1: LSTM Baseline

```
Input [B,8,11] → LSTM(128, 2 lớp, dropout=0.3) → LayerNorm
→ Linear(128→64) → ReLU → Dropout → Linear(64→4) → Output [B,4]
```

### Model 2: BiLSTM + Self-Attention

```
Input [B,8,11] → BiLSTM(128, bidirectional) → Attention(softmax)
→ Context vector → LayerNorm → MLP(256→128→4) → Output [B,4]
```

### Model 3: Temporal Transformer

```
Input [B,8,11] → Linear(11→64) + PosEncoding
→ TransformerEncoder(d=64, heads=4, layers=2) → [:,-1,:]
→ MLP(64→32→4) → Output [B,4]
```

### Loss Function: HaversineLoss

```python
Loss = 0.6 × Haversine(pred_24h, true_24h) + 0.4 × Haversine(pred_48h, true_48h)
# Đơn vị: km. Ưu tiên 24h vì quan trọng hơn cho cảnh báo thiên tai.
# KHÔNG dùng MSE vì 1° kinh độ ≠ km nhất quán theo vĩ độ.
```

---

## 7. CHECKPOINTS — AGENT PHẢI PASS TRƯỚC KHI TIẾP TỤC

| Checkpoint        | Điều kiện pass                 | Hành động nếu fail                 |
| ----------------- | --------------------------------- | -------------------------------------- |
| **G1**      | ≥300 storms, 0 NaN lat/lon       | Kiểm tra lại filter, URL download    |
| **G2**      | feature_matrix có 11 cột, 0 NaN | Debug từng bước feature engineering |
| **G3**      | X.shape == (N, 8, 11), N > 5000   | Kiểm tra sliding window logic         |
| **G4**      | scaler.pkl tồn tại              | Chạy lại normalize()                 |
| **G5-LSTM** | Val MAE < 200 km                  | Giảm lr, tăng dropout, debug loss    |
| **G6-Best** | Cải thiện ≥10% vs LSTM         | Thử BiLSTM trước Transformer        |
| **G7**      | Skill Score > 20%                 | Kiểm tra CLIPER baseline code         |

---

## 8. NGUỒN DỮ LIỆU

| Dataset           | URL tải                                                                       | Ghi chú                       |
| ----------------- | ------------------------------------------------------------------------------ | ------------------------------ |
| IBTrACS v04r01 WP | `ncei.noaa.gov/data/ibtracs/.../ibtracs.WP.list.v04r01.csv`                  | Không cần đăng ký, ~15MB  |
| NOAA OI SST       | `downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.YYYY.nc` | ~80MB/năm, vòng lặp         |
| ERA5 (tùy chọn) | `cds.climate.copernicus.eu`qua `cdsapi`                                    | Cần đăng ký CDS miễn phí |
| ENSO Niño 3.4    | `psl.noaa.gov/data/correlation/nina34.data`                                  | Bổ sung, <1MB                 |

### Quy ước cột IBTrACS quan trọng:

* `WMO_WIND` = gió 10 phút (JMA, chuẩn WMO) — ưu tiên 1
* `USA_WIND` = gió 1 phút (JTWC, cao hơn ~12%) — fallback
* `ISO_TIME` = UTC (≠ giờ VN, VN = UTC+7)
* `TRACK_TYPE` = chỉ giữ `'main'`, bỏ `'spur'`
* Hàng 2 của CSV = units row → dùng `skiprows=[1]`

---

## 9. ĐÁNH GIÁ

### Baseline bắt buộc (tính trước model):

```python
# CLIPER = persistence: ngoại suy tuyến tính từ 2 bước gần nhất
dlat = LAT[t] - LAT[t-1]
dlon = LON[t] - LON[t-1]
pred_lat_24 = LAT[t] + 4*dlat   # 4 bước × 6h = 24h
pred_lon_24 = LON[t] + 4*dlon
```

### Skill Score:

```
Skill = (MAE_CLIPER - MAE_Model) / MAE_CLIPER × 100%
Skill > 0%  → model tốt hơn CLIPER
Skill < 0%  → model tệ hơn CLIPER (vấn đề nghiêm trọng)
```

### Phân loại quỹ đạo (để phân tích riêng):

```python
def classify_track(group):
    curvature = max_deviation_from_straight_line(group['LAT'], group['LON'])
    total_dlat = LAT[-1] - LAT[0]
    if curvature > 3.0:   return 'erratic'    # ~5%  — báo cáo riêng
    if total_dlat > 3.0:  return 'recurving'  # ~20% — khó hơn
    if total_dlat < -2.0: return 'southward'  # ~15%
    return 'westward'                          # ~65% — phổ biến nhất
```

---

## 10. VISUALIZATION

### Thứ tự ưu tiên:

1. **Matplotlib/Seaborn** — 6 biểu đồ báo cáo (learning curve, scatter, bar MAE...)
2. **Folium** — Bản đồ HTML tương tác track dự đoán vs thực tế
3. **Streamlit** (tuần 8 nếu còn thời gian) — Web app demo

### 6 biểu đồ bắt buộc trong báo cáo:

```
figures/learning_curve.png      — train/val loss theo epoch
figures/mae_by_type.png         — MAE theo loại quỹ đạo
figures/scatter_pred_actual.png — dự đoán vs thực tế
figures/mae_vs_horizon.png      — MAE tăng theo horizon 6h→48h
figures/model_comparison.png    — so sánh 3 model + CLIPER
figures/track_XXXXXXX.html      — bản đồ Folium (≥3 cơn bão)
```

---

## 11. MÔI TRƯỜNG THỰC THI

```
Platform:   Google Colab (GPU T4, 16GB VRAM)
Python:     3.10+
PyTorch:    2.0+ (CUDA)
pandas:     2.0+  |  numpy: 1.24+
xarray:     2023+ |  netCDF4: 1.6+
scikit-learn: 1.3+
folium: 0.14+ | plotly: 5.0+
matplotlib: 3.7+ | seaborn: 0.12+
```

### Setup cell đầu tiên mỗi notebook:

```python
!pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
!pip install pandas numpy xarray netCDF4 folium plotly scikit-learn tqdm -q
import torch
assert torch.cuda.is_available(), "PHẢI BẬT GPU: Runtime > Change runtime type > GPU"
```

---

## 12. NGUYÊN TẮC AGENT

| # | Nguyên tắc                    | Chi tiết                                                     |
| - | ------------------------------- | ------------------------------------------------------------- |
| 1 | **Config trước**        | Đọc `config.yaml`trước khi viết bất kỳ số nào      |
| 2 | **Checkpoint bắt buộc** | Không chuyển giai đoạn nếu checkpoint chưa pass         |
| 3 | **Fail fast**             | Sau 3 lần thử thất bại → dừng và báo lỗi chi tiết   |
| 4 | **Log tất cả**          | Ghi kết quả vào `results/results_log.json`với timestamp |
| 5 | **Full track**            | Luôn lấy toàn bộ track của storm (kể cả ngoài SCS)    |
| 6 | **No data leakage**       | Scaler chỉ fit trên train, split theo năm không random    |
| 7 | **Haversine**             | Dùng Haversine loss và MAE, không dùng MSE cho tọa độ  |
| 8 | **CLIPER trước**        | Tính baseline CLIPER trước khi train bất kỳ model nào   |

---

## 13. TIẾN ĐỘ 8 TUẦN

| Tuần | Giai đoạn                      | Đầu ra                    | Checkpoint         |
| ----- | -------------------------------- | --------------------------- | ------------------ |
| 1     | Thu thập & làm sạch dữ liệu | `bao_bien_dong_clean.csv` | ≥300 storms       |
| 2     | EDA + Feature Engineering        | `feature_matrix.csv`      | 11 features, 0 NaN |
| 3     | Sliding window + setup           | Tensors X/y split           | X.shape đúng     |
| 4     | Train LSTM Baseline              | `model_lstm.pt`           | Val MAE <200 km    |
| 5     | BiLSTM + Transformer             | `model_best.pt`           | +10% vs LSTM       |
| 6     | Đánh giá toàn diện          | `eval_report.json`        | Skill Score >30%   |
| 7     | Visualization                    | HTML maps + slide           | Demo end-to-end    |
| 8     | Báo cáo + Bảo vệ             | Báo cáo 5 chương        | GVHD duyệt        |

---

## 14. FILE ĐÃ TẠO TRONG DỰ ÁN

| File                                 | Nội dung                                                        |
| ------------------------------------ | ---------------------------------------------------------------- |
| `agent_master_plan_bao.docx`       | Hướng dẫn kỹ thuật chi tiết + code mẫu từng giai đoạn  |
| `de_cuong_datn_bao.docx`           | Đề cương chính thức nộp trường ĐHBK Đà Nẵng         |
| `pham_vi_phuong_phap_chitiet.docx` | Mục 3 & 4 đề cương — phạm vi và phương pháp chi tiết |
| `dataset_bao_bien_dong.xlsx`       | Dataset manager 6 sheet: raw → clean → features → results     |
| `link_data.xlsx`                   | 10 link tải dataset có thể click, chia 3 nhóm ưu tiên      |
| `giai_thich_cot_du_lieu.docx`      | Giải thích 160+ cột IBTrACS + ERA5 + SST + ENSO               |
| `PROJECT_OVERVIEW.md`              | File này — tổng quan hệ thống cho agent                     |

---

*Cập nhật lần cuối: 2025-03-22 | Đà Nẵng, Việt Nam*
