# Tiến độ triển khai — Hệ thống dự đoán đường đi bão Biển Đông

Cập nhật lần cuối: 2026-03-22

---

## Tổng quan pipeline

```
data_raw.csv → [G1] lọc SCS → [G2] features → [G3] sequences
             → [G4] models → [G5] train LSTM → [G6] train BiLSTM/Transformer
             → [G7] evaluate → [G8] export ONNX
             → [Backend] FastAPI → [Frontend] Leaflet map
```

---

## Chi tiết tiến độ

### ✅ G1 — Data Loader (`src/data_loader.py`)
- **Kết quả:** 561 storms đi qua SCS (1980–2024), 35,432 rows full track
- **Output:** `data/processed/bao_bien_dong_clean.csv` (16.4 MB)
- **Checkpoint:** PASS (>= 300 storms, 0 NaN lat/lon)

### ✅ G2 — Feature Engineering (`src/features.py`)
- **Features (12):** lat_norm, lon_norm, dlat, dlon, speed_kmh, direction, vmax, pmin, sst_c, month_sin, month_cos, storm_age_h
- **SST:** Climatology theo tháng (chưa có file NetCDF)
- **Output:** `data/features/feature_matrix.csv` (7.4 MB)
- **Checkpoint:** PASS (0 NaN sau fill median)

### ✅ G3 — Sequences & Scaler (`src/features.py`)
- **Sequences:** X(27,055 × 8 × 12), y(27,055 × 4)
- **Split theo năm:** Train 22,567 | Val 2,557 | Test 1,931
- **Output:** `data/features/sequences.npz`, `models/scaler.pkl`
- **Checkpoint:** PASS (>= 5,000 sequences)

### ⬜ G4 — Model Architecture (`src/models.py`)
- [ ] LSTM baseline (2 layer, hidden=128)
- [ ] BiLSTM + Attention
- [ ] Temporal Transformer
- [ ] HaversineLoss

### ⬜ G5 — Train LSTM (`src/train.py`)
- [ ] Training loop, early stopping (patience=20)
- [ ] Checkpoint tốt nhất theo val loss
- [ ] Checkpoint: val MAE < 200 km

### ⬜ G6 — Train BiLSTM & Transformer
- [ ] Cải thiện >= 10% so với LSTM baseline
- [ ] Lưu `models/final/model_best.pt`

### ⬜ G7 — Evaluation (`src/evaluate.py`)
- [ ] CLIPER baseline
- [ ] MAE 24h / 48h (km) cho từng model
- [ ] Skill Score (%) so với CLIPER
- [ ] Sinh 6 biểu đồ thesis (lưu vào `results/figures/`)
- [ ] Checkpoint: Skill Score >= 20%

### ⬜ G8 — Export ONNX (`src/export_onnx.py`)
- [ ] `torch.onnx.export()` model tốt nhất
- [ ] Verify output ONNX == PyTorch (tolerance 1e-5)
- [ ] Output: `models/final/model_best.onnx`

---

## Backend (`source/backend/`)

### ⬜ B1 — Scaffold & Schemas
- [ ] FastAPI project structure
- [ ] Pydantic request/response schemas

### ⬜ B2 — Inference Services
- [ ] `services/preprocessor.py` — load scaler.pkl, tính features
- [ ] `services/predictor.py` — ONNX Runtime inference
- [ ] `services/track_builder.py` — interpolate 6h steps, build cone polygon

### ⬜ B3 — API Endpoints
- [ ] `POST /predict`
- [ ] `GET /storms/historical`
- [ ] `POST /storms/historical/{id}/replay`
- [ ] `GET /models`
- [ ] `GET /health`

### ⬜ B4 — Historical Data
- [ ] Tiền xử lý test set (2021–2024) → `historical_tracks.json`

### ⬜ B5 — Deploy
- [ ] CORS, uvicorn config
- [ ] Deploy lên Render.com (free tier)

---

## Frontend (`source/frontend/`)

### ⬜ F1 — Map base
- [ ] Leaflet.js + CartoDB Dark tiles
- [ ] Vietnam coastline GeoJSON overlay
- [ ] SCS bounding box

### ⬜ F2 — Track visualization
- [ ] Past track (màu theo Saffir-Simpson)
- [ ] Predicted track (nét đứt)
- [ ] Cone of uncertainty (turf.js buffer)

### ⬜ F3 — Timeline & Controls
- [ ] Slider 6h steps + Play/Pause
- [ ] Storm selector dropdown

### ⬜ F4 — Side panel
- [ ] Storm info, category badge
- [ ] Model selector (LSTM/BiLSTM/Transformer)
- [ ] MAE/Skill Score table
- [ ] Chart.js intensity timeline

### ⬜ F5 — Connect API & Deploy
- [ ] Kết nối với backend thật
- [ ] Deploy GitHub Pages

---

## Files quan trọng

| File | Mô tả |
|------|-------|
| `config.yaml` | Nguồn sự thật duy nhất cho hyperparameter |
| `src/data_loader.py` | G1: lọc IBTrACS → SCS |
| `src/features.py` | G2/G3: feature engineering + sequences |
| `src/models.py` | G4: định nghĩa LSTM, BiLSTM, Transformer |
| `src/train.py` | G5/G6: training loop |
| `src/evaluate.py` | G7: metrics + biểu đồ |
| `src/export_onnx.py` | G8: export model |
| `data/processed/bao_bien_dong_clean.csv` | Dữ liệu đã lọc |
| `data/features/sequences.npz` | Sequences sẵn sàng cho training |
| `models/scaler.pkl` | StandardScaler (fit trên train) |

---

## Ghi chú

- SST hiện dùng **climatology** theo tháng, chưa có file NetCDF thực tế.
  Nếu bổ sung SST thực → cập nhật `sst_climatology()` trong `features.py`.
- Hàm 2 trong IBTrACS CSV là units row → luôn dùng `skiprows=[1]`.
- Split **theo năm** (không random) để tránh data leakage.
- Model dự đoán **trực tiếp** lat/lon tại 24h và 48h; các bước 6h–42h được **nội suy tuyến tính**.
