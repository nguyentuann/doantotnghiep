# Dự báo quỹ đạo bão Biển Đông bằng Deep Learning

**Luận văn tốt nghiệp** — Ứng dụng các mô hình học sâu để dự báo quỹ đạo bão và áp thấp nhiệt đới khu vực Biển Đông (South China Sea), tích hợp REST API và giao diện trực quan 3D.

---

## Mục tiêu

Bão nhiệt đới là một trong những thiên tai nguy hiểm nhất tại khu vực Biển Đông, ảnh hưởng trực tiếp đến Việt Nam và các nước Đông Nam Á. Việc dự báo chính xác quỹ đạo bão trong 24–48 giờ tới đóng vai trò quan trọng trong công tác phòng chống thiên tai.

Đề tài này xây dựng hệ thống dự báo quỹ đạo bão dựa trên **học sâu (Deep Learning)** kết hợp dữ liệu quan trắc lịch sử và các đặc trưng khí tượng môi trường, với các mục tiêu cụ thể:

- MAE 24h < 150 km
- Skill Score > 20% so với baseline thống kê CLIPER

---

## Kết quả đạt được

Mô hình champion **Transformer (`scs_v12_lb6`)** vượt cả hai mục tiêu đề ra:

| Model                 | MAE 24h            | Skill 24h          | MAE 48h            | Skill 48h          | Tham số        |
| --------------------- | ------------------ | ------------------ | ------------------ | ------------------ | --------------- |
| LSTM                  | 134.0 km           | 23.2%              | 362.2 km           | 16.1%              | ~213k           |
| BiLSTM                | 134.2 km           | 23.1%              | 359.3 km           | 16.7%              | ~575k           |
| BiGRU                 | 118.9 km           | 31.9%              | 337.3 km           | 21.8%              | ~432k           |
| **Transformer** | **101.3 km** | **42.0%** ✅ | **249.1 km** | **42.3%** ✅ | **~110k** |

> **Baseline CLIPER MAE 24h = 174.6 km** (tập test SCS 2021–2024, 279 sequences).
> Transformer đạt Skill Score **42%** — gấp đôi mục tiêu đề ra, với số tham số ít nhất trong 4 mô hình.

---

## Dữ liệu

### Nguồn dữ liệu

| Nguồn                                                                           | Mô tả                                                                              | Tổ chức |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | --------- |
| [IBTrACS WP](https://www.ncei.noaa.gov/products/international-best-track-archive) | Quỹ đạo bão Tây Thái Bình Dương 1945–2024, tần suất 6h                   | NOAA/NCEI |
| [ERA5](https://cds.climate.copernicus.eu)                                         | Trường gió (u/v), địa thế vị, xoáy tại 4 mực áp suất 200/500/700/850 hPa | ECMWF     |
| [NOAA OISST](https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html)           | Nhiệt độ mặt biển tháng, độ phân giải 1°×1°                             | NOAA      |

### Phạm vi và phân chia

- **Khu vực:** Biển Đông — 8–22°N, 102–120°E
- **Tổng số bão SCS:** 561 cơn (sau lọc địa lý từ IBTrACS WP)

| Tập       | Năm       | Số sequences |
| ---------- | ---------- | ------------- |
| Train      | 1980–2016 | 7,880         |
| Validation | 2017–2020 | 829           |
| Test       | 2021–2024 | 660           |

> Phân chia theo thời gian (temporal split) để tránh data leakage — mô hình không "nhìn thấy" tương lai khi học.

File dữ liệu đã lọc SCS đính kèm tại `source/model_ai/data/processed/bao_bien_dong_1960.csv`.

---

## Đặc trưng đầu vào (31 features)

| Nhóm                         | Đặc trưng                                                                                 | Nguồn                                     |
| ----------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------ |
| Trạng thái bão (13)        | lat/lon, Δlat/Δlon, speed, direction, vmax, pmin, SST, month_sin/cos, storm_age, dist2land | IBTrACS + NOAA SST                         |
| Môi trường trung tầng (6) | wind_shear, u500, v500, u700, v700, z500                                                     | ERA5 hộp ±10°                           |
| Môi trường trên/thấp (6) | u200, v200, u850, v850, vort500, vort850                                                     | ERA5 hộp ±10°                           |
| Beta drift vật lý (2)       | beta_lat_drift, beta_lon_drift                                                               | Holland (1983)                             |
| Annulus DLM (2)               | asteer_u, asteer_v                                                                           | ERA5 vành 3–7°, trung bình 500+700 hPa |
| Annulus 850 hPa (2)           | asteer_u850, asteer_v850                                                                     | ERA5 vành 3–7°, 850 hPa                 |

**Đặc trưng quan trọng nhất** (đo bằng permutation importance):

| Hạng | Feature                | MAE tăng khi xóa |
| ----- | ---------------------- | ------------------ |
| 1     | lon_norm               | +319.8 km          |
| 2     | lat_norm               | +305.3 km          |
| 3     | asteer_u (annulus DLM) | +59.5 km           |
| 4     | dlon                   | +57.3 km           |
| 5     | asteer_v (annulus DLM) | +27.4 km           |

> Hai đặc trưng annulus vành đai (asteer_u/v) đứng thứ 3–5 về tầm quan trọng, đóng góp +4.7% Skill Score khi thêm vào.

---

## Kiến trúc các mô hình

### 1. LSTM

```
Input [B, 6, 31] → LSTM(128, 2 layers) → LayerNorm → Linear(128→64) → ReLU → Dropout → Linear(64→16)
```

Học phụ thuộc dài hạn qua cơ chế cổng (forget/input/output gate) và cell state.

### 2. BiLSTM + Attention

```
Input [B, 6, 31] → BiLSTM(128×2=256) → Additive Attention → context [B,256] → LayerNorm → Linear → Output
```

Đọc chuỗi theo cả 2 chiều, attention tự động trọng số các bước thời gian quan trọng.

### 3. BiGRU + Attention

```
Input [B, 6, 31] → BiGRU(128×2=256) → Additive Attention → LayerNorm → Linear → Output
```

Tương tự BiLSTM nhưng đơn giản hơn (không có cell state riêng).

### 4. Transformer (Champion)

```
Input [B, 6, 31] → Linear(31→64) → Learnable PE → TransformerEncoder(Pre-LN, 4 heads) → Last-token → Head → Output
```

4 tinh chỉnh so với Transformer chuẩn:

- **Pre-LN** (norm_first=True): LayerNorm trước attention → huấn luyện ổn định hơn
- **Learnable Positional Encoding**: học vị trí theo dữ liệu thay vì dùng hàm sin/cos cố định
- **Last-token pooling**: lấy token cuối cùng (t=6) làm context — chứa thông tin tích lũy toàn chuỗi
- **Bias initialization**: khởi tạo bias head về tâm Biển Đông (15°N, 115°E) để hội tụ nhanh hơn

---

## Chiến lược huấn luyện

### Hàm loss — MultiHorizonLoss

```
Loss = 0.5 × Haversine(24h) + 0.2 × Haversine(48h) + 1.0 × L_smooth + 0.1 × L_dir
```

- `L_smooth`: phạt đạo hàm bậc 2 của quỹ đạo (track mượt)
- `L_dir`: phạt sự thay đổi hướng đột ngột

### 4 cơ chế huấn luyện

| Cơ chế                                    | Mô tả                                                                       |
| ------------------------------------------- | ----------------------------------------------------------------------------- |
| **Residual learning**                 | Model học độ lệch (Δ) so với dự báo CLIPER thay vì học tuyệt đối |
| **Multi-seed ensemble**               | Train 3 seed (42, 123, 2024) → trung bình dự báo để giảm phương sai  |
| **SWA** (Stochastic Weight Averaging) | Trung bình trọng số từ epoch 225+ → tổng quát hóa tốt hơn           |
| **Early stopping**                    | Dừng khi val_loss không cải thiện sau 40 epochs                           |

---

## Tiến trình cải tiến

| Phiên bản               | Thay đổi chính               | Skill 24h       | Δ    |
| ------------------------- | ------------------------------- | --------------- | ----- |
| `scs_v7`                | Baseline SCS, 25 features       | 33.9%           | —    |
| `scs_v9`                | + Beta drift + SWA + Multi-seed | 34.5%           | +0.6% |
| `scs_v9_lb6`            | Lookback 8→6                   | 36.4%           | +1.9% |
| `scs_v11_lb6`           | + Annulus DLM 500+700 hPa       | 41.1%           | +4.7% |
| **`scs_v12_lb6`** | **+ Annulus 850 hPa**     | **42.0%** | +0.9% |

### Thí nghiệm thất bại (negative results)

| Tag              | Giả thuyết                   | Kết quả                |
| ---------------- | ------------------------------ | ------------------------ |
| `scs_finetune` | Pre-train WP → fine-tune SCS  | -11% (negative transfer) |
| `scs_v10`      | Mở rộng vùng train 5–25°N | -15% (domain quá rộng) |
| `scs_lb12`     | Lookback dài L=12             | -7%                      |
| `scs_v13_lb6`  | + Annulus 200 hPa              | -4.5% (redundant)        |

> **3 nguyên tắc rút ra:** (1) Domain tập trung > dữ liệu nhiều; (2) Prior vật lý > độ phức tạp; (3) Ít feature tốt hơn nhiều feature.

---

## Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                     React Frontend                       │
│         Globe 3D (react-globe.gl) + Dự báo real-time   │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP
┌──────────────────────▼──────────────────────────────────┐
│                   FastAPI Backend                        │
│   /predict  →  preprocessor → ONNX Runtime → response  │
│   /storms   →  historical tracks (IBTrACS)              │
└──────────────────────┬──────────────────────────────────┘
                       │
              model_best_scs_v12_lb6_s42.onnx
              scaler_scs_v12_lb6.pkl
```

---

## Cài đặt & chạy

**Backend (Python 3.13):**

```bash
cd source/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**

```bash
cd source/frontend
npm install
npm run dev   # http://localhost:5173
```

**Docker (khuyến nghị):**

```bash
cd source
docker compose up --build
```

Chi tiết xem `source/DEPLOY.md`.

---

## Công nghệ sử dụng

| Layer     | Công nghệ                           |
| --------- | ------------------------------------- |
| AI/ML     | Python, PyTorch, ONNX Runtime         |
| Backend   | FastAPI, Python 3.13, Docker          |
| Frontend  | React, Vite, react-globe.gl           |
| Dữ liệu | xarray, pandas, scikit-learn, netCDF4 |
