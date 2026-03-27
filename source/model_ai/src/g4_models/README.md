# G4 — Kiến trúc mô hình

**File:** `models.py`
**Stage:** Định nghĩa kiến trúc — chưa train
**Checkpoint:** Import thành công, sanity check pass ✅

---

## Chạy sanity check

```bash
cd source/model_ai
python -m src.g4_models.models
```

---

## Tổng quan 3 mô hình

| # | Model | Tham số | Mục tiêu | Stage |
|---|-------|---------|----------|-------|
| 1 | LSTM Baseline | 213,572 | Val MAE 24h < 200 km | G5 |
| 2 | BiLSTM + Attention | 574,852 | Cải thiện ≥ 10% vs LSTM | G6 |
| 3 | Temporal Transformer | 103,524 | Cải thiện ≥ 10% vs LSTM | G6 |

Train theo thứ tự 1 → 2 → 3. **Mô hình sau chỉ train khi mô hình trước đạt checkpoint.**

---

## Input / Output

```
Input  X : [Batch, 8, 12]   — 8 bước × 6h = 48h lịch sử, 12 features (đã StandardScaler)
Output y : [Batch, 4]       — [lat_24h, lon_24h, lat_48h, lon_48h] (độ thực, chưa scale)
```

---

## Hyperparameter từ config.yaml

| Tham số | Giá trị | Giải thích |
|---------|---------|-----------|
| `n_features` | **12** | Số features đầu vào |
| `hidden_size` | **128** | Kích thước hidden state LSTM/BiLSTM |
| `num_layers` | **2** | Số lớp LSTM xếp chồng. Layer 1 → pattern cơ bản, Layer 2 → pattern cao hơn |
| `dropout` | **0.3** | Tắt 30% neurons khi train. Bằng 0 khi eval |
| `lookback` | **8** | 8 bước × 6h = 48h nhìn về quá khứ |

---

## Model 1 — LSTM Baseline

```
[B, 8, 12] → LSTM(128, 2 lớp, dropout=0.3) → lấy bước cuối [B, 128]
           → LayerNorm(128)
           → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→4)
           → [B, 4]
```

**Tại sao lấy hidden state bước cuối?**
LSTM đọc tuần tự — hidden state bước 8 đã "nhớ" toàn bộ 7 bước trước qua cơ chế forget/input gate.

---

## Model 2 — BiLSTM + Self-Attention

```
[B, 8, 12] → BiLSTM(hidden=128, bidirectional) → [B, 8, 256]
           → Attention: Linear(256→1) → Softmax → trọng số [B, 8]
           → Weighted sum → context [B, 256]
           → LayerNorm → Linear(256→128) → GELU → Dropout → Linear(128→4)
           → [B, 4]
```

**Tại sao Bidirectional?**
Chuỗi lịch sử đã có đủ — BiLSTM đọc cả xuôi lẫn ngược → tại bước t biết cả t−1 lẫn t+1.

**Tại sao cần Attention?**
Không phải mọi bước đều quan trọng như nhau. Bước bão đổi hướng đột ngột quan trọng hơn bước đi thẳng. Attention tự học trọng số này.

**GELU thay vì ReLU?**
GELU mềm hơn (không cắt cứng tại 0), phù hợp hơn với context vector có thể âm.

---

## Model 3 — Temporal Transformer

```
[B, 8, 12] → Linear(12→64) → + Learnable PE(1, 8, 64)
           → TransformerEncoder(d=64, heads=4, ff=256, layers=2, norm_first=True)
           → lấy token cuối [B, 64]
           → Linear(64→32) → ReLU → Linear(32→4)
           → [B, 4]
```

**Transformer settings:**

| Tham số | Giá trị | Giải thích |
|---------|---------|-----------|
| `d_model` | **64** | Nhỏ hơn hidden_size=128 vì Self-Attention O(n²) — bù lại bằng d_model nhỏ |
| `nhead` | **4** | 64/4 = 16 chiều/head. Mỗi head học 1 loại phụ thuộc khác nhau |
| `dim_feedforward` | **256** | FFN bên trong encoder = 4×d_model (quy tắc chuẩn) |
| `norm_first` | **True** | Pre-LN: normalize trước Attention → gradient ổn định hơn |
| Positional Encoding | **Learnable** | Sin/cos cố định cho chuỗi dài (NLP). Learnable tốt hơn cho chuỗi ngắn 8 bước |

---

## Loss Function — HaversineLoss

```
Loss = 0.6 × Haversine(pred_24h, true_24h)
     + 0.4 × Haversine(pred_48h, true_48h)
```

**Tại sao Haversine thay vì MSE?**
MSE: `(Δlat)² + (Δlon)²` — nhưng 1° kinh độ ≠ 1° vĩ độ về km (phụ thuộc vĩ độ). Ở 20°N: 1° kinh ≈ 103 km, 1° vĩ ≈ 111 km. Haversine tính khoảng cách thực trên mặt cầu → đơn vị km nhất quán.

**Tại sao trọng số 0.6 / 0.4?**
Dự báo 24h quan trọng hơn cho ra quyết định sơ tán. 48h có sai số tích lũy lớn hơn tự nhiên.

---

## API

```python
from src.g4_models import build_model, build_loss, count_parameters

cfg    = load_config()
model  = build_model("lstm", cfg)        # hoặc "bilstm", "transformer"
loss   = build_loss(cfg)                 # HaversineLoss(0.6, 0.4)

y_pred = model(x)                        # [B, 4]
l      = loss(y_pred, y_true)            # scalar km
mae24, mae48 = loss.mae_km(y_pred, y_true)

print(count_parameters(model))           # 213572
```
