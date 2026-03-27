# G5/G6 — Training

**File:** `train.py` (TODO)
**Stage G5:** Train LSTM baseline
**Stage G6:** Train BiLSTM + Transformer

---

## Checkpoint cần đạt

| Stage | Điều kiện |
|-------|-----------|
| G5 | Val MAE 24h < 200 km |
| G6 | Cải thiện ≥ 10% so với LSTM baseline |

---

## Kế hoạch triển khai

### Chiến lược training
- **Optimizer:** Adam (lr=0.001 từ config)
- **Scheduler:** `ReduceLROnPlateau` (patience=10, factor=0.5) — giảm lr khi val loss không cải thiện
- **Early stopping:** patience=20 epoch (từ config)
- **Gradient clipping:** norm=1.0 — tránh exploding gradient
- **Checkpoint:** lưu model tốt nhất theo val loss vào `models/checkpoints/best_{model_name}.pt`

### Thứ tự train
```
1. LSTM      → đạt G5 checkpoint → lưu val MAE làm baseline
2. BiLSTM    → so sánh với LSTM baseline → đạt G6 nếu cải thiện ≥10%
3. Transformer → so sánh với LSTM baseline → đạt G6 nếu cải thiện ≥10%
4. Lưu model tốt nhất tổng thể → models/final/model_best.pt
```

### Input data
```python
import numpy as np
data = np.load("data/features/sequences.npz")
X_train, y_train = data["X_train"], data["y_train"]  # (22567, 8, 12), (22567, 4)
X_val,   y_val   = data["X_val"],   data["y_val"]    # (2557, 8, 12),  (2557, 4)
```

### Output
```
models/checkpoints/best_lstm.pt
models/checkpoints/best_bilstm.pt
models/checkpoints/best_transformer.pt
models/final/model_best.pt       ← model tốt nhất tổng thể
results/results_log.json         ← log tất cả experiments
```
