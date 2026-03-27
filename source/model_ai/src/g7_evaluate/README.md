# G7 — Đánh giá

**File:** `evaluate.py` (TODO)
**Checkpoint:** Skill Score ≥ 20% so với CLIPER

---

## Kế hoạch triển khai

### CLIPER Baseline (bắt buộc tính trước)

```python
# Persistence: ngoại suy tuyến tính từ 2 bước gần nhất
dlat = LAT[t] - LAT[t-1]
dlon = LON[t] - LON[t-1]
pred_lat_24h = LAT[t] + 4 * dlat   # 4 bước × 6h = 24h
pred_lon_24h = LON[t] + 4 * dlon
pred_lat_48h = LAT[t] + 8 * dlat
pred_lon_48h = LON[t] + 8 * dlon
```

### 4 Chỉ số đánh giá

| Chỉ số | Đơn vị | Mục tiêu |
|--------|--------|---------|
| MAE 24h | km | < 150 km |
| MAE 48h | km | < 250 km |
| RMSE 24h | km | — |
| Skill Score | % | > 30% |

```
Skill Score = (MAE_CLIPER − MAE_Model) / MAE_CLIPER × 100%
> 0% → model tốt hơn CLIPER
< 0% → model tệ hơn CLIPER (vấn đề nghiêm trọng)
```

### Phân tích theo loại quỹ đạo

```
westward  (~65%) — phổ biến nhất, model nên đạt MAE thấp nhất
recurving (~20%) — quặt bắc, khó hơn
southward (~15%)
erratic   (~5%)  — không tính vào Skill Score tổng, báo cáo riêng
```

### 6 Biểu đồ bắt buộc trong báo cáo

```
results/figures/learning_curve.png      — train/val loss theo epoch
results/figures/mae_by_type.png         — MAE theo loại quỹ đạo
results/figures/scatter_pred_actual.png — dự đoán vs thực tế
results/figures/mae_vs_horizon.png      — MAE tăng theo horizon 6h→48h
results/figures/model_comparison.png    — so sánh 3 model + CLIPER
results/figures/track_XXXXXXX.html      — bản đồ Folium (≥ 3 cơn bão)
```
