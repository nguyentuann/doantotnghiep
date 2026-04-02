# Hướng cải thiện model trong tương lai

> Tài liệu này ghi lại các hướng nâng cấp có thể thực hiện sau khi hoàn thành ĐATN.
> Kết quả hiện tại: Skill Score ~10–11% vs CLIPER baseline (giới hạn do dữ liệu, không phải model).

---

## 1. Bổ sung dữ liệu môi trường khí quyển (tác động lớn nhất)

**Vấn đề hiện tại:** Model chỉ dùng dữ liệu track (vị trí, gió, áp suất) từ IBTrACS.
CLIPER cũng chỉ dùng thông tin này → model khó vượt CLIPER nhiều.

### 1.1 ERA5 Reanalysis (ECMWF)
- **Wind shear (0–200 hPa):** yếu tố quan trọng nhất ảnh hưởng đến hướng và cường độ bão
- **Geopotential height 500 hPa:** steering flow (dòng dẫn bão)
- **Relative humidity 500–700 hPa:** ảnh hưởng cường độ
- **Sea surface temperature thực tế:** thay thế climatology tháng hiện tại
- Nguồn: https://cds.climate.copernicus.eu (miễn phí, cần đăng ký)
- Kích thước: ~10–50 GB cho khu vực Biển Đông từ 1980–2024

### 1.2 NOAA OISST
- SST thực tế độ phân giải 0.25°, hàng ngày
- Thay thế `sst_c` (climatology tháng) bằng SST tại đúng thời điểm bão
- Nguồn: https://www.ncei.noaa.gov/products/optimum-interpolation-sst

### 1.3 JMA Best Track (độc lập)
- Dữ liệu track từ JMA độc lập, có thể dùng ensemble với IBTrACS
- Nguồn: https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/trackarchives.html

**Kỳ vọng cải thiện:** Skill Score có thể tăng lên 20–35% với wind shear + SST thực.

---

## 2. Cải thiện kiến trúc model

### 2.1 Graph Neural Network (GNN)
- Mô hình hóa tương tác giữa nhiều cơn bão đồng thời trên Biển Đông
- Mỗi bão là một node, khoảng cách/tương tác khí quyển là edge
- Framework: PyTorch Geometric

### 2.2 Probabilistic Forecasting
- Thay vì dự đoán 1 điểm (lat/lon), dự đoán **phân phối xác suất** (uncertainty cone)
- Dùng: Gaussian Process, Bayesian LSTM, hoặc Normalizing Flow
- Kết quả: vẽ được vùng không chắc chắn trên bản đồ (như NHC/JTWC)

### 2.3 Multi-step Autoregressive
- Hiện tại: dự đoán trực tiếp +24h và +48h từ 8 bước lịch sử
- Cải thiện: dự đoán tuần tự +6h → +12h → ... → +72h (mỗi bước dùng output trước)
- Cho phép dự báo đến 72h hoặc 120h

### 2.4 ConvLSTM
- Thay phép nhân ma trận trong LSTM bằng **tích chập (convolution)**
- Input: lưới khí quyển dạng `(T, H, W, channels)` thay vì chuỗi 1D
- Học được **pattern không gian di chuyển theo thời gian** — ví dụ: front lạnh từ bắc đẩy bão sang phải
- Ứng dụng rộng trong radar nowcasting và dự báo mưa
- **Yêu cầu:** dữ liệu lưới ERA5, không dùng được với IBTrACS đơn thuần
- Kỳ vọng Skill Score: ~25–35%

### 2.5 Spatio-temporal Transformer
- Mở rộng Transformer hiện tại: attention trên **cả 2 trục — thời gian VÀ không gian** cùng lúc
- Temporal attention: "bước t=6 quan trọng hơn t=1"
- Spatial attention: "ô lưới phía tây bắc ảnh hưởng nhiều hơn phía đông"
- Có thể mô hình hóa tương tác giữa nhiều cơn bão (multi-storm attention)
- Hướng nghiên cứu tiên tiến nhất hiện nay (2022–2024)
- **Yêu cầu:** dữ liệu lưới ERA5 + track IBTrACS
- Kỳ vọng Skill Score: ~30–40%

### 2.6 Transformer với positional encoding tốt hơn
- Thêm **rotary positional embedding (RoPE)** để encode vị trí địa lý thực tế
- Thêm **cross-attention** giữa track sequence và atmospheric grid data

---

## 3. Cải thiện feature engineering

### 3.1 Features từ ERA5 (nếu có)
```
wind_shear_200_850    # |V200 - V850|, hPa
steering_flow_u       # u-component 500 hPa
steering_flow_v       # v-component 500 hPa
rhum_700              # relative humidity 700 hPa
sst_actual            # SST tại điểm bão (thay sst_c climatology)
```

### 3.2 Features bổ sung từ IBTrACS hiện có
```
intensity_change_6h   # vmax_t - vmax_{t-1}: rapid intensification signal
track_curvature       # độ cong của track 3 bước gần nhất
basin_region          # vị trí trong Biển Đông: SCS_N / SCS_S / WPAC
```

### 3.3 Lookback dài hơn
- Hiện tại: lookback = 8 bước (48h lịch sử)
- Thử: lookback = 16 bước (96h lịch sử) — có thể giúp Transformer hơn LSTM
- Cần tăng `model.lookback` trong `config.yaml`

---

## 4. Cải thiện quá trình training

### 4.1 Tăng dữ liệu (Data Augmentation)
- **Mirroring:** lật track theo trục kinh độ (giả lập bão Nam Bán Cầu)
- **Noise injection:** thêm nhiễu nhỏ vào vị trí để tăng robustness
- **Mixup:** kết hợp 2 sequences có trọng số ngẫu nhiên

### 4.2 Learning rate scheduler tốt hơn
- Hiện tại: ReduceLROnPlateau (giảm khi plateau)
- Thử: **Cosine Annealing with Warm Restarts** (torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
- Hoặc: **One-Cycle Policy** — thường tốt hơn cho Transformer

### 4.3 Ensemble model
- Kết hợp prediction của LSTM + BiLSTM + Transformer bằng weighted average
- Weight theo Skill Score trên validation set
- Kỳ vọng: giảm variance, tăng Skill Score thêm 1–3%

### 4.4 Hyperparameter tuning
- Dùng **Optuna** để tìm tự động: hidden_size, dropout, lr, lookback, batch_size
- Ước tính: ~50–100 trials, mỗi trial ~5 phút → chạy qua đêm

---

## 5. Mở rộng vùng dự báo

### 5.1 Dự báo cường độ (intensity forecasting)
- Hiện tại: chỉ dự báo vị trí (lat/lon)
- Bổ sung: dự báo `vmax` tại +24h và +48h
- Target: 4 → 6 outputs `[lat_24h, lon_24h, lat_48h, lon_48h, vmax_24h, vmax_48h]`

### 5.2 Dự báo đến 72h
- Thêm anchor step +12 (72h) vào target
- Cần điều chỉnh `sequences.py` và loss function

### 5.3 Rapid Intensification detection
- Phân loại nhị phân: bão có RI (tăng ≥30 kt trong 24h) hay không?
- Kết hợp với dự báo track để cảnh báo sớm

---

## 6. Tích hợp hệ thống thực tế

### 6.1 Real-time data pipeline
- Kết nối với IBTrACS Near-Real-Time (NRT) feed
- Tự động download và xử lý khi có bão mới
- Dùng: Apache Airflow hoặc cron job đơn giản

### 6.2 Operational ONNX deployment
- Hiện tại: export ONNX thành công
- Tiếp theo: serve qua **ONNX Runtime** trong FastAPI (nhanh hơn PyTorch ~2–3x)
- Hoặc: chuyển sang **TensorRT** nếu có GPU NVIDIA

### 6.3 So sánh với mô hình nghiệp vụ
- So sánh với ECMWF (IFS), GFS, HWRF để biết khoảng cách còn lại
- Dữ liệu forecast archives: https://www.nco.ncep.noaa.gov/pmb/nwprod/

---

## Tóm tắt ưu tiên

| Hướng | Độ khó | Tác động | Ưu tiên |
|---|---|---|---|
| ERA5 wind shear + SST thực | Cao | Rất cao (+10–20% Skill) | ★★★★★ |
| Ensemble LSTM+BiLSTM+Transformer | Thấp | Trung bình (+2–3%) | ★★★★☆ |
| Lookback dài hơn (16 bước) | Thấp | Thấp–Trung bình | ★★★☆☆ |
| Hyperparameter tuning (Optuna) | Thấp | Trung bình | ★★★☆☆ |
| Probabilistic forecasting | Cao | Cao (uncertainty cone) | ★★★☆☆ |
| Dự báo cường độ (vmax) | Trung bình | Cao | ★★★☆☆ |
| GNN (multi-storm interaction) | Rất cao | Không chắc | ★★☆☆☆ |
| Real-time pipeline | Cao | Cao (production) | ★★☆☆☆ |
