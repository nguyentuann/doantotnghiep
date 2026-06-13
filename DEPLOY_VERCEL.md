# Deploy lên Vercel (Frontend + Backend, KHÔNG cần Docker)

Toàn bộ dự án (React frontend + FastAPI backend) chạy trên **một** project Vercel:
- **Frontend**: build tĩnh từ `source/frontend` (Vite).
- **Backend**: serverless Python function tại `api/index.py` (FastAPI ASGI).

## Cấu trúc đã chuẩn bị sẵn (ở gốc repo)

```
vercel.json            # cấu hình build FE + route /api → function
requirements.txt       # deps Python cho serverless (fastapi, onnxruntime, sklearn, numpy, pandas)
.vercelignore          # loại dữ liệu nặng (ERA5, csv, venv) khỏi upload
api/
  index.py             # entry FastAPI (mount router ở cả "" và "/api")
  _app/                # bundle tự chứa (2.3 MB) — KHÔNG cần source/ lúc chạy
    backend/{routers,services,schemas,data}
    model_ai/
      models/final/model_best_scs_v12_lb6_s42.onnx (+ .onnx.data)
      models/scaler_scs_v12_lb6.pkl
      results/{results_log.json, figures/*/*.json}
```

> Backend dùng ONNX seed s42 (31 feat, lookback 6). Không cần file ERA5 — tự dùng climatology fallback.

## Cách deploy

### Cách A — Qua web (dễ nhất)
1. Push repo lên GitHub.
2. Vào https://vercel.com → **Add New → Project** → import repo.
3. Để **Root Directory = `.`** (gốc repo — KHÔNG đặt `source/frontend`).
4. Vercel tự đọc `vercel.json`. Bấm **Deploy**.

### Cách B — Qua CLI
```bash
npm i -g vercel
cd "d:/1. Do An Tot Nghiep/doantotnghiep"
vercel            # preview
vercel --prod     # production
```

## Sau khi deploy — kiểm tra
- `https://<app>.vercel.app/`            → giao diện web (globe 3D, dashboard)
- `https://<app>.vercel.app/api/health`  → `{"status":"ok","model_ready":true}`
- `https://<app>.vercel.app/api/storms`  → danh sách 40 bão demo

## Lưu ý
- **Cold start** request đầu ~2–4s (serverless nạp ONNX). Sau đó nhanh.
- **Độ chính xác**: trên cloud không có ERA5 thật → 18 đặc trưng môi trường/annulus dùng climatology. Demo trên các bão có sẵn vẫn chạy đầy đủ; dự báo bão hoàn toàn mới sẽ kém chính xác hơn con số 101.3 km (giới hạn dữ liệu chung của mọi deploy cloud).
- **Python runtime**: Vercel mặc định 3.12 (đủ cho onnxruntime/sklearn). Không cần chỉnh.
- **Đồng bộ code**: `api/_app/backend` là BẢN SAO của `source/backend`. Khi sửa backend gốc, nhớ copy lại (hoặc chạy lại bước bundle).

## ⚠️ Tối ưu cho giới hạn 500 MB của Vercel
Bundle `api/_app/backend` đã được **patch riêng cho Vercel** để bỏ `scikit-learn`,
`scipy` và `pandas` (tổng ~400 MB) — nếu giữ thì bundle 596 MB > giới hạn 500 MB:
- Scaler nạp từ `scaler_scs_v12_lb6.npz` (numpy) thay vì `.pkl` (sklearn) — xem `_ScalerShim` trong `preprocessor.py`.
- Mọi `pd.Timestamp` → `datetime` (stdlib) trong `preprocessor.py` & `predictor.py`.
- `requirements.txt` chỉ còn: fastapi, onnxruntime, numpy, pydantic, python-multipart.

→ **KHÔNG copy đè** `api/_app/backend/services/{preprocessor,predictor}.py` từ `source/backend`
(sẽ kéo lại pandas/sklearn → vỡ giới hạn). Chỉ copy router/schemas/data nếu cần.

Tạo lại npz khi đổi model:
```bash
python -c "import pickle,numpy as np; s=pickle.load(open('source/model_ai/models/scaler_scs_v12_lb6.pkl','rb')); np.savez('api/_app/model_ai/models/scaler_scs_v12_lb6.npz', mean=s.mean_, scale=s.scale_, n_features=np.array(int(s.n_features_in_)))"
```
