# DEPLOY.md — Hướng dẫn triển khai Typhoon Tracker

Triển khai hệ thống dự đoán đường đi bão Biển Đông gồm 2 service:
- **Backend** — FastAPI + ONNX Runtime (port 8000)
- **Frontend** — React + Vite, serve bằng Nginx (port 5173 → 80)

---

## 1. Kiến trúc container

```
┌─────────────────────────────────────────────────────────┐
│ Host                                                       │
│                                                            │
│  :5173 ──► [frontend / nginx]                              │
│                 │  /api/* (proxy, strip prefix)            │
│                 ▼                                          │
│  :8000 ──► [backend / uvicorn] ──► /app/model_ai/models/   │
│                                      ├─ final/*.onnx        │
│                                      └─ scaler_*.pkl        │
│                                                            │
│  typhoon-net (bridge network) ─ frontend ↔ backend         │
└─────────────────────────────────────────────────────────┘
```

- Trình duyệt gọi `/api/storms` (same-origin) → Nginx proxy → `backend:8000/storms`.
- Không bị CORS vì frontend và API cùng origin (`localhost:5173`).
- Backend chỉ copy **ONNX + scaler** (vài KB) vào image — KHÔNG copy ERA5/data (nhiều GB).

---

## 2. Yêu cầu

- **Docker** ≥ 24.x và **Docker Compose** v2 (`docker compose`, không phải `docker-compose`).
- Kiểm tra:
  ```powershell
  docker --version
  docker compose version
  ```

---

## 3. Workflow đầy đủ (lần đầu deploy / khi update model)

Thứ tự đề xuất khi muốn deploy model mới (vd `scs_v11_lb6`):

### Bước 1 — Chuẩn bị ONNX
Chạy ở `source/model_ai/` với venv activated:
```powershell
.venv\Scripts\activate

# Export ONNX cho model tốt nhất (chọn 1 seed hoặc nhiều)
$env:PYTHONIOENCODING="utf-8"; python -m src.g8_export.export_onnx --tag scs_v11_lb6 --model transformer --seed 42

# Verify file đã tạo: model_best_scs_v11_lb6_s42.onnx
ls models\final\model_best_scs_v11_lb6*.onnx
```

### Bước 2 — Test backend local (không Docker)
Trước khi build container, đảm bảo backend chạy ổn với model mới:
```powershell
cd ..\backend
.venv\Scripts\activate                          # tạo venv nếu chưa có
uvicorn main:app --reload --port 8000

# Trong terminal khác / browser:
#   http://localhost:8000/health  →  {"status":"ok","model_ready":true}
#   http://localhost:8000/docs    →  Swagger UI

# Log cần thấy:
#   [preprocessor] Scaler: scaler_scs_v11_lb6.pkl (29 features)
#   [predictor] ONNX loaded: model_best_scs_v11_lb6_s42.onnx (output=16, lookback=6)
```

### Bước 3 — Build & chạy Docker
```powershell
cd ..                                            # về source/

# Build image + chạy compose (cả backend + frontend)
docker compose up --build -d

# Verify
docker compose ps                                # cả 2 service "running"
docker compose logs backend --tail=30            # check log scaler/ONNX
curl http://localhost:8000/health                # model_ready: true

# Mở browser
#   http://localhost:5173   →  Frontend
#   http://localhost:8000/docs  →  Backend Swagger
```

### Bước 4 — Cleanup khi cần
```powershell
docker compose down                              # dừng + xóa container (giữ image)
docker compose down --rmi all                    # + xóa image
docker compose down -v                           # + xóa volume (không dùng ở đây nhưng để biết)
```

---

## 4. Triển khai nhanh bằng Docker Compose (khuyên dùng)

Tất cả lệnh chạy từ thư mục **`source/`**:

```powershell
# Build image + chạy nền cả 2 service
docker compose up --build -d

# Xem log realtime
docker compose logs -f

# Kiểm tra trạng thái
docker compose ps

# Dừng + xóa container (giữ image)
docker compose down
```

Sau khi `up`:
- Frontend: http://localhost:5173
- Backend docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health

---

## 5. Build / chạy từng container thủ công

### Backend
> Build context BẮT BUỘC là `source/` (cần truy cập `model_ai/models/`).

```powershell
# Từ source/
docker build -f backend/Dockerfile -t typhoon-backend:latest .

docker run -d --name typhoon-backend -p 8000:8000 typhoon-backend:latest
```

### Frontend
```powershell
# Từ source/
docker build -t typhoon-frontend:latest ./frontend

# Lưu ý: chạy thủ công thì nginx proxy "backend:8000" sẽ không resolve
# nếu không cùng network. Nên dùng docker compose, hoặc tạo network chung:
docker network create typhoon-net
docker run -d --name typhoon-backend  --network typhoon-net -p 8000:8000 typhoon-backend:latest
docker run -d --name typhoon-frontend --network typhoon-net -p 5173:80   typhoon-frontend:latest
```

---

## 6. Chạy local KHÔNG dùng Docker (dev)

### Backend
```powershell
cd source\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### Frontend
```powershell
cd source\frontend
npm install
npm run dev        # http://localhost:5173 (vite proxy /api → :8000)
```

---

## 7. Cập nhật model mới (lần sau khi có tag mới)

Backend hiện đã được chuẩn bị cho `scs_v11_lb6` (29 features, lookback=6).
Khi có tag mới hơn (vd `scs_v12`), làm theo các bước:

1. **Export ONNX** từ checkpoint:
   ```powershell
   cd source\model_ai
   .venv\Scripts\activate
   $env:PYTHONIOENCODING="utf-8"; python -m src.g8_export.export_onnx --tag {tag_moi} --model transformer --seed 42
   ```
   → tạo `model_ai/models/final/model_best_{tag_moi}_s42.onnx`

2. **Thêm vào đầu priority list** trong [predictor.py](backend/services/predictor.py):
   ```python
   _ONNX_CANDIDATES = [
       _BASE / "model_best_{tag_moi}_s42.onnx",   # ← thêm
       _BASE / "model_best_scs_v11_lb6_s42.onnx",
       ...
   ]
   ```

3. **Thêm scaler** lên đầu trong [preprocessor.py](backend/services/preprocessor.py):
   ```python
   _SCALER_CANDIDATES = [
       _BASE / "scaler_{tag_moi}.pkl",            # ← thêm
       _BASE / "scaler_scs_v11_lb6.pkl",
       ...
   ]
   ```

4. ⚠️ **Kiểm tra n_features**: nếu thay đổi (vd thêm/bớt feature), cần update `prepare_input()` để build đúng thứ tự cột — backend tự detect `lookback` và `n_outputs` từ ONNX nhưng KHÔNG tự sinh feature mới.

5. **Rebuild image**:
   ```powershell
   docker compose up --build -d backend
   ```

---

## 8. Cấu hình môi trường

| Biến / cấu hình | Vị trí | Mặc định |
|-----------------|--------|----------|
| Backend port | `docker-compose.yml` ports | 8000 |
| Frontend port | `docker-compose.yml` ports | 5173 → 80 |
| CORS origins | [backend/main.py](backend/main.py) | localhost:5173, 4173 |
| API proxy | [frontend/nginx.conf](frontend/nginx.conf) | `/api/` → `backend:8000/` |
| Frontend API base | [frontend/src/api/storms.js](frontend/src/api/storms.js) | `/api` |

Khi deploy lên domain thật, cập nhật `allow_origins` trong `main.py` thêm domain production.

---

## 9. Triển khai production (gợi ý)

### 9.1 Reverse proxy + HTTPS
Đặt một reverse proxy (Caddy / Traefik / Nginx) phía trước, terminate TLS:
- `https://typhoon.example.com`        → frontend container
- `https://typhoon.example.com/api/*`  → đã được nginx frontend proxy sang backend

Với Caddy (tự động HTTPS), `Caddyfile`:
```
typhoon.example.com {
    reverse_proxy localhost:5173
}
```

### 9.2 Các nền tảng cloud
- **Backend** (image nhẹ, CPU-only): Render, Railway, Fly.io, Google Cloud Run, AWS ECS.
- **Frontend** (static): có thể tách build `npm run build` rồi deploy `dist/` lên Cloudflare Pages / Netlify / Vercel, trỏ `/api` về backend URL (sửa `nginx.conf` hoặc dùng biến env Vite).
- Cloud Run: push image `typhoon-backend` lên Artifact Registry, deploy, set `--port 8000`.

### 9.3 Tài nguyên tối thiểu
- Backend: 0.5 vCPU / 512 MB RAM (ONNX CPU inference rất nhẹ).
- Frontend: 0.25 vCPU / 128 MB RAM (nginx static).

---

## 10. Troubleshooting

| Triệu chứng | Nguyên nhân | Cách xử lý |
|-------------|-------------|------------|
| `model_ready: false` ở `/health` | ONNX không tìm thấy / lỗi load | Kiểm tra `model_ai/models/final/*.onnx` đã copy vào image; xem `docker compose logs backend` |
| Frontend 502 khi gọi `/api` | backend chưa sẵn sàng hoặc khác network | Đảm bảo cùng `typhoon-net`; chờ healthcheck backend pass |
| `input.size(-1) must be equal to ...` | scaler/ONNX features không khớp | Đồng bộ ONNX + scaler cùng tag; check `n_features` |
| Build backend quá lâu / nặng | build context kéo cả `model_ai/data` | Đã loại trong `.dockerignore`; xác nhận file tồn tại |
| `docker compose` not found | Docker Compose v1 cũ | Cài Docker Desktop mới (Compose v2) |
| Frontend trắng trang | route SPA 404 | `nginx.conf` đã có `try_files ... /index.html` — rebuild frontend |

### Lệnh debug hữu ích
```powershell
# Vào trong container backend
docker exec -it typhoon-backend sh
ls /app/model_ai/models/final/        # kiểm tra ONNX
ls /app/model_ai/models/*.pkl         # kiểm tra scaler

# Test API trực tiếp
curl http://localhost:8000/health
curl http://localhost:8000/storms

# Xem log riêng từng service
docker compose logs backend
docker compose logs frontend
```

---

## 11. Checklist deploy

- [ ] `docker compose version` ≥ v2
- [ ] Đã export ONNX cho tag muốn dùng (vd `scs_v11_lb6_s42`)
- [ ] Test backend local (`uvicorn`) trả `model_ready: true` trước khi build Docker
- [ ] `docker compose up --build -d` chạy không lỗi
- [ ] `docker compose logs backend` thấy đúng scaler + ONNX được load
- [ ] `/health` trả `model_ready: true`
- [ ] Frontend load được danh sách bão + vẽ track
- [ ] (Production) Cập nhật CORS origins + HTTPS reverse proxy
