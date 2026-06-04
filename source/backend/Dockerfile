# ============================================================
# Backend Dockerfile — Typhoon Tracker API (FastAPI + ONNX)
# ------------------------------------------------------------
# Build context BẮT BUỘC là thư mục `source/` (không phải backend/)
# vì image cần copy cả model_ai/models/ (ONNX + scaler).
#
#   docker build -f backend/Dockerfile -t typhoon-backend:latest .
#   (chạy từ source/)
# ============================================================
FROM python:3.13-slim

# curl cho HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# 1) Cài dependencies trước (tận dụng layer cache)
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Copy source code backend
COPY backend/ /app/backend/

# 3) Copy CHỈ model artifacts cần cho inference (không copy toàn bộ model_ai)
#    predictor.py / preprocessor.py resolve path = /app/model_ai/models/...
COPY model_ai/models/final/*.onnx /app/model_ai/models/final/
COPY model_ai/models/*.pkl        /app/model_ai/models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
