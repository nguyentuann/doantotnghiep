"""
api/index.py — Entry point serverless cho Vercel (FastAPI ASGI)
================================================================
Tái sử dụng nguyên backend trong api/_app/backend (bản sao tối thiểu của
source/backend). Mount router ở cả "" và "/api" để chạy đúng dù Vercel
truyền path gốc (/api/storms) hay path đã strip (/storms).

Cấu trúc bundle (giữ nguyên đường dẫn tương đối của code gốc):
  api/_app/backend/{routers,services,schemas,data}
  api/_app/model_ai/{models,results}
"""
import os
import sys

# Cho phép import 'routers', 'services', 'schemas' từ bản sao backend
_BACKEND = os.path.join(os.path.dirname(__file__), "_app", "backend")
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import predict, storms, models, dashboard  # noqa: E402

app = FastAPI(
    title="Typhoon Tracker API (Vercel)",
    description="Dự đoán đường đi bão Biển Đông — ĐATN 2026",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # same-origin trên Vercel; mở rộng cho preview domains
    allow_methods=["*"],
    allow_headers=["*"],
)

_ROUTERS = (predict.router, storms.router, models.router, dashboard.router)


def _health():
    from services.predictor import is_ready
    return {"status": "ok", "model_ready": is_ready()}


# Mount 2 lần: prefix "/api" (path gốc) và "" (nếu Vercel strip /api)
for _r in _ROUTERS:
    app.include_router(_r)                 # /storms, /predict, /models, /dashboard/*
    app.include_router(_r, prefix="/api")  # /api/storms, /api/predict, ...

app.add_api_route("/health", _health, tags=["health"])
app.add_api_route("/api/health", _health, tags=["health"])
