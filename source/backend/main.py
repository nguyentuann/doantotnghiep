"""
main.py — FastAPI app cho Typhoon Tracker
-----------------------------------------
Chạy dev:
  cd source/backend
  pip install -r requirements.txt
  uvicorn main:app --reload --port 8000

Endpoints:
  GET  /health
  GET  /storms
  GET  /storms/{sid}
  POST /predict
  GET  /models
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import predict, storms, models

app = FastAPI(
    title="Typhoon Tracker API",
    description="Dự đoán đường đi bão Biển Đông — ĐATN 2026",
    version="1.0.0",
)

# CORS — cho phép frontend dev (localhost:5173) và production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:4173",
        "https://typhoon-tracker.pages.dev",  # cập nhật khi deploy
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(storms.router)
app.include_router(models.router)


@app.get("/health", tags=["health"])
def health():
    from services.predictor import is_ready
    return {
        "status": "ok",
        "model_ready": is_ready(),
    }
