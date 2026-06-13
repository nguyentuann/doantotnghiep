from fastapi import APIRouter, HTTPException
from schemas.request import PredictRequest
from schemas.response import PredictResponse
from services import preprocessor, predictor, track_builder

router = APIRouter(prefix="/predict", tags=["predict"])

# MAE tham chiếu từ G7 (km) — cập nhật khi có kết quả G7
_MAE_REFERENCE = {
    "lstm":        {"mae_24h": 120.1, "mae_48h": 202.1},
    "bilstm":      {"mae_24h": 82.9,  "mae_48h": 161.5},
    "transformer": {"mae_24h": 74.9,  "mae_48h": 150.1},
}
_DEFAULT_MODEL = "transformer"


@router.post("", response_model=PredictResponse)
def predict_track(req: PredictRequest):
    points = [p.model_dump() for p in req.track]
    model_name = (req.model_name or _DEFAULT_MODEL).lower()

    if model_name not in ("lstm", "bilstm", "transformer"):
        raise HTTPException(status_code=400, detail=f"model_name không hợp lệ: {model_name}")

    if not predictor.is_ready():
        raise HTTPException(status_code=503, detail="ONNX model chưa load được. Kiểm tra model_best.onnx.")

    try:
        x = preprocessor.prepare_input(points, lookback=predictor._lookback)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Lỗi tính features: {e}")

    try:
        pred = predictor.predict(x)                       # (1, 16) residual delta scaled
        scaler = preprocessor._load_scaler()
        coords = preprocessor.decode_output(pred, x, scaler)  # (16,) abs lat/lon cho 8 bước 6h
        # bước: 6,12,18,24,30,36,42,48h → 24h=idx[6,7], 48h=idx[14,15]
        lat_24h, lon_24h = float(coords[6]), float(coords[7])
        lat_48h, lon_48h = float(coords[14]), float(coords[15])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi inference: {e}")
    origin = points[-1]
    mae_ref = _MAE_REFERENCE.get(model_name, {})

    forecast = track_builder.build_forecast(
        origin_lat=origin["lat"],
        origin_lon=origin["lon"],
        lat_24h=float(lat_24h),
        lon_24h=float(lon_24h),
        lat_48h=float(lat_48h),
        lon_48h=float(lon_48h),
        model_name=model_name,
        mae_24h=mae_ref.get("mae_24h"),
        mae_48h=mae_ref.get("mae_48h"),
    )

    return {"forecast": forecast}
