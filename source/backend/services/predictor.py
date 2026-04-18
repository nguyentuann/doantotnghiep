"""
predictor.py — ONNX Runtime inference service
----------------------------------------------
Load model_best.onnx một lần khi khởi động server.
Nhận input đã được scale, trả về [lat_24h, lon_24h, lat_48h, lon_48h].
"""

import numpy as np
from pathlib import Path

# ONNX model path (relative từ source/backend/)
_STEERING_ONNX = Path(__file__).parent.parent.parent / "model_ai/models/final/model_best_wp_steering.onnx"
_WP_FULL_ONNX  = Path(__file__).parent.parent.parent / "model_ai/models/final/model_best_wp_full.onnx"
_ONNX_PATH = _STEERING_ONNX if _STEERING_ONNX.exists() else _WP_FULL_ONNX

_session = None
_input_name = None
_output_name = None


def _load_session():
    global _session, _input_name, _output_name
    if _session is not None:
        return

    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime chưa cài. Chạy: pip install onnxruntime")

    if not _ONNX_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy ONNX model: {_ONNX_PATH}")

    _session     = ort.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])
    _input_name  = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name


def predict(x: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    x : np.ndarray, shape (batch, lookback, n_features), dtype float32

    Returns
    -------
    np.ndarray, shape (batch, 4) — [lat_24h, lon_24h, lat_48h, lon_48h] (scaled)
    """
    _load_session()
    return _session.run([_output_name], {_input_name: x.astype(np.float32)})[0]


def is_ready() -> bool:
    try:
        _load_session()
        return True
    except Exception:
        return False


def rolling_predict(track: list[dict], cutoff_index: int, max_steps: int = 20) -> list[dict]:
    """
    Rolling prediction từ cutoff_index, mỗi bước tiến 24h.

    Dùng kết quả dự đoán +24h làm input cho bước tiếp theo,
    tạo ra cả đường đi dự đoán qua SCS.

    Parameters
    ----------
    track        : full track (list of dict với lat, lon, iso_time, vmax, pmin)
    cutoff_index : index điểm bão vào SCS
    max_steps    : số bước dự đoán tối đa

    Returns
    -------
    list of dict — các điểm dự đoán từ cutoff trở đi
    """
    import pandas as pd
    from services.preprocessor import prepare_input, decode_output, _load_scaler

    _load_session()
    scaler = _load_scaler()

    # Số bước cần dự đoán ≈ độ dài phần SCS thực tế (chia 4 vì mỗi bước = 24h = 4×6h)
    remaining = len(track) - cutoff_index
    target_steps = min(max(remaining // 4, 3), max_steps)

    # Khởi tạo cửa sổ với track thực trước cutoff
    current = [dict(p) for p in track[:cutoff_index]]

    # Lấy vmax/pmin cuối cùng để carry-forward
    last_vmax = current[-1].get("vmax") or 35.0
    last_pmin = current[-1].get("pmin") or 1000.0
    try:
        last_time = pd.Timestamp(current[-1].get("iso_time") or "2021-01-01T00:00:00")
    except Exception:
        last_time = pd.Timestamp("2021-01-01")

    predicted = []

    for _ in range(target_steps):
        if len(current) < 8:
            break
        try:
            x      = prepare_input(current)
            pred   = predict(x)
            coords = decode_output(pred, x, scaler)
            lat_24h, lon_24h = float(coords[0]), float(coords[1])
        except Exception:
            break

        # Interpolate 4 × 6h points vào `current` để giữ đúng 6h spacing
        # (model được train với điểm 6h — không append thẳng điểm 24h)
        lat0 = current[-1]["lat"]
        lon0 = current[-1]["lon"]
        for step in range(1, 5):
            frac = step / 4
            interp_time = last_time + pd.Timedelta(hours=6 * step)
            current.append({
                "lat":      lat0 + frac * (lat_24h - lat0),
                "lon":      lon0 + frac * (lon_24h - lon0),
                "iso_time": interp_time.strftime("%Y-%m-%d %H:%M:%S"),
                "vmax":     last_vmax,
                "pmin":     last_pmin,
            })

        last_time = last_time + pd.Timedelta(hours=24)

        # Chỉ lưu điểm +24h vào predicted để hiển thị
        predicted.append({
            "lat":      round(lat_24h, 4),
            "lon":      round(lon_24h, 4),
            "iso_time": last_time.strftime("%Y-%m-%d %H:%M:%S"),
            "vmax":     last_vmax,
            "pmin":     last_pmin,
        })

    return predicted
