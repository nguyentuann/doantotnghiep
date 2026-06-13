"""
predictor.py — ONNX Runtime inference + rolling prediction

scs_v12_lb6 model (BEST 2026-05-21): input (batch, 6, 31), output (batch, 16)
                     Transformer 3-seed, SCS-only training, 31 features
                     + asteer_u/v (DLM annulus) + asteer_u850/v850 (low-level annulus)
                     MAE 24h = 101.3 km, Skill = 42.0% vs CLIPER (174.6 km)

Rolling strategy:
  - Sprint1 (16 out): advance 6h/step, output all 6h points
  - Legacy  ( 4 out): advance 24h/step, output 24h points
"""

import numpy as np
from pathlib import Path

_BASE = Path(__file__).parent.parent.parent / "model_ai/models/final"

# Priority: scs_v12_lb6 (BEST hiện tại) → scs_v11_lb6 → scs_v9_lb6 → ...
_ONNX_CANDIDATES = [
    _BASE / "model_best_scs_v12_lb6.onnx",          # 31 feat, lookback=6, 42.0% skill
    _BASE / "model_best_scs_v12_lb6_s42.onnx",      # single seed fallback
    _BASE / "model_best_scs_v11_lb6.onnx",          # 29 feat, lookback=6, 41.1%
    _BASE / "model_best_scs_v11_lb6_s42.onnx",      # single seed fallback
    _BASE / "model_best_scs_v9_lb6.onnx",           # 27 feat, lookback=6, 36.4%
    _BASE / "model_best_scs_v9.onnx",               # 27 feat, lookback=8, 34.5%
    _BASE / "model_best_scs_v7.onnx",               # 25 feat, lookback=8, 33.9%
    _BASE / "model_best_wp_6h_v7.onnx",
    _BASE / "model_best_wp_6h_v6.onnx",
    _BASE / "model_best_wp_6h_v5.onnx",
    _BASE / "model_best_wp_6h_ext.onnx",
    _BASE / "model_best_wp_6h_enso.onnx",
    _BASE / "model_best_wp_6h_v3.onnx",
    _BASE / "model_best_wp_6h_v2.onnx",
    _BASE / "model_best_wp_6h.onnx",
    _BASE / "model_best_sprint1_6h.onnx",
    _BASE / "model_best_sprint1.onnx",
    _BASE / "model_best_14feat.onnx",
    _BASE / "model_best_wp_full.onnx",
    _BASE / "model_best.onnx",
]
_ONNX_PATH = next((p for p in _ONNX_CANDIDATES if p.exists()), _ONNX_CANDIDATES[-1])

_session = None
_input_name = None
_output_name = None
_n_outputs = None   # 16 (sprint1) or 4 (legacy)
_lookback = 8       # auto-detected from ONNX input shape


def _load_session():
    global _session, _input_name, _output_name, _n_outputs, _lookback
    if _session is not None:
        return

    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime chưa cài. Chạy: pip install onnxruntime")

    if not _ONNX_PATH.exists():
        raise FileNotFoundError(f"ONNX model không tồn tại: {_ONNX_PATH}")

    _session     = ort.InferenceSession(str(_ONNX_PATH), providers=["CPUExecutionProvider"])
    _input_name  = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name
    _n_outputs   = _session.get_outputs()[0].shape[1]
    # Auto-detect lookback từ input shape (batch, lookback, n_features)
    input_shape = _session.get_inputs()[0].shape
    _lookback   = int(input_shape[1]) if len(input_shape) >= 2 and isinstance(input_shape[1], int) else 8
    print(f"[predictor] ONNX loaded: {_ONNX_PATH.name} (output={_n_outputs}, lookback={_lookback})")


def predict(x: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    x : np.ndarray, shape (batch, lookback, n_features), dtype float32

    Returns
    -------
    np.ndarray shape (batch, n_outputs)
    """
    _load_session()
    return _session.run([_output_name], {_input_name: x.astype(np.float32)})[0]


def is_ready() -> bool:
    try:
        _load_session()
        return True
    except Exception:
        return False


def rolling_predict(track: list[dict], cutoff_index: int, max_steps: int = 40) -> list[dict]:
    """
    Dự đoán quỹ đạo từ cutoff_index.

    Sprint1 (16-output): 1 lần gọi model → 8 bước × 6h = 48h trực tiếp.
                         Không rolling để tránh tích lũy sai số velocity.
    Legacy  ( 4-output): rolling 24h/step như cũ.

    Parameters
    ----------
    track        : full track list (lat, lon, iso_time, vmax, pmin)
    cutoff_index : index điểm đầu tiên trong SCS
    max_steps    : giữ để tương thích API, sprint1 luôn trả về 8 bước (48h)
    """
    import pandas as pd
    from services.preprocessor import prepare_input, decode_output, _load_scaler

    _load_session()
    scaler = _load_scaler()
    is_sprint1 = (_n_outputs == 16)

    current = [dict(p) for p in track[:cutoff_index]]
    if len(current) < _lookback:
        return []

    last_vmax = current[-1].get("vmax") or 35.0
    last_pmin = current[-1].get("pmin") or 1000.0
    try:
        last_time = pd.Timestamp(current[-1].get("iso_time") or "2021-01-01T00:00:00")
    except Exception:
        last_time = pd.Timestamp("2021-01-01")

    predicted = []

    if is_sprint1:
        # Sprint1: 1 lần gọi model → lấy trực tiếp 8 bước output (coords[0..15])
        # Tránh rolling error accumulation do velocity features bị sai sau mỗi bước
        try:
            x      = prepare_input(current, lookback=_lookback)
            pred   = predict(x)
            coords = decode_output(pred, x, scaler)  # shape (16,)
        except Exception as e:
            print(f"[rolling_predict] error: {e}")
            return []

        for step in range(8):  # 8 bước × 6h = 48h
            lat_pred = float(coords[step * 2])
            lon_pred = float(coords[step * 2 + 1])
            t_pred   = last_time + pd.Timedelta(hours=6 * (step + 1))
            predicted.append({
                "lat":      round(lat_pred, 4),
                "lon":      round(lon_pred, 4),
                "iso_time": t_pred.strftime("%Y-%m-%d %H:%M:%S"),
                "vmax":     last_vmax,
                "pmin":     last_pmin,
            })
        return predicted

    else:
        # Legacy: rolling 24h/step
        target_steps = min(max(len(track) - cutoff_index // 4, 3), max_steps // 4)
        for _ in range(target_steps):
            if len(current) < 8:
                break
            try:
                x      = prepare_input(current)
                pred   = predict(x)
                coords = decode_output(pred, x, scaler)
            except Exception as e:
                print(f"[rolling_predict] error: {e}")
                break

            # Legacy: coords[0], coords[1] = +24h lat, lon
            lat_24h = float(coords[0])
            lon_24h = float(coords[1])

            # Interpolate 4×6h points into window
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
            predicted.append({
                "lat":      round(lat_24h, 4),
                "lon":      round(lon_24h, 4),
                "iso_time": last_time.strftime("%Y-%m-%d %H:%M:%S"),
                "vmax":     last_vmax,
                "pmin":     last_pmin,
            })

    return predicted
