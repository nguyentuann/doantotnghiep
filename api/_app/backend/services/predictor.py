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

from _artifacts import model_ai_dir
_BASE = model_ai_dir() / "models/final"

DEFAULT_ARCH = "transformer"

_ARCH_CANDIDATES = {
    "transformer": [
        _BASE / "model_best_transformer_scs_v12_lb6_s42.onnx",
        _BASE / "model_best_scs_v12_lb6_s42.onnx",
        _BASE / "model_best_scs_v11_lb6_s42.onnx",
        _BASE / "model_best_scs_v7.onnx",
        _BASE / "model_best.onnx",
    ],
    "lstm": [
        _BASE / "model_best_lstm_scs_v12_lb6_s42.onnx",
        _BASE / "model_best_lstm_scs_v11_lb6_s42.onnx",
    ],
    "bilstm": [
        _BASE / "model_best_bilstm_scs_v12_lb6_s42.onnx",
        _BASE / "model_best_bilstm_scs_v11_lb6_s42.onnx",
    ],
    "bigru": [
        _BASE / "model_best_bigru_scs_v12_lb6_s42.onnx",
        _BASE / "model_best_bigru_scs_v11_lb6_s42.onnx",
    ],
}

_sessions: dict = {}


def _find_onnx(arch: str):
    for p in _ARCH_CANDIDATES.get(arch, []):
        if p.exists():
            return p
    raise FileNotFoundError(f"Không tìm thấy ONNX cho arch='{arch}'")


def _load_session(arch: str = DEFAULT_ARCH):
    if arch in _sessions:
        return _sessions[arch]
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime chưa cài")
    onnx_path   = _find_onnx(arch)
    sess        = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_shape = sess.get_inputs()[0].shape
    lookback    = int(input_shape[1]) if len(input_shape) >= 2 and isinstance(input_shape[1], int) else 8
    n_outputs   = sess.get_outputs()[0].shape[1]
    print(f"[predictor] ONNX loaded ({arch}): {onnx_path.name} (out={n_outputs}, lb={lookback})")
    _sessions[arch] = {
        "session":     sess,
        "input_name":  sess.get_inputs()[0].name,
        "output_name": sess.get_outputs()[0].name,
        "n_outputs":   n_outputs,
        "lookback":    lookback,
    }
    return _sessions[arch]


def predict(x: np.ndarray, arch: str = DEFAULT_ARCH) -> np.ndarray:
    s = _load_session(arch)
    return s["session"].run([s["output_name"]], {s["input_name"]: x.astype(np.float32)})[0]


def available_archs() -> list:
    return [a for a, cands in _ARCH_CANDIDATES.items() if any(p.exists() for p in cands)]


def is_ready(arch: str = DEFAULT_ARCH) -> bool:
    try:
        _load_session(arch)
        return True
    except Exception:
        return False


def rolling_predict(track: list[dict], cutoff_index: int, arch: str = DEFAULT_ARCH, max_steps: int = 40) -> list[dict]:
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
    from datetime import datetime, timedelta
    from services.preprocessor import prepare_input, decode_output, _load_scaler

    sess_info  = _load_session(arch)
    scaler     = _load_scaler()
    _lookback  = sess_info["lookback"]
    _n_outputs = sess_info["n_outputs"]
    is_sprint1 = (_n_outputs == 16)

    current = [dict(p) for p in track[:cutoff_index]]
    if len(current) < _lookback:
        return []

    last_vmax = current[-1].get("vmax") or 35.0
    last_pmin = current[-1].get("pmin") or 1000.0
    try:
        last_time = datetime.fromisoformat(
            str(current[-1].get("iso_time") or "2021-01-01T00:00:00").replace("Z", "").replace("z", ""))
    except Exception:
        last_time = datetime(2021, 1, 1)

    predicted = []

    if is_sprint1:
        # Sprint1: 1 lần gọi model → lấy trực tiếp 8 bước output (coords[0..15])
        # Tránh rolling error accumulation do velocity features bị sai sau mỗi bước
        try:
            x      = prepare_input(current, lookback=_lookback)
            pred   = predict(x, arch=arch)
            coords = decode_output(pred, x, scaler)  # shape (16,)
        except Exception as e:
            print(f"[rolling_predict] error: {e}")
            return []

        for step in range(8):  # 8 bước × 6h = 48h
            lat_pred = float(coords[step * 2])
            lon_pred = float(coords[step * 2 + 1])
            t_pred   = last_time + timedelta(hours=6 * (step + 1))
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
                pred   = predict(x, arch=arch)
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
                interp_time = last_time + timedelta(hours=6 * step)
                current.append({
                    "lat":      lat0 + frac * (lat_24h - lat0),
                    "lon":      lon0 + frac * (lon_24h - lon0),
                    "iso_time": interp_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "vmax":     last_vmax,
                    "pmin":     last_pmin,
                })

            last_time = last_time + timedelta(hours=24)
            predicted.append({
                "lat":      round(lat_24h, 4),
                "lon":      round(lon_24h, 4),
                "iso_time": last_time.strftime("%Y-%m-%d %H:%M:%S"),
                "vmax":     last_vmax,
                "pmin":     last_pmin,
            })

    return predicted
