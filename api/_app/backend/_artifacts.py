"""
_artifacts.py — Giải nén artifact nhúng (base64) ra thư mục tạm lúc chạy.

Lý do: Vercel KHÔNG bundle file data tham chiếu động (onnx/scaler/json) vào
serverless function. Nhúng base64 trong _embedded.py (module Python → luôn được
bundle) rồi giải nén ra /tmp là cách chắc chắn nhất.
"""
import base64
import tempfile
from pathlib import Path

_ROOT = None


def model_ai_dir() -> Path:
    """Trả về thư mục 'model_ai' (đã giải nén ra /tmp, chỉ làm 1 lần)."""
    global _ROOT
    if _ROOT is not None:
        return _ROOT
    from _embedded import FILES
    base = Path(tempfile.gettempdir()) / "typhoon_model_ai"
    for rel, b64 in FILES.items():
        p = base / rel
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(base64.b64decode(b64))
    _ROOT = base
    return base
