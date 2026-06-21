# Sinh _embedded.py: nhúng artifact (onnx/scaler/json) dạng base64 vào Python
# để Vercel luôn bundle (code Python luôn được đóng gói, khác file data).
import base64, glob, os

ROOT = os.path.join(os.path.dirname(__file__), "backend", "model_ai")
_ARCH_ONNX = [
    "models/final/model_best_scs_v12_lb6_s42.onnx",        # transformer (tên cũ)
    "models/final/model_best_scs_v12_lb6_s42.onnx.data",
    "models/final/model_best_transformer_scs_v12_lb6_s42.onnx",      # transformer (tên mới)
    "models/final/model_best_transformer_scs_v12_lb6_s42.onnx.data",
    "models/final/model_best_lstm_scs_v12_lb6_s42.onnx",
    "models/final/model_best_lstm_scs_v12_lb6_s42.onnx.data",
    "models/final/model_best_bilstm_scs_v12_lb6_s42.onnx",
    "models/final/model_best_bilstm_scs_v12_lb6_s42.onnx.data",
    "models/final/model_best_bigru_scs_v12_lb6_s42.onnx",
    "models/final/model_best_bigru_scs_v12_lb6_s42.onnx.data",
]
# Chỉ nhúng các file thực sự tồn tại
files = [f for f in _ARCH_ONNX if os.path.exists(os.path.join(ROOT, f))]
files += [
    "models/scaler_scs_v12_lb6.npz",
    "models/scaler_scs_v12_lb6.pkl",
    "results/results_log.json",
]
for p in (glob.glob(os.path.join(ROOT, "results/figures/*/per_storm_metrics.json"))
          + glob.glob(os.path.join(ROOT, "results/figures/*/feature_importance.json"))):
    files.append(os.path.relpath(p, ROOT).replace(os.sep, "/"))

lines = ["# AUTO-GENERATED — artifact nhung base64 (Vercel khong bundle file data dong).",
         "FILES = {"]
total = 0
for rel in files:
    data = open(os.path.join(ROOT, rel), "rb").read()
    total += len(data)
    lines.append("  %r: %r," % (rel, base64.b64encode(data).decode()))
lines.append("}")

out = os.path.join(os.path.dirname(__file__), "backend", "_embedded.py")
open(out, "w", encoding="utf-8").write("\n".join(lines))
print("embedded %d files | raw %.2f MB | _embedded.py %.2f MB"
      % (len(files), total / 1e6, os.path.getsize(out) / 1e6))
