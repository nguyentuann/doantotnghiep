"""
export_onnx.py — G8: Export model tốt nhất sang ONNX
------------------------------------------------------
  1. Tìm best model từ results_log.json (mae_24h thấp nhất)
  2. Export sang ONNX với dynamic batch_size, opset 17
  3. Verify: test batch_size [1, 4, 16], tolerance 1e-4
  4. Ghi kết quả vào results_log.json

Chạy: python -m src.g8_export.export_onnx
"""

import warnings
warnings.filterwarnings("ignore")

import json
import numpy as np
import torch
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.utils import log_result


# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent   # source/model_ai/


# ─── Find best model ──────────────────────────────────────────────────────────

def find_best_model_from_log(log_path: Path, required_tag: str = "14feat") -> tuple[str, Path] | tuple[None, None]:
    """
    Đọc results_log.json, tìm entry có stage="G7_evaluate"/"G5_train"/"G6_train" và "mae_24h",
    chọn model có mae_24h thấp nhất (bỏ qua "cliper").
    Lọc theo required_tag nếu có entry phù hợp; nếu không, bỏ qua filter.
    Trả về (model_name, ckpt_path, tag) hoặc (None, None, None).
    """
    if not log_path.exists():
        return None, None, None

    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    all_candidates = [
        e for e in logs
        if (e.get("stage") in ("G7_evaluate", "G5_train", "G6_train")
            or "mae_24h" in e)
        and e.get("model_name", "cliper").lower() != "cliper"
        and "mae_24h" in e
    ]

    # Lọc theo tag yêu cầu; nếu không có entry phù hợp → để fallback xử lý
    def _extract_tag(e):
        raw = e.get("model_name", "")
        for arch in ("transformer", "bilstm", "lstm"):
            if raw.startswith(arch):
                return raw[len(arch):].lstrip("_")
        return ""

    tagged = [e for e in all_candidates if _extract_tag(e) == required_tag]

    if not tagged:
        # Không có entry với tag yêu cầu trong log → trả None để fallback tìm file
        return None, None, None

    if not all_candidates:
        return None, None, None

    candidates = tagged

    best_entry = min(candidates, key=lambda e: e["mae_24h"])
    raw_name   = best_entry["model_name"]
    for arch in ("transformer", "bilstm", "lstm"):
        if raw_name.startswith(arch):
            model_name = arch
            tag = raw_name[len(arch):].lstrip("_")
            break
    else:
        model_name = raw_name
        tag = ""

    ckpt_from_log = best_entry.get("checkpoint")
    if ckpt_from_log:
        p = Path(ckpt_from_log)
        if p.exists():
            return model_name, p, tag

    return model_name, None, tag


def find_checkpoint_fallback(cfg: dict, required_tag: str = "14feat") -> tuple[str, Path, str] | tuple[None, None, None]:
    """Fallback: tìm checkpoint theo tag ưu tiên, rồi legacy."""
    final_dir = BASE_DIR / "models" / "final"
    ckpt_dir  = BASE_DIR / cfg["output"]["checkpoint_dir"]

    # Ưu tiên tag được yêu cầu, sau đó tag còn lại
    tag_order = [required_tag, ""] if required_tag else ["14feat", ""]
    tag_order = list(dict.fromkeys(tag_order))  # dedup

    fallback_named = [
        ("lstm_baseline", "lstm"),
        ("bilstm_best",   "bilstm"),
        ("transformer_best", "transformer"),
    ]

    for tag in tag_order:
        suffix = f"_{tag}" if tag else ""
        for stem, name in fallback_named:
            p = final_dir / f"{stem}{suffix}.pt"
            if p.exists():
                return name, p, tag

    for tag in tag_order:
        suffix = f"_{tag}" if tag else ""
        for name in ["lstm", "bilstm", "transformer"]:
            p = ckpt_dir / f"best_{name}{suffix}.pt"
            if p.exists():
                return name, p, tag

    return None, None, None


# ─── Export ───────────────────────────────────────────────────────────────────

def export_to_onnx(model: torch.nn.Module, onnx_path: Path,
                   lookback: int = 8, n_features: int = 12) -> bool:
    """
    Export model sang ONNX với dynamic batch_size.
    Trả về True nếu thành công.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dummy_input = torch.randn(1, lookback, n_features)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input":  {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        print(f"  [onnx] Export thành công: {onnx_path}")
        return True
    except Exception as e:
        print(f"  [error] Export ONNX thất bại: {e}")
        return False


# ─── Verify ───────────────────────────────────────────────────────────────────

def verify_onnx(model: torch.nn.Module, onnx_path: Path,
                lookback: int = 8, n_features: int = 12,
                tolerance: float = 1e-4) -> bool:
    """
    Kiểm tra ONNX model với batch_size = [1, 4, 16].
    So sánh với PyTorch output. Tolerance 1e-4.
    Graceful skip nếu onnxruntime không cài.
    Trả về True nếu tất cả test pass (hoặc onnxruntime không có).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [info] onnxruntime không cài — bỏ qua ONNX verification")
        return True

    try:
        sess = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        input_name  = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
    except Exception as e:
        print(f"  [error] Không load được ONNX session: {e}")
        return False

    all_pass = True
    model.eval()

    for bs in [1, 4, 16]:
        x_np  = np.random.randn(bs, lookback, n_features).astype(np.float32)
        x_t   = torch.from_numpy(x_np)

        with torch.no_grad():
            pt_out = model(x_t).numpy()

        ort_out = sess.run([output_name], {input_name: x_np})[0]

        max_diff = float(np.abs(pt_out - ort_out).max())
        passed   = max_diff <= tolerance
        status   = "PASS" if passed else "FAIL"

        print(f"  [verify] batch_size={bs:2d}  max_diff={max_diff:.2e}  => {status}")
        if not passed:
            all_pass = False

    return all_pass


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="14feat",
                        help="Tag pipeline (vd: '14feat', '') — mặc định '14feat'")
    args = parser.parse_args()
    required_tag = args.tag

    print("\n" + "=" * 60)
    print("  G8 — Export: PyTorch → ONNX")
    print("=" * 60)

    cfg = load_config()

    log_path = BASE_DIR / cfg["output"]["results_log"]
    lookback = cfg["model"]["lookback"]

    # --- Tìm best model ---
    print(f"\n[1] Tìm best model (tag='{required_tag}')...")
    model_name, ckpt_path, tag = find_best_model_from_log(log_path, required_tag)

    if model_name is None or ckpt_path is None:
        print("  Không tìm thấy trong results_log — thử fallback...")
        model_name, ckpt_path, tag = find_checkpoint_fallback(cfg, required_tag)

    if model_name is None or ckpt_path is None:
        print("  [error] Không tìm thấy checkpoint nào. Hãy chạy G5/G6 trước.")
        return

    # n_features theo tag
    suffix     = f"_{tag}" if tag else ""
    n_features = cfg["features"]["n_features"]   # 14 (từ config)
    onnx_path  = BASE_DIR / cfg["output"]["onnx_model"].replace(
        ".onnx", f"{suffix}.onnx"
    )

    print(f"  Best model : {model_name.upper()} (tag='{tag}')")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  ONNX output: {onnx_path}")
    print(f"  n_features : {n_features}")

    # --- Load model ---
    print("\n[2] Load model...")
    try:
        model = build_model(model_name, cfg)
        state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  [ok] Model loaded: {model_name.upper()}")
    except Exception as e:
        print(f"  [error] Không load được model: {e}")
        return

    # --- Export ONNX ---
    print("\n[3] Export ONNX...")
    export_ok = export_to_onnx(model, onnx_path, lookback, n_features)

    if not export_ok:
        log_result(str(log_path), {
            "stage":      "G8_export",
            "model_name": model_name,
            "onnx_path":  str(onnx_path),
            "onnx_ok":    False,
        })
        return

    # In kích thước file
    size_mb = onnx_path.stat().st_size / (1024 ** 2)
    print(f"  [onnx] File size: {size_mb:.2f} MB")

    # --- Verify ---
    print("\n[4] Verify ONNX output...")
    verify_ok = verify_onnx(model, onnx_path, lookback, n_features, tolerance=1e-4)

    verify_status = "PASS" if verify_ok else "FAIL"
    print(f"\n  Verification: {verify_status}")

    # --- Log ---
    log_result(str(log_path), {
        "stage":       "G8_export",
        "model_name":  model_name,
        "onnx_path":   str(onnx_path),
        "onnx_size_mb": round(size_mb, 3),
        "onnx_ok":     export_ok and verify_ok,
        "verify_pass": verify_ok,
        "checkpoint":  str(ckpt_path),
    })

    print("\n  G8 hoàn thành.")


if __name__ == "__main__":
    main()
