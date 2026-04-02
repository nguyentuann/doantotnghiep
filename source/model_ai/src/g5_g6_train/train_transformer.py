"""
train_transformer.py — G6: Train Temporal Transformer
-------------------------------------------------------
Chạy SAU KHI train_lstm.py đã pass checkpoint G5.

Chạy:
    cd source/model_ai
    python -m src.g5_g6_train.train_transformer

Checkpoint G6: cải thiện >= 10% Val MAE 24h so với LSTM baseline
Output: models/checkpoints/best_transformer.pt
"""

import json
import torch
import numpy as np
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.trainer import run_training
from src.g5_g6_train.utils import save_checkpoint, load_scaler, load_lstm_baseline_mae


def main():
    cfg      = load_config()
    base_dir = Path(__file__).parent.parent.parent

    # --- Kiểm tra LSTM baseline ---
    lstm_mae = load_lstm_baseline_mae(base_dir, cfg)
    if lstm_mae is None:
        print("[G6] WARN: Chưa có kết quả LSTM trong results_log.json.")
        print("      Chạy train_lstm.py trước để có baseline.")
    else:
        print(f"[G6] LSTM baseline MAE 24h = {lstm_mae:.1f} km")

    # --- Load sequences ---
    print("[G6-Transformer] Load sequences...")
    data    = np.load(base_dir / "data/features/sequences.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    print(f"  X_train={X_train.shape}  X_val={X_val.shape}")

    scaler = load_scaler(cfg, base_dir)

    # --- Khởi tạo model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("transformer", cfg)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Train ---
    result = run_training(
        model=model,
        model_name="transformer",
        cfg=cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        scaler=scaler,
    )

    # --- Checkpoint G6 ---
    mae_24h     = result["best_mae_24h"]
    min_improve = cfg["checkpoints"]["g6_min_improvement_pct"]  # 10%

    print(f"\n{'='*55}")
    if lstm_mae is not None:
        improve_pct = (lstm_mae - mae_24h) / lstm_mae * 100
        print(f"  Transformer MAE 24h : {mae_24h:.1f} km")
        print(f"  LSTM        MAE 24h : {lstm_mae:.1f} km")
        print(f"  Cải thiện           : {improve_pct:+.1f}%")

        if improve_pct >= min_improve:
            print(f"  [G6] PASS: Transformer cải thiện {improve_pct:.1f}% >= {min_improve}%")
            final_path = base_dir / "models/final/transformer_best.pt"
            save_checkpoint(model, str(final_path))
        else:
            print(f"  [G6] FAIL: Cải thiện {improve_pct:.1f}% < {min_improve}%")
            print("  Gợi ý: Tăng d_model hoặc num_encoder_layers trong factory.py")
    else:
        print(f"  Transformer MAE 24h = {mae_24h:.1f} km (không có LSTM để so sánh)")
    print(f"{'='*55}")

    # --- So sánh tổng hợp nếu đã có cả BiLSTM ---
    _print_comparison(base_dir, cfg)

    return result


def _print_comparison(base_dir: Path, cfg: dict) -> None:
    """In bảng so sánh tất cả model đã train."""
    log_path = base_dir / cfg["output"]["results_log"]
    if not log_path.exists():
        return
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    # Lấy kết quả mới nhất của mỗi model
    latest = {}
    for e in logs:
        latest[e["model_name"]] = e

    if len(latest) < 2:
        return

    print(f"\n{'─'*55}")
    print("  Tổng hợp kết quả:")
    print(f"  {'Model':<15} {'MAE 24h':>10} {'MAE 48h':>10} {'Epochs':>8}")
    print(f"  {'─'*46}")
    for name, e in sorted(latest.items()):
        print(f"  {name:<15} {e['best_mae_24h']:>9.1f}km {e['best_mae_48h']:>9.1f}km {e['epochs_trained']:>8}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
