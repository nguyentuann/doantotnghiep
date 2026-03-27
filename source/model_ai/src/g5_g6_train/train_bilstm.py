"""
train_bilstm.py — G6: Train BiLSTM + Attention
------------------------------------------------
Chạy SAU KHI train_lstm.py đã pass checkpoint G5.

Chạy:
    cd source/model_ai
    python -m src.g5_g6_train.train_bilstm

Checkpoint G6: cải thiện >= 10% Val MAE 24h so với LSTM baseline
Output: models/checkpoints/best_bilstm.pt
"""

import json
import torch
import numpy as np
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.trainer import run_training
from src.g5_g6_train.utils import save_checkpoint


def _load_lstm_baseline_mae(base_dir: Path, cfg: dict) -> float | None:
    """Đọc MAE 24h của LSTM từ results_log.json."""
    log_path = base_dir / cfg["output"]["results_log"]
    if not log_path.exists():
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    lstm_entries = [e for e in logs if e.get("model_name") == "lstm"]
    if not lstm_entries:
        return None
    # Lấy entry mới nhất
    return lstm_entries[-1]["best_mae_24h"]


def main():
    cfg      = load_config()
    base_dir = Path(__file__).parent.parent.parent

    # --- Kiểm tra LSTM baseline ---
    lstm_mae = _load_lstm_baseline_mae(base_dir, cfg)
    if lstm_mae is None:
        print("[G6] WARN: Chưa có kết quả LSTM trong results_log.json.")
        print("      Chạy train_lstm.py trước để có baseline.")
        print("      Tiếp tục train BiLSTM nhưng không thể tính Skill Score so sánh.")
    else:
        print(f"[G6] LSTM baseline MAE 24h = {lstm_mae:.1f} km")

    # --- Load sequences ---
    print("[G6-BiLSTM] Load sequences...")
    data    = np.load(base_dir / "data/features/sequences.npz")
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    print(f"  X_train={X_train.shape}  X_val={X_val.shape}")

    # --- Khởi tạo model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("bilstm", cfg)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Train ---
    result = run_training(
        model=model,
        model_name="bilstm",
        cfg=cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
    )

    # --- Checkpoint G6 ---
    mae_24h     = result["best_mae_24h"]
    min_improve = cfg["checkpoints"]["g6_min_improvement_pct"]  # 10%

    print(f"\n{'='*55}")
    if lstm_mae is not None:
        improve_pct = (lstm_mae - mae_24h) / lstm_mae * 100
        print(f"  BiLSTM MAE 24h : {mae_24h:.1f} km")
        print(f"  LSTM   MAE 24h : {lstm_mae:.1f} km")
        print(f"  Cải thiện      : {improve_pct:+.1f}%")

        if improve_pct >= min_improve:
            print(f"  [G6] PASS: BiLSTM cải thiện {improve_pct:.1f}% >= {min_improve}%")
            final_path = base_dir / "models/final/bilstm_best.pt"
            save_checkpoint(model, str(final_path))
        else:
            print(f"  [G6] FAIL: Cải thiện {improve_pct:.1f}% < {min_improve}%")
            print("  Gợi ý: Tăng hidden_size hoặc kiểm tra attention weights")
    else:
        print(f"  BiLSTM MAE 24h = {mae_24h:.1f} km (không có LSTM để so sánh)")
    print(f"{'='*55}")

    return result


if __name__ == "__main__":
    main()
