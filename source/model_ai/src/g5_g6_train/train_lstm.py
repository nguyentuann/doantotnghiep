"""
train_lstm.py — G5: Train LSTM Baseline
-----------------------------------------
Chạy:
    cd source/model_ai
    python -m src.g5_g6_train.train_lstm

Checkpoint G5: Val MAE 24h < 200 km
Output: models/checkpoints/best_lstm.pt
"""

import torch
import numpy as np
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.trainer import run_training
from src.g5_g6_train.utils import save_checkpoint, load_scaler


def main():
    cfg      = load_config()
    base_dir = Path(__file__).parent.parent.parent

    # --- Load sequences đã chuẩn bị từ G3 ---
    print("[G5] Load sequences...")
    data     = np.load(base_dir / "data/features/sequences.npz")
    X_train  = data["X_train"]   # (22567, 8, 12)
    y_train  = data["y_train"]   # (22567, 4)
    X_val    = data["X_val"]     # (2557, 8, 12)
    y_val    = data["y_val"]     # (2557, 4)
    print(f"  X_train={X_train.shape}  X_val={X_val.shape}")

    scaler = load_scaler(cfg, base_dir)

    # --- Khởi tạo model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("lstm", cfg)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Train ---
    result = run_training(
        model=model,
        model_name="lstm",
        cfg=cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        scaler=scaler,
    )

    # --- Checkpoint G5 ---
    max_mae = cfg["checkpoints"]["g5_lstm_max_mae_km"]   # 200 km
    mae_24h = result["best_mae_24h"]

    print(f"\n{'='*55}")
    if mae_24h <= max_mae:
        print(f"  [G5] PASS: Val MAE 24h = {mae_24h:.1f} km <= {max_mae} km")

        # Lưu thêm vào models/final/ để G6 có thể đọc kết quả baseline
        final_path = base_dir / "models/final/lstm_baseline.pt"
        save_checkpoint(model, str(final_path))
        print(f"  LSTM baseline lưu tại: {final_path}")
    else:
        print(f"  [G5] FAIL: Val MAE 24h = {mae_24h:.1f} km > {max_mae} km")
        print("  Gợi ý:")
        print("    - Giảm learning rate: cfg['model']['lr'] = 0.0005")
        print("    - Tăng dropout nếu overfit (train loss << val loss)")
        print("    - Kiểm tra lại features (NaN, scale)")
    print(f"{'='*55}")

    return result


if __name__ == "__main__":
    main()
