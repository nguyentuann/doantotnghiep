"""
train_lstm.py — G5: Train LSTM Baseline
-----------------------------------------
Chạy:
    cd source/model_ai
    python -m src.g5_g6_train.train_lstm              # 14 features (mặc định)
    python -m src.g5_g6_train.train_lstm --tag 14feat # 14 features (tường minh)
    python -m src.g5_g6_train.train_lstm --tag ""     # 12 features (legacy)

Checkpoint G5: Val MAE 24h < 200 km
Output: models/checkpoints/best_lstm_{tag}.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.trainer import run_training
from src.g5_g6_train.utils import save_checkpoint, load_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="14feat",
                        help="Tag phân biệt phiên bản (default: 14feat)")
    args = parser.parse_args()
    tag  = args.tag

    cfg      = load_config()
    base_dir = Path(__file__).parent.parent.parent
    suffix   = f"_{tag}" if tag else ""

    # --- Load sequences ---
    seq_path = base_dir / f"data/features/sequences{suffix}.npz"
    print(f"[G5] Load sequences: {seq_path.name}")
    data    = np.load(seq_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    print(f"  X_train={X_train.shape}  X_val={X_val.shape}")

    # Tự động detect n_features từ sequences (hỗ trợ 12/14/N features)
    cfg["features"]["n_features"] = X_train.shape[2]

    scaler = load_scaler(cfg, base_dir, tag=tag)

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
        tag=tag,
    )

    # --- Checkpoint G5 ---
    max_mae = cfg["checkpoints"]["g5_lstm_max_mae_km"]   # 200 km
    mae_24h = result["best_mae_24h"]

    print(f"\n{'='*55}")
    if mae_24h <= max_mae:
        print(f"  [G5] PASS: Val MAE 24h = {mae_24h:.1f} km <= {max_mae} km")
        final_path = base_dir / f"models/final/lstm_baseline{suffix}.pt"
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
