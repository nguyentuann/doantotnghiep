"""
train_bilstm.py — G6: Train BiLSTM + Attention
------------------------------------------------
Chạy SAU KHI train_lstm.py đã pass checkpoint G5.

Chạy:
    cd source/model_ai
    python -m src.g5_g6_train.train_bilstm              # 14 features (mặc định)
    python -m src.g5_g6_train.train_bilstm --tag 14feat # 14 features (tường minh)
    python -m src.g5_g6_train.train_bilstm --tag ""     # 12 features (legacy)

Checkpoint G6: cải thiện >= 10% Val MAE 24h so với LSTM baseline
Output: models/checkpoints/best_bilstm_{tag}.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.trainer import run_training
from src.g5_g6_train.utils import save_checkpoint, load_scaler, load_lstm_baseline_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="14feat",
                        help="Tag phân biệt phiên bản (default: 14feat)")
    args = parser.parse_args()
    tag  = args.tag

    cfg      = load_config()
    base_dir = Path(__file__).parent.parent.parent
    suffix   = f"_{tag}" if tag else ""

    # --- Kiểm tra LSTM baseline cùng tag ---
    lstm_mae = load_lstm_baseline_mae(base_dir, cfg, tag=tag)
    if lstm_mae is None:
        print(f"[G6] WARN: Chưa có kết quả lstm_{tag} trong results_log.json.")
        print(f"      Chạy train_lstm.py --tag {tag} trước.")
        print("      Tiếp tục train BiLSTM nhưng không thể tính so sánh.")
    else:
        print(f"[G6] LSTM{suffix} baseline MAE 24h = {lstm_mae:.1f} km")

    # --- Load sequences ---
    seq_path = base_dir / f"data/features/sequences{suffix}.npz"
    print(f"[G6-BiLSTM] Load sequences: {seq_path.name}")
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
        scaler=scaler,
        tag=tag,
    )

    # --- Checkpoint G6 ---
    mae_24h     = result["best_mae_24h"]
    min_improve = cfg["checkpoints"]["g6_min_improvement_pct"]  # 10%

    print(f"\n{'='*55}")
    if lstm_mae is not None:
        improve_pct = (lstm_mae - mae_24h) / lstm_mae * 100
        print(f"  BiLSTM{suffix} MAE 24h : {mae_24h:.1f} km")
        print(f"  LSTM{suffix}   MAE 24h : {lstm_mae:.1f} km")
        print(f"  Cải thiện              : {improve_pct:+.1f}%")

        if improve_pct >= min_improve:
            print(f"  [G6] PASS: BiLSTM cải thiện {improve_pct:.1f}% >= {min_improve}%")
            final_path = base_dir / f"models/final/bilstm_best{suffix}.pt"
            save_checkpoint(model, str(final_path))
        else:
            print(f"  [G6] FAIL: Cải thiện {improve_pct:.1f}% < {min_improve}%")
            print("  Gợi ý: Tăng hidden_size hoặc kiểm tra attention weights")
    else:
        print(f"  BiLSTM{suffix} MAE 24h = {mae_24h:.1f} km (không có LSTM để so sánh)")
    print(f"{'='*55}")

    return result


if __name__ == "__main__":
    main()
