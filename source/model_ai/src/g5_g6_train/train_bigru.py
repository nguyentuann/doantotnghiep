"""
train_bigru.py — G6: Train BiGRU + Attention
----------------------------------------------
Kiến trúc BiGRU + Attention (Song et al. 2022) — GRU ít tham số hơn LSTM,
hội tụ nhanh hơn, thường tốt hơn BiLSTM trong track prediction.

Chạy:
    python -m src.g5_g6_train.train_bigru --tag wp_6h_v6

Output: models/checkpoints/best_bigru_{tag}.pt
"""

import argparse
import random
import torch
import numpy as np
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.trainer import run_training
from src.g5_g6_train.utils import save_checkpoint, load_scaler, load_lstm_baseline_mae


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="14feat")
    parser.add_argument("--init-from", default=None,
                        help="Path tới pretrained checkpoint .pt để init weights (transfer learning)")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="Scale LR (dùng 0.1-0.3 cho fine-tuning)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Override max_epochs từ config")
    parser.add_argument("--patience", type=int, default=None,
                        help="Override early stopping patience")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed cho multi-seed ensemble (vd: 42, 123, 2024)")
    args = parser.parse_args()
    tag  = args.tag

    seed_suffix = ""
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        seed_suffix = f"_s{args.seed}"
        print(f"[seed] random seed = {args.seed}")

    cfg      = load_config()
    base_dir = Path(__file__).parent.parent.parent
    suffix   = f"_{tag}" if tag else ""

    lstm_mae = load_lstm_baseline_mae(base_dir, cfg, tag=tag)
    if lstm_mae is None:
        print(f"[G6] WARN: Chưa có kết quả lstm_{tag}. Tiếp tục không so sánh.")
    else:
        print(f"[G6] LSTM{suffix} baseline MAE 24h = {lstm_mae:.1f} km")

    seq_path = base_dir / f"data/features/sequences{suffix}.npz"
    print(f"[G6-BiGRU] Load sequences: {seq_path.name}")
    data    = np.load(seq_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    print(f"  X_train={X_train.shape}  X_val={X_val.shape}")

    cfg["features"]["n_features"] = X_train.shape[2]
    cfg["model"]["output_size"]   = y_train.shape[1]
    cfg["model"]["lookback"]      = X_train.shape[1]

    scaler = load_scaler(cfg, base_dir, tag=tag)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("bigru", cfg)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from không tồn tại: {init_path}")
        state_dict = torch.load(str(init_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  [init] Loaded pretrained weights from: {init_path.name}")

    result = run_training(
        model=model,
        model_name="bigru",
        cfg=cfg,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        device=device,
        scaler=scaler,
        tag=tag,
        lr_scale=args.lr_scale,
        max_epochs_override=args.max_epochs,
        patience_override=args.patience,
        seed_suffix=seed_suffix,
    )

    mae_24h     = result["best_mae_24h"]
    min_improve = cfg["checkpoints"]["g6_min_improvement_pct"]

    print(f"\n{'='*55}")
    if lstm_mae is not None:
        improve_pct = (lstm_mae - mae_24h) / lstm_mae * 100
        print(f"  BiGRU{suffix} MAE 24h : {mae_24h:.1f} km")
        print(f"  LSTM{suffix}  MAE 24h : {lstm_mae:.1f} km")
        print(f"  Cải thiện             : {improve_pct:+.1f}%")
        if improve_pct >= min_improve:
            print(f"  [G6] PASS: BiGRU cải thiện {improve_pct:.1f}% >= {min_improve}%")
            final_path = base_dir / f"models/final/bigru_best{suffix}.pt"
            save_checkpoint(model, str(final_path))
        else:
            print(f"  [G6] INFO: Cải thiện {improve_pct:.1f}% < {min_improve}%")
    else:
        print(f"  BiGRU{suffix} MAE 24h = {mae_24h:.1f} km")
    print(f"{'='*55}")

    return result


if __name__ == "__main__":
    main()
