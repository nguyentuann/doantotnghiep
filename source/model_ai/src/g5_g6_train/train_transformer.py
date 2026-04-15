"""
train_transformer.py — G6: Train Temporal Transformer
-------------------------------------------------------
Chạy SAU KHI train_lstm.py đã pass checkpoint G5.

Chạy:
    cd source/model_ai
    python -m src.g5_g6_train.train_transformer              # 14 features (mặc định)
    python -m src.g5_g6_train.train_transformer --tag 14feat # 14 features (tường minh)
    python -m src.g5_g6_train.train_transformer --tag ""     # 12 features (legacy)

Checkpoint G6: cải thiện >= 10% Val MAE 24h so với LSTM baseline
Output: models/checkpoints/best_transformer_{tag}.pt
"""

import argparse
import json
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
    else:
        print(f"[G6] LSTM{suffix} baseline MAE 24h = {lstm_mae:.1f} km")

    # --- Load sequences ---
    seq_path = base_dir / f"data/features/sequences{suffix}.npz"
    print(f"[G6-Transformer] Load sequences: {seq_path.name}")
    data    = np.load(seq_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val   = data["X_val"]
    y_val   = data["y_val"]
    print(f"  X_train={X_train.shape}  X_val={X_val.shape}")

    scaler = load_scaler(cfg, base_dir, tag=tag)

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
        tag=tag,
    )

    # --- Checkpoint G6 ---
    mae_24h     = result["best_mae_24h"]
    min_improve = cfg["checkpoints"]["g6_min_improvement_pct"]  # 10%

    print(f"\n{'='*55}")
    if lstm_mae is not None:
        improve_pct = (lstm_mae - mae_24h) / lstm_mae * 100
        print(f"  Transformer{suffix} MAE 24h : {mae_24h:.1f} km")
        print(f"  LSTM{suffix}        MAE 24h : {lstm_mae:.1f} km")
        print(f"  Cải thiện                   : {improve_pct:+.1f}%")

        if improve_pct >= min_improve:
            print(f"  [G6] PASS: Transformer cải thiện {improve_pct:.1f}% >= {min_improve}%")
            final_path = base_dir / f"models/final/transformer_best{suffix}.pt"
            save_checkpoint(model, str(final_path))
        else:
            print(f"  [G6] FAIL: Cải thiện {improve_pct:.1f}% < {min_improve}%")
            print("  Gợi ý: Tăng d_model hoặc num_encoder_layers trong factory.py")
    else:
        print(f"  Transformer{suffix} MAE 24h = {mae_24h:.1f} km (không có LSTM để so sánh)")
    print(f"{'='*55}")

    # --- So sánh tổng hợp sau khi train xong ---
    _print_comparison(base_dir, cfg, tag=tag)

    return result


def _print_comparison(base_dir: Path, cfg: dict, tag: str = "") -> None:
    """In bảng so sánh các model cùng tag đã train."""
    log_path = base_dir / cfg["output"]["results_log"]
    if not log_path.exists():
        return
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    suffix  = f"_{tag}" if tag else ""
    targets = {f"lstm{suffix}", f"bilstm{suffix}", f"transformer{suffix}"}

    latest = {}
    for e in logs:
        if e["model_name"] in targets and "best_mae_24h" in e:
            latest[e["model_name"]] = e

    if len(latest) < 2:
        return

    print(f"\n{'─'*60}")
    print(f"  Tổng hợp kết quả (tag='{tag}'):")
    print(f"  {'Model':<22} {'MAE 24h':>10} {'MAE 48h':>10} {'Epochs':>8}")
    print(f"  {'─'*52}")
    for name, e in sorted(latest.items()):
        print(f"  {name:<22} {e['best_mae_24h']:>9.1f}km {e['best_mae_48h']:>9.1f}km {e['epochs_trained']:>8}")
    print(f"{'─'*60}")


if __name__ == "__main__":
    main()
