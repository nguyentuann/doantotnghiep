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
import random
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
    parser.add_argument("--hem-from", default=None,
                        help="Path tới checkpoint để compute hard-example weights. "
                             "Nếu set, dùng WeightedRandomSampler cho training set.")
    parser.add_argument("--hem-alpha", type=float, default=0.5,
                        help="Aggressiveness HEM weighting: w = err^alpha (default 0.5)")
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

    # Tự động detect n_features, lookback và output_size từ sequences
    # (Transformer pos_embed phụ thuộc lookback — bắt buộc set trước build_model)
    cfg["features"]["n_features"] = X_train.shape[2]
    cfg["model"]["lookback"]      = X_train.shape[1]
    cfg["model"]["output_size"]   = y_train.shape[1]

    # Sprint 1: giữ residual=True cho Transformer khi dùng multi-horizon output
    # residual=False chỉ ổn với 2-anchor output (4 values), không ổn với 16 values

    scaler = load_scaler(cfg, base_dir, tag=tag)

    # --- Khởi tạo model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = build_model("transformer", cfg)
    print(f"  Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Init from pretrained (transfer learning) ---
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"--init-from không tồn tại: {init_path}")
        state_dict = torch.load(str(init_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"  [init] Loaded pretrained weights from: {init_path.name}")

    # --- Hard Example Mining: compute weights từ checkpoint hiện tại ---
    hem_weights = None
    if args.hem_from:
        from src.g7_evaluate.evaluate import _build_model_from_state
        from src.g5_g6_train.trainer import cliper_from_X
        from src.g7_evaluate.evaluate import haversine_km

        hem_path = Path(args.hem_from)
        if not hem_path.exists():
            raise FileNotFoundError(f"--hem-from không tồn tại: {hem_path}")
        print(f"\n  [HEM] Compute weights from: {hem_path.name}")
        sd = torch.load(str(hem_path), map_location="cpu", weights_only=True)
        hem_model, _ = _build_model_from_state(sd, "transformer", cfg)
        hem_model.to(device).eval()

        anchor_steps = cfg["model"].get("anchor_steps", [4, 8])
        residual = cfg["model"].get("residual", False)
        c_train = cliper_from_X(X_train, scaler, anchor_steps=anchor_steps) if residual else None

        # Predict trên toàn train set (batch để khỏi OOM)
        with torch.no_grad():
            X_t = torch.from_numpy(X_train).float()
            preds = []
            B = 256
            for i in range(0, len(X_t), B):
                xb = X_t[i:i+B].to(device)
                out = hem_model(xb).cpu().numpy()
                if residual:
                    out = c_train[i:i+B] + out
                preds.append(out)
            preds = np.concatenate(preds, axis=0)

        # Compute error 24h per sample (step 4 = idx 6,7 trong multi-horizon flat)
        err_24 = haversine_km(preds[:, 6], preds[:, 7],
                              y_train[:, 6], y_train[:, 7])
        # Weight = err^alpha — clip tránh extreme weights
        weights = np.clip(err_24, 1.0, None) ** args.hem_alpha
        weights = weights / weights.mean()  # normalize
        hem_weights = weights.astype(np.float32)
        print(f"  [HEM] alpha={args.hem_alpha}, err_24 range [{err_24.min():.1f}, {err_24.max():.1f}] km, "
              f"weight range [{weights.min():.3f}, {weights.max():.3f}]")
        del hem_model

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
        lr_scale=args.lr_scale,
        max_epochs_override=args.max_epochs,
        patience_override=args.patience,
        seed_suffix=seed_suffix,
        hem_weights=hem_weights,
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
