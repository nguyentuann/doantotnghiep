"""
trainer.py — Training loop dùng chung cho cả 3 model
------------------------------------------------------
Hàm run_training() nhận bất kỳ model nào (LSTM / BiLSTM / Transformer)
và chạy toàn bộ vòng lặp train → val → early stopping → lưu checkpoint.

Hỗ trợ residual learning: model dự đoán delta từ CLIPER,
loss tính trên vị trí tuyệt đối (CLIPER + delta) vs actual.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.g4_models import HaversineLoss, build_loss
from .utils import EarlyStopping, log_result


# ─── CLIPER prediction từ X sequences ────────────────────────────────────────

def cliper_from_X(X: np.ndarray, scaler, anchor_steps: list = None) -> np.ndarray:
    """
    Tính CLIPER (persistence extrapolation) từ X sequences đã scaled.

    Args:
        X            : (N, lookback, n_features) — đã StandardScaler
        scaler       : fitted StandardScaler
        anchor_steps : list bước dự đoán, e.g. [1,2,3,4,5,6,7,8] hoặc [4,8]
                       None → fallback về [4, 8] (legacy)

    Returns:
        (N, len(anchor_steps)*2) — [lat_s1, lon_s1, lat_s2, lon_s2, ...]
    """
    if anchor_steps is None:
        anchor_steps = [4, 8]

    N, lookback, n_feat = X.shape
    last_step_scaled = X[:, -1, :]
    last_step_orig   = scaler.inverse_transform(last_step_scaled)

    # Features 0,1 = lat_norm, lon_norm; 2,3 = dlat, dlon
    lat_current = last_step_orig[:, 0] * 14.0 + 8.0    # lat_norm → [8, 22]
    lon_current = last_step_orig[:, 1] * 18.0 + 102.0  # lon_norm → [102, 120]

    # Dùng velocity trung bình 3 bước cuối — ổn định hơn last-step khi bão tăng/giảm tốc
    n_avg = min(3, lookback)
    last_n_orig = scaler.inverse_transform(
        X[:, -n_avg:, :].reshape(-1, n_feat)
    ).reshape(N, n_avg, n_feat)
    dlat = last_n_orig[:, :, 2].mean(axis=1)
    dlon = last_n_orig[:, :, 3].mean(axis=1)

    cols = []
    for step in anchor_steps:
        cols.append(lat_current + step * dlat)
        cols.append(lon_current + step * dlon)

    return np.stack(cols, axis=1).astype(np.float32)


# ─── 1 epoch train ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: HaversineLoss,
    device: torch.device,
    grad_clip: float = 1.0,
    residual: bool = False,
    noise_std: float = 0.0,
) -> float:
    """
    Chạy 1 epoch trên tập train.
    Nếu residual=True, loader trả về (xb, yb, cb) với cb = CLIPER predictions,
    model dự đoán delta, loss tính trên (cb + delta) vs yb.
    noise_std > 0: thêm Gaussian noise vào input (data augmentation).
    Returns: loss trung bình (km)
    """
    import torch
    model.train()
    total_loss = 0.0

    for batch in loader:
        if residual:
            xb, yb, cb = batch
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)

        if noise_std > 0:
            xb = xb + torch.randn_like(xb) * noise_std

        optimizer.zero_grad()
        out = model(xb)
        pred = (cb + out) if residual else out
        loss = loss_fn(pred, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item() * len(xb)

    return total_loss / len(loader.dataset)


# ─── 1 epoch val ──────────────────────────────────────────────────────────────

def eval_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: HaversineLoss,
    device: torch.device,
    residual: bool = False,
) -> tuple[float, float, float]:
    """
    Đánh giá trên tập val/test.
    Returns: (val_loss_km, mae_24h_km, mae_48h_km)
    """
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            if residual:
                xb, yb, cb = batch
                xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)

            out = model(xb)
            pred = (cb + out) if residual else out
            loss = loss_fn(pred, yb)
            total_loss += loss.item() * len(xb)
            all_pred.append(pred.cpu())
            all_true.append(yb.cpu())

    all_pred = torch.cat(all_pred)
    all_true = torch.cat(all_true)
    mae_24h, mae_48h = loss_fn.mae_km(all_pred, all_true)

    return total_loss / len(loader.dataset), mae_24h, mae_48h


# ─── Full training run ────────────────────────────────────────────────────────

def run_training(
    model: nn.Module,
    model_name: str,
    cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device = None,
    scaler=None,
    tag: str = "",
    lr_scale: float = 1.0,
    max_epochs_override: int = None,
    patience_override: int = None,
    seed_suffix: str = "",
    hem_weights: np.ndarray = None,
) -> dict:
    """
    Toàn bộ training loop cho 1 model.

    Args:
        model       : nn.Module chưa train
        model_name  : 'lstm' | 'bilstm' | 'transformer'
        cfg         : dict từ config.yaml
        X_train/val : np.ndarray đã StandardScaler
        y_train/val : np.ndarray lat/lon thực (chưa scale)
        device      : CPU hoặc CUDA (None → tự detect)
        scaler      : fitted StandardScaler (bắt buộc nếu residual=True)

    Returns:
        dict chứa best_mae_24h, best_mae_48h, epochs_trained, history
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    m_cfg    = cfg["model"]
    bs       = m_cfg["batch_size"]        # 32
    lr       = m_cfg["lr"] * lr_scale    # scale LR for fine-tuning
    max_ep   = max_epochs_override if max_epochs_override else m_cfg["max_epochs"]
    patience = patience_override if patience_override else m_cfg["patience"]
    residual = m_cfg.get("residual", False)
    base_dir = Path(__file__).parent.parent.parent
    suffix   = f"_{tag}" if tag else ""
    model_key = f"{model_name}{suffix}{seed_suffix}"   # e.g. "lstm_scs_v9_s42" — dùng cho log + ckpt

    if residual and scaler is None:
        raise ValueError("residual=True nhưng chưa truyền scaler vào run_training()")

    # ════════════════════════════════════════════════════════════
    # COMPREHENSIVE CONFIG DUMP — in mọi tham số trước khi train
    # ════════════════════════════════════════════════════════════
    # Compute anchor_steps + ckpt_path sớm để dùng trong dump
    anchor_steps = m_cfg.get("anchor_steps", [4, 8])
    ckpt_path = base_dir / cfg["output"]["checkpoint_dir"] / f"best_{model_key}.pt"

    feat_names = cfg.get("features", {}).get("names", [])
    n_feat_actual = X_train.shape[2]
    lookback_actual = X_train.shape[1]
    out_size_actual = y_train.shape[1]
    n_steps = out_size_actual // 2

    # Step weights cho MultiHorizonLoss
    loss_cfg = cfg.get("loss", {})
    w24 = loss_cfg.get("weight_24h", 0.5)
    w48 = loss_cfg.get("weight_48h", 0.2)
    n_other = max(n_steps - 2, 1)
    other_w = max(0.0, 1.0 - w24 - w48) / n_other
    step_weights = []
    for step in anchor_steps:
        if step == 4:   step_weights.append(w24)
        elif step == 8: step_weights.append(w48)
        else:           step_weights.append(other_w)

    print(f"\n{'═'*60}")
    print(f"  TRAINING CONFIG DUMP — {model_key.upper()}")
    print(f"{'═'*60}")

    print(f"\n[DEVICE & SEED]")
    print(f"  device                : {device}")
    print(f"  seed_suffix           : {seed_suffix or '(none)'}")

    print(f"\n[DATA SHAPE]")
    print(f"  X_train               : {X_train.shape}  ({len(X_train):,} sequences)")
    print(f"  X_val                 : {X_val.shape}    ({len(X_val):,} sequences)")
    print(f"  y_train               : {y_train.shape}")
    print(f"  lookback              : {lookback_actual} steps × 6h = {lookback_actual*6}h history")
    print(f"  horizon               : {n_steps} steps × 6h = {n_steps*6}h forecast")
    print(f"  output_size           : {out_size_actual} (= {n_steps} steps × 2 coords)")

    print(f"\n[FEATURES] ({n_feat_actual} total)")
    if feat_names and len(feat_names) <= n_feat_actual:
        # Print theo nhóm để dễ đọc
        groups = {
            "Storm-state":      feat_names[:13],
            "ERA5 mid (500-700)": [f for f in feat_names[13:19] if f in feat_names],
            "ERA5 upper/low":   [f for f in feat_names[19:25] if f in feat_names],
            "Physics prior":    [f for f in feat_names if f.startswith("beta_")],
            "Annulus steering": [f for f in feat_names if f.startswith("asteer_")],
            "Other":            [f for f in feat_names if f.startswith("enso") or f == "in_buffer"],
        }
        for gname, gfeats in groups.items():
            if gfeats:
                print(f"  {gname:<22}: ({len(gfeats)}) {', '.join(gfeats)}")
    else:
        print(f"  (auto-detected from X_train.shape[2] = {n_feat_actual})")

    print(f"\n[MODEL ARCHITECTURE]")
    print(f"  model_name            : {model_name}")
    print(f"  hidden_size / d_model : {m_cfg.get('hidden_size', 'N/A')}")
    print(f"  num_layers            : {m_cfg.get('num_layers', 'N/A')}")
    if model_name == "transformer":
        print(f"  nhead                 : {m_cfg.get('nhead', 'N/A')}  ({m_cfg.get('hidden_size', 0)//m_cfg.get('nhead', 1)} dim/head)")
        print(f"  dim_feedforward       : {m_cfg.get('hidden_size', 128) * 2}  (2× d_model)")
        print(f"  norm_first            : True  (Pre-LayerNorm)")
        print(f"  positional encoding   : Learnable (init ×0.01)")
    print(f"  dropout               : {m_cfg.get('dropout', 0.2)}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params          : {n_params:,}")

    print(f"\n[TRAINING HYPERPARAMS]")
    print(f"  optimizer             : Adam (weight_decay=1e-4)")
    print(f"  base lr               : {m_cfg.get('lr', 5e-4):.2e}")
    if lr_scale != 1.0:
        print(f"  lr_scale              : {lr_scale}  → effective lr={lr:.2e}")
    print(f"  scheduler             : CosineAnnealingWarmRestarts (T_0=50, T_mult=2, η_min=1e-5)")
    print(f"  batch_size            : {bs}")
    print(f"  max_epochs            : {max_ep}")
    print(f"  patience (early-stop) : {patience}")
    print(f"  grad_clip             : 1.0")
    print(f"  noise_std (Gaussian aug): {m_cfg.get('noise_std', 0.0)}  (scaled space)")
    print(f"  feature value clip    : ±5σ  (sau StandardScaler)")

    use_swa_cfg = m_cfg.get("swa", False)
    print(f"\n[SWA — Stochastic Weight Averaging]")
    print(f"  enabled               : {use_swa_cfg}")
    if use_swa_cfg:
        swa_start_frac = m_cfg.get('swa_start_frac', 0.75)
        swa_lr_scale_v = m_cfg.get('swa_lr_scale', 0.5)
        print(f"  swa_start_frac        : {swa_start_frac}  (= epoch {int(max_ep * swa_start_frac)})")
        print(f"  swa_lr_scale          : {swa_lr_scale_v}  (swa_lr = {lr * swa_lr_scale_v:.2e})")

    print(f"\n[LOSS FUNCTION — MultiHorizonLoss]")
    print(f"  anchor_steps          : {anchor_steps}")
    print(f"  weight_24h (step 4)   : {w24}")
    print(f"  weight_48h (step 8)   : {w48}")
    print(f"  weight_other          : {other_w:.4f} mỗi step ({n_other} steps khác)")
    print(f"  step_weights chi tiết : {['%.3f' % w for w in step_weights]}")
    print(f"  lambda_smooth         : {loss_cfg.get('lambda_smooth', 1.0)}  (L2 of 2nd derivative)")
    print(f"  lambda_dir            : {loss_cfg.get('lambda_dir', 0.1)}  (1 - cos bearing mismatch)")

    print(f"\n[RESIDUAL LEARNING]")
    print(f"  residual              : {residual}")
    if residual:
        print(f"  formula               : final_pred = CLIPER(X) + Δ_model(X)")
        print(f"  CLIPER velocity avg   : 3 bước cuối (ổn định hơn last-step)")

    if hem_weights is not None:
        print(f"\n[HARD EXAMPLE MINING]")
        print(f"  enabled               : True (WeightedRandomSampler)")
        print(f"  n_samples             : {len(hem_weights)}")
        print(f"  weight range          : [{hem_weights.min():.3f}, {hem_weights.max():.3f}]")
        print(f"  weight mean / std     : {hem_weights.mean():.3f} / {hem_weights.std():.3f}")

    print(f"\n[DATA SPLIT (theo SEASON)]")
    split_cfg = cfg.get("split", {})
    print(f"  Train years           : {split_cfg.get('train', 'N/A')}")
    print(f"  Val years             : {split_cfg.get('val', 'N/A')}")
    print(f"  Test years            : {split_cfg.get('test', 'N/A')}")

    print(f"\n[CHECKPOINT]")
    print(f"  output                : {ckpt_path}")

    print(f"\n{'═'*60}\n")

    # --- CLIPER predictions (nếu residual) ---
    if residual:
        c_train = cliper_from_X(X_train, scaler, anchor_steps=anchor_steps)
        c_val   = cliper_from_X(X_val, scaler,   anchor_steps=anchor_steps)
        print(f"  CLIPER computed: train={c_train.shape}, val={c_val.shape}")

        train_ds = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
            torch.from_numpy(c_train),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
            torch.from_numpy(c_val),
        )
    else:
        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_val)
        )

    # --- DataLoader ---
    # Hard Example Mining: dùng WeightedRandomSampler nếu có hem_weights
    if hem_weights is not None:
        if len(hem_weights) != len(train_ds):
            raise ValueError(f"hem_weights length {len(hem_weights)} != train_ds {len(train_ds)}")
        from torch.utils.data import WeightedRandomSampler
        weights_t = torch.from_numpy(hem_weights.astype(np.float64))
        sampler   = WeightedRandomSampler(weights_t, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler,
                                  pin_memory=(device.type == "cuda"))
        print(f"  HEM enabled: WeightedRandomSampler ({len(hem_weights)} samples, "
              f"weight range [{hem_weights.min():.3f}, {hem_weights.max():.3f}])")
    else:
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                  pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            pin_memory=(device.type == "cuda"))

    model = model.to(device)

    # --- Optimizer + Scheduler ---
    noise_std = m_cfg.get("noise_std", 0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # CosineAnnealingWarmRestarts: thoát local minima tốt hơn ReduceLROnPlateau
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-5
    )

    # --- SWA (Stochastic Weight Averaging, Izmailov et al. 2018) ---
    use_swa        = m_cfg.get("swa", False)
    swa_start_frac = m_cfg.get("swa_start_frac", 0.75)
    swa_lr_scale   = m_cfg.get("swa_lr_scale", 0.5)
    swa_start_ep   = int(max_ep * swa_start_frac) if use_swa else max_ep + 1
    swa_model      = None
    swa_scheduler  = None
    if use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer,
            anneal_strategy="linear",
            anneal_epochs=5,
            swa_lr=lr * swa_lr_scale,
        )
        print(f"  SWA enabled: start at epoch {swa_start_ep}, swa_lr={lr*swa_lr_scale:.2e}")

    # --- Loss: MultiHorizonLoss (sprint1) hoặc HaversineLoss (legacy) ---
    loss_fn = build_loss(cfg)

    # --- Early stopping ---
    # ckpt_path đã được định nghĩa ở config dump section ở đầu
    stopper   = EarlyStopping(patience=patience, checkpoint_path=str(ckpt_path))

    # --- History ---
    history = {"train_loss": [], "val_loss": [], "mae_24h": [], "mae_48h": [], "lr": []}
    best_mae_24h = float("inf")
    best_mae_48h = float("inf")
    t_start = time.time()

    # --- Training loop ---
    for epoch in range(1, max_ep + 1):
        t_ep = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, residual=residual, noise_std=noise_std)
        val_loss, mae_24h, mae_48h = eval_one_epoch(model, val_loader, loss_fn, device, residual=residual)

        # SWA: dùng SWA scheduler + update weights sau swa_start_ep
        if use_swa and epoch >= swa_start_ep:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(round(train_loss, 3))
        history["val_loss"].append(round(val_loss, 3))
        history["mae_24h"].append(round(mae_24h, 2))
        history["mae_48h"].append(round(mae_48h, 2))
        history["lr"].append(current_lr)

        if mae_24h < best_mae_24h:
            best_mae_24h = mae_24h
            best_mae_48h = mae_48h

        ep_time = time.time() - t_ep
        print(
            f"  Epoch {epoch:3d}/{max_ep} | "
            f"train={train_loss:7.1f} km | "
            f"val={val_loss:7.1f} km | "
            f"MAE 24h={mae_24h:6.1f} km | "
            f"MAE 48h={mae_48h:6.1f} km | "
            f"lr={current_lr:.2e} | "
            f"{ep_time:.1f}s"
        )

        if stopper.step(val_loss, model):
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    total_time = time.time() - t_start
    epochs_trained = len(history["train_loss"])

    # --- Load best (early-stop) weights ---
    stopper.load_best(model)
    _, best_mae_24h_es, best_mae_48h_es = eval_one_epoch(model, val_loader, loss_fn, device, residual=residual)

    final_mae_24h, final_mae_48h = best_mae_24h_es, best_mae_48h_es
    swa_used = False

    # --- SWA evaluation: compare SWA weights vs best early-stop ---
    if use_swa and swa_model is not None and epochs_trained > swa_start_ep:
        from torch.optim.swa_utils import update_bn
        try:
            # SWA cần update BatchNorm stats — model này không có BN, nhưng giữ để an toàn
            update_bn(train_loader, swa_model, device=device)
        except Exception:
            pass
        # Đánh giá SWA model trên val
        swa_inner = swa_model.module
        _, swa_mae_24h, swa_mae_48h = eval_one_epoch(swa_inner, val_loader, loss_fn, device, residual=residual)
        print(f"\n  SWA result: MAE 24h = {swa_mae_24h:.1f} km  vs  best ES = {best_mae_24h_es:.1f} km")

        # Nếu SWA tốt hơn → dùng SWA weights làm checkpoint cuối
        if swa_mae_24h < best_mae_24h_es:
            print(f"  → SWA wins (+{best_mae_24h_es - swa_mae_24h:.1f} km), saving SWA weights")
            # Copy SWA weights vào model gốc và lưu
            model.load_state_dict(swa_inner.state_dict())
            torch.save(model.state_dict(), str(ckpt_path))
            final_mae_24h, final_mae_48h = swa_mae_24h, swa_mae_48h
            swa_used = True
        else:
            print(f"  → Early-stop wins, keeping ES weights")

    print(f"\n  Best result ({model_name.upper()}):")
    print(f"    Val MAE 24h = {final_mae_24h:.1f} km  {'[SWA]' if swa_used else '[ES]'}")
    print(f"    Val MAE 48h = {final_mae_48h:.1f} km")
    print(f"    Epochs      = {epochs_trained}")
    print(f"    Thời gian   = {total_time/60:.1f} phút")
    print(f"    Checkpoint  = {ckpt_path}")

    result = {
        "model_name":    model_key,
        "best_mae_24h":  round(final_mae_24h, 2),
        "best_mae_48h":  round(final_mae_48h, 2),
        "epochs_trained": epochs_trained,
        "total_time_min": round(total_time / 60, 1),
        "swa_used":      swa_used,
        "history":       history,
        "checkpoint":    str(ckpt_path),
    }

    # --- Ghi log ---
    log_path = base_dir / cfg["output"]["results_log"]
    log_result(str(log_path), result)

    return result
