"""
trainer.py — Training loop dùng chung cho cả 3 model
------------------------------------------------------
Hàm run_training() nhận bất kỳ model nào (LSTM / BiLSTM / Transformer)
và chạy toàn bộ vòng lặp train → val → early stopping → lưu checkpoint.

Hỗ trợ residual learning: model dự đoán delta từ CLIPER,
loss tính trên vị trí tuyệt đối (CLIPER + delta) vs actual.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

from src.g4_models import HaversineLoss
from .utils import EarlyStopping, log_result


# ─── CLIPER prediction từ X sequences ────────────────────────────────────────

def cliper_from_X(X: np.ndarray, scaler) -> np.ndarray:
    """
    Tính CLIPER (persistence extrapolation) từ X sequences đã scaled.

    Args:
        X      : (N, lookback, 12) — đã StandardScaler
        scaler : fitted StandardScaler

    Returns:
        (N, 4) = [lat_24h, lon_24h, lat_48h, lon_48h] in degrees
    """
    N = X.shape[0]
    last_step_scaled = X[:, -1, :]                              # (N, 12)
    last_step_orig   = scaler.inverse_transform(last_step_scaled)  # (N, 12)

    # Denormalize position: lat_norm → lat, lon_norm → lon
    lat_current = last_step_orig[:, 0] * 14.0 + 8.0    # lat_norm → [8, 22]
    lon_current = last_step_orig[:, 1] * 18.0 + 102.0  # lon_norm → [102, 120]

    # Displacement (degrees / 6h)
    dlat = last_step_orig[:, 2]
    dlon = last_step_orig[:, 3]

    return np.stack([
        lat_current + 4 * dlat,   # 24h = 4 × 6h
        lon_current + 4 * dlon,
        lat_current + 8 * dlat,   # 48h = 8 × 6h
        lon_current + 8 * dlon,
    ], axis=1).astype(np.float32)


# ─── 1 epoch train ────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: HaversineLoss,
    device: torch.device,
    grad_clip: float = 1.0,
    residual: bool = False,
) -> float:
    """
    Chạy 1 epoch trên tập train.
    Nếu residual=True, loader trả về (xb, yb, cb) với cb = CLIPER predictions,
    model dự đoán delta, loss tính trên (cb + delta) vs yb.
    Returns: loss trung bình (km)
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        if residual:
            xb, yb, cb = batch
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
        else:
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)

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
    lr       = m_cfg["lr"]               # 0.001
    max_ep   = m_cfg["max_epochs"]       # 150
    patience = m_cfg["patience"]         # 20
    residual = m_cfg.get("residual", False)
    base_dir = Path(__file__).parent.parent.parent

    if residual and scaler is None:
        raise ValueError("residual=True nhưng chưa truyền scaler vào run_training()")

    print(f"\n{'='*55}")
    print(f"  Training: {model_name.upper()}  |  device: {device}"
          f"  |  residual: {residual}")
    print(f"{'='*55}")

    # --- CLIPER predictions (nếu residual) ---
    if residual:
        c_train = cliper_from_X(X_train, scaler)  # (N_train, 4)
        c_val   = cliper_from_X(X_val, scaler)    # (N_val, 4)
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
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, pin_memory=(device.type == "cuda"))

    model = model.to(device)

    # --- Optimizer + Scheduler ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5, min_lr=1e-5
    )
    # ReduceLROnPlateau: giảm lr×0.5 khi val_loss không cải thiện sau 10 epoch
    # min_lr=1e-5: sàn lr — không giảm xuống dưới mức này

    # --- Loss ---
    loss_fn = HaversineLoss(
        weight_24h=cfg["loss"]["weight_24h"],
        weight_48h=cfg["loss"]["weight_48h"],
    )

    # --- Early stopping ---
    ckpt_path = base_dir / cfg["output"]["checkpoint_dir"] / f"best_{model_name}.pt"
    stopper   = EarlyStopping(patience=patience, checkpoint_path=str(ckpt_path))

    # --- History ---
    history = {"train_loss": [], "val_loss": [], "mae_24h": [], "mae_48h": [], "lr": []}
    best_mae_24h = float("inf")
    best_mae_48h = float("inf")
    t_start = time.time()

    # --- Training loop ---
    for epoch in range(1, max_ep + 1):
        t_ep = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, residual=residual)
        val_loss, mae_24h, mae_48h = eval_one_epoch(model, val_loader, loss_fn, device, residual=residual)

        scheduler.step(val_loss)
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

    # --- Load best weights ---
    stopper.load_best(model)
    _, final_mae_24h, final_mae_48h = eval_one_epoch(model, val_loader, loss_fn, device, residual=residual)

    print(f"\n  Best result ({model_name.upper()}):")
    print(f"    Val MAE 24h = {final_mae_24h:.1f} km")
    print(f"    Val MAE 48h = {final_mae_48h:.1f} km")
    print(f"    Epochs      = {epochs_trained}")
    print(f"    Thời gian   = {total_time/60:.1f} phút")
    print(f"    Checkpoint  = {ckpt_path}")

    result = {
        "model_name":    model_name,
        "best_mae_24h":  round(final_mae_24h, 2),
        "best_mae_48h":  round(final_mae_48h, 2),
        "epochs_trained": epochs_trained,
        "total_time_min": round(total_time / 60, 1),
        "history":       history,
        "checkpoint":    str(ckpt_path),
    }

    # --- Ghi log ---
    log_path = base_dir / cfg["output"]["results_log"]
    log_result(str(log_path), result)

    return result
