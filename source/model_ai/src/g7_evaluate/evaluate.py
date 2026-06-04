"""
evaluate.py — G7: Đánh giá model trên tập test
------------------------------------------------
  1. CLIPER baseline (persistence extrapolation)
  2. Load checkpoint cho từng model (lstm, bilstm, transformer)
  3. Tính Haversine MAE, RMSE, Skill Score
  4. Vẽ 7 biểu đồ + Folium track maps
  5. Ghi kết quả vào results_log.json
  6. Checkpoint G7: Skill Score >= 20%

Chạy: python -m src.g7_evaluate.evaluate
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import json
import pickle
import numpy as np
import torch
from pathlib import Path

from src.g4_models import build_model, load_config
from src.g5_g6_train.utils import log_result, load_scaler
from src.g5_g6_train.trainer import cliper_from_X


# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.parent   # source/model_ai/


# ─── Haversine (numpy, vectorized) ────────────────────────────────────────────

def haversine_km(lat1: np.ndarray, lon1: np.ndarray,
                 lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance (km). All inputs in degrees."""
    R = 6371.0
    lat1r = np.deg2rad(lat1)
    lat2r = np.deg2rad(lat2)
    dlat  = np.deg2rad(lat2 - lat1)
    dlon  = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    return R * c


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _get_24h_48h_indices(n_outputs: int) -> tuple[int, int]:
    """
    Trả về (flat_idx_24h, flat_idx_48h) trong mảng pred/true.
    - Legacy (n_outputs=4):  24h=0, 48h=2
    - Multi-horizon (n_outputs=16): step4=24h → idx 6, step8=48h → idx 14
    """
    if n_outputs == 4:
        return 0, 2
    # Multi-horizon: n_steps = n_outputs // 2, step 4 (24h) = index 3 → flat 6,7
    # step 8 (48h) = index 7 → flat 14,15
    return 6, 14


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    pred, true: (N, n_outputs) — hỗ trợ cả legacy (4) và multi-horizon (16).
    Returns dict với mae_24h, rmse_24h, mae_48h, rmse_48h.
    """
    idx_24, idx_48 = _get_24h_48h_indices(pred.shape[1])
    err_24 = haversine_km(pred[:, idx_24], pred[:, idx_24+1],
                          true[:, idx_24], true[:, idx_24+1])
    err_48 = haversine_km(pred[:, idx_48], pred[:, idx_48+1],
                          true[:, idx_48], true[:, idx_48+1])
    return {
        "mae_24h":  float(np.mean(err_24)),
        "rmse_24h": float(np.sqrt(np.mean(err_24 ** 2))),
        "mae_48h":  float(np.mean(err_48)),
        "rmse_48h": float(np.sqrt(np.mean(err_48 ** 2))),
        "err_24":   err_24,
        "err_48":   err_48,
    }


def skill_score(cliper_mae: float, model_mae: float) -> float:
    """Skill Score (%) = (cliper - model) / cliper * 100."""
    if cliper_mae == 0:
        return 0.0
    return (cliper_mae - model_mae) / cliper_mae * 100.0


# cliper_predict = alias of shared cliper_from_X (same logic, reuse from trainer)
cliper_predict = cliper_from_X


# ─── Trajectory classification ────────────────────────────────────────────────

def classify_trajectories(X_test: np.ndarray, scaler) -> np.ndarray:
    """
    Phân loại trajectory từ 4 bước cuối của X_test.
    Returns: array of str (N,) — "westward" | "recurving" | "southward" | "erratic"
    """
    N = X_test.shape[0]
    # Lấy 4 bước cuối, inverse transform
    last4_scaled = X_test[:, -4:, :]  # (N, 4, 12)
    n_feat = last4_scaled.shape[2]
    last4_flat   = last4_scaled.reshape(N * 4, n_feat)
    last4_orig   = scaler.inverse_transform(last4_flat).reshape(N, 4, n_feat)

    dlat_avg = last4_orig[:, :, 2].mean(axis=1)  # avg dlat (index 2)
    dlon_avg = last4_orig[:, :, 3].mean(axis=1)  # avg dlon (index 3)

    labels = np.empty(N, dtype=object)
    for i in range(N):
        dl = dlat_avg[i]
        dn = dlon_avg[i]
        if dn < -0.05 and abs(dn) > abs(dl):
            labels[i] = "westward"
        elif dl > 0.05 and dn > -0.05:
            labels[i] = "recurving"
        elif dl < -0.05:
            labels[i] = "southward"
        else:
            labels[i] = "erratic"
    return labels


# ─── Load model ───────────────────────────────────────────────────────────────

def _detect_hidden_size(state_dict: dict, model_name: str) -> int:
    """Tự detect hidden_size từ shape của layer đầu tiên trong checkpoint."""
    if model_name == "lstm":
        w = state_dict.get("lstm.weight_ih_l0")
        return int(w.shape[0] / 4) if w is not None else 128
    elif model_name == "bilstm":
        w = state_dict.get("bilstm.weight_ih_l0")
        return int(w.shape[0] / 4) if w is not None else 128
    elif model_name == "bigru":
        w = state_dict.get("bigru.weight_ih_l0")
        return int(w.shape[0] / 3) if w is not None else 128  # GRU: 3 gates
    elif model_name == "transformer":
        w = state_dict.get("input_proj.weight")
        return int(w.shape[0]) if w is not None else 128
    return 128


def _detect_n_features(state_dict: dict, model_name: str) -> int:
    """Detect n_features từ shape của input layer trong checkpoint."""
    if model_name == "lstm":
        w = state_dict.get("lstm.weight_ih_l0")
        return int(w.shape[1]) if w is not None else 14
    elif model_name == "bilstm":
        w = state_dict.get("bilstm.weight_ih_l0")
        return int(w.shape[1]) if w is not None else 14
    elif model_name == "bigru":
        w = state_dict.get("bigru.weight_ih_l0")
        return int(w.shape[1]) if w is not None else 14
    elif model_name == "transformer":
        w = state_dict.get("input_proj.weight")
        return int(w.shape[1]) if w is not None else 14
    return 14


def _detect_lookback(state_dict: dict, model_name: str) -> int:
    """Detect lookback từ pos_embed shape (Transformer) hoặc trả về None."""
    if model_name == "transformer":
        w = state_dict.get("pos_embed")
        return int(w.shape[1]) if w is not None else None
    return None


def _detect_output_size(state_dict: dict) -> int:
    """Detect output_size từ head layer cuối của checkpoint."""
    # Tìm Linear layer cuối trong head (bias shape = (output_size,))
    for key in reversed(list(state_dict.keys())):
        if "head" in key and "bias" in key:
            return int(state_dict[key].shape[0])
    return 4  # legacy fallback


def _detect_num_layers(state_dict: dict, model_name: str) -> int:
    """Detect num_layers bằng cách đếm layer keys."""
    prefix = {"lstm": "lstm", "bilstm": "bilstm", "bigru": "bigru"}.get(model_name)
    if prefix is None:
        # Transformer: đếm encoder layers
        layer_idx = 0
        while f"encoder.layers.{layer_idx}.self_attn.in_proj_weight" in state_dict:
            layer_idx += 1
        return max(layer_idx, 1)
    layer = 0
    while f"{prefix}.weight_ih_l{layer}" in state_dict:
        layer += 1
    return max(layer, 1)


def _build_model_from_state(state_dict: dict, model_name: str, cfg: dict):
    """Build model + load weights, tự detect kiến trúc từ state_dict."""
    patched_cfg = {**cfg, "model": {**cfg["model"]},
                   "features": {**cfg["features"]}}
    patched_cfg["model"]["hidden_size"] = _detect_hidden_size(state_dict, model_name)
    patched_cfg["model"]["num_layers"]  = _detect_num_layers(state_dict, model_name)
    patched_cfg["model"]["output_size"] = _detect_output_size(state_dict)

    n_feat = _detect_n_features(state_dict, model_name)
    patched_cfg["features"]["n_features"] = n_feat

    lookback = _detect_lookback(state_dict, model_name)
    if lookback is not None:
        patched_cfg["model"]["lookback"] = lookback

    model = build_model(model_name, patched_cfg)
    model.load_state_dict(state_dict)
    model.eval()
    return model, patched_cfg


def load_model_checkpoint(model_name: str, cfg: dict, tag: str = "14feat"):
    """
    Thử load từ models/checkpoints/best_{name}_{tag}.pt trước,
    rồi models/final/{name}_*_{tag}.pt.
    Auto-detect hidden_size và num_layers từ checkpoint.
    Trả về (model, ckpt_path) hoặc (None, None) nếu không tìm thấy.
    """
    suffix = f"_{tag}" if tag else ""
    ckpt_dir  = BASE_DIR / cfg["output"]["checkpoint_dir"]
    final_dir = BASE_DIR / "models" / "final"
    candidates = [
        ckpt_dir  / f"best_{model_name}{suffix}.pt",
        final_dir / f"{model_name}_baseline{suffix}.pt",
        final_dir / f"{model_name}_best{suffix}.pt",
    ]
    # Chỉ fallback về legacy (non-tagged) khi chạy với tag rỗng
    if not tag:
        candidates.extend([
            ckpt_dir  / f"best_{model_name}.pt",
            final_dir / f"{model_name}.pt",
        ])
    found_path = next((p for p in candidates if p.exists()), None)
    if found_path is None:
        return None, None

    try:
        state_dict = torch.load(str(found_path), map_location="cpu", weights_only=True)

        # Patch config với kiến trúc thực tế từ checkpoint
        patched_cfg = {**cfg, "model": {**cfg["model"]},
                       "features": {**cfg["features"]}}
        patched_cfg["model"]["hidden_size"] = _detect_hidden_size(state_dict, model_name)
        patched_cfg["model"]["num_layers"]  = _detect_num_layers(state_dict, model_name)
        patched_cfg["model"]["output_size"] = _detect_output_size(state_dict)

        n_feat = _detect_n_features(state_dict, model_name)
        patched_cfg["features"]["n_features"] = n_feat

        lookback = _detect_lookback(state_dict, model_name)
        if lookback is not None:
            patched_cfg["model"]["lookback"] = lookback

        model = build_model(model_name, patched_cfg)
        model.load_state_dict(state_dict)
        model.eval()
        h = patched_cfg["model"]["hidden_size"]
        l = patched_cfg["model"]["num_layers"]
        lb = patched_cfg["model"]["lookback"]
        print(f"  [load] {model_name.upper()} <- {found_path.name}  (hidden={h}, layers={l}, feat={n_feat}, lookback={lb})")
        return model, found_path
    except Exception as e:
        print(f"  [warn] Không load được {model_name}: {e}")
        return None, None


def load_seed_checkpoints(model_name: str, cfg: dict, tag: str = ""):
    """
    Trả về list (model, ckpt_path) cho mỗi seed checkpoint:
      best_{model_name}_{tag}_s{N}.pt
    Trả về list rỗng nếu không tìm thấy seed nào.
    """
    suffix    = f"_{tag}" if tag else ""
    ckpt_dir  = BASE_DIR / cfg["output"]["checkpoint_dir"]
    pattern   = f"best_{model_name}{suffix}_s*.pt"
    seed_files = sorted(ckpt_dir.glob(pattern))
    if not seed_files:
        return []

    models = []
    for fp in seed_files:
        try:
            sd = torch.load(str(fp), map_location="cpu", weights_only=True)
            m, _ = _build_model_from_state(sd, model_name, cfg)
            models.append((m, fp))
            print(f"  [seed] {model_name.upper()} <- {fp.name}")
        except Exception as e:
            print(f"  [warn] Bỏ qua {fp.name}: {e}")
    return models


def model_predict(model, X_test: np.ndarray,
                   cliper_pred: np.ndarray = None,
                   residual: bool = False) -> np.ndarray:
    """
    Chạy inference, trả về (N, 4) degrees.
    Nếu residual=True, model output là delta → cộng CLIPER để ra vị trí tuyệt đối.
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float()
        out = model(X_t).cpu().numpy()  # (N, 4)

    if residual and cliper_pred is not None:
        return cliper_pred + out
    return out


# ─── Plots ────────────────────────────────────────────────────────────────────

COLORS = {
    "cliper":      "#6b7280",
    "lstm":        "#3b82f6",
    "bilstm":      "#10b981",
    "transformer": "#f59e0b",
    "ensemble":    "#8b5cf6",
}


def _savefig(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    out = path / name
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_learning_curve(log_path: Path, fig_dir: Path):
    """Vẽ learning curve nếu results_log.json có entry với 'history' key."""
    if not log_path.exists():
        return
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    entries_with_history = [e for e in logs if "history" in e]
    if not entries_with_history:
        return

    entry = entries_with_history[-1]
    history = entry["history"]
    model_name = entry.get("model_name", "model")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train loss", color="#3b82f6")
    ax.plot(epochs, history["val_loss"],   label="Val loss",   color="#f59e0b")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (km)")
    ax.set_title(f"Learning Curve — {model_name.upper()}")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["mae_24h"], label="MAE 24h", color="#10b981")
    ax.plot(epochs, history["mae_48h"], label="MAE 48h", color="#f59e0b")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (km)")
    ax.set_title("MAE theo Epoch")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    _savefig(fig, fig_dir, "learning_curve.png")


def plot_model_comparison(all_metrics: dict, fig_dir: Path):
    """Grouped bar chart MAE 24h / 48h cho CLIPER + available models."""
    names  = list(all_metrics.keys())
    mae24  = [all_metrics[n]["mae_24h"] for n in names]
    mae48  = [all_metrics[n]["mae_48h"] for n in names]

    x    = np.arange(len(names))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2), 5))
    b1 = ax.bar(x - w/2, mae24, w, label="MAE 24h",
                color=[COLORS.get(n, "#8b5cf6") for n in names], alpha=0.85)
    b2 = ax.bar(x + w/2, mae48, w, label="MAE 48h",
                color=[COLORS.get(n, "#8b5cf6") for n in names], alpha=0.5)

    ax.set_xticks(x); ax.set_xticklabels([n.upper() for n in names])
    ax.set_ylabel("MAE (km)"); ax.set_title("So sánh MAE giữa các mô hình")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _savefig(fig, fig_dir, "model_comparison.png")


def plot_skill_scores(skill_scores_dict: dict, checkpoint_pct: float, fig_dir: Path):
    """Bar chart Skill Score. Đường ngang 20% (checkpoint) và 30% (target)."""
    names  = list(skill_scores_dict.keys())
    scores = [skill_scores_dict[n] for n in names]

    fig, ax = plt.subplots(figsize=(max(5, len(names) * 2), 5))
    bars = ax.bar(names, scores,
                  color=[COLORS.get(n, "#8b5cf6") for n in names], alpha=0.85)
    ax.axhline(checkpoint_pct, color="red",    linestyle="--", linewidth=1.5,
               label=f"Checkpoint {checkpoint_pct}%")
    ax.axhline(30.0,           color="green",  linestyle="--", linewidth=1.5,
               label="Target 30%")
    ax.set_ylabel("Skill Score (%)"); ax.set_title("Skill Score so với CLIPER")
    ax.set_xticks(range(len(names))); ax.set_xticklabels([n.upper() for n in names])
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    for bar, s in zip(bars, scores):
        ax.annotate(f"{s:.1f}%", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _savefig(fig, fig_dir, "skill_scores.png")


def plot_scatter(best_pred: np.ndarray, y_test: np.ndarray,
                 best_name: str, fig_dir: Path):
    """Scatter predicted vs actual LAT 24h và 48h."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    color = COLORS.get(best_name, "#8b5cf6")
    idx_24, idx_48 = _get_24h_48h_indices(best_pred.shape[1])

    for ax, h_idx, label in [(axes[0], idx_24, "24h"), (axes[1], idx_48, "48h")]:
        ax.scatter(y_test[:, h_idx], best_pred[:, h_idx],
                   alpha=0.35, s=8, color=color, label=best_name.upper())
        lo = min(y_test[:, h_idx].min(), best_pred[:, h_idx].min()) - 1
        hi = max(y_test[:, h_idx].max(), best_pred[:, h_idx].max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect")
        ax.set_xlabel("Actual LAT (°)"); ax.set_ylabel("Predicted LAT (°)")
        ax.set_title(f"Scatter: Predicted vs Actual LAT {label}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.tight_layout()
    _savefig(fig, fig_dir, "scatter_pred_actual.png")


def plot_mae_vs_horizon(all_metrics: dict, fig_dir: Path):
    """Line chart MAE 24h và 48h cho tất cả models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    horizons = [24, 48]

    for name, m in all_metrics.items():
        color = COLORS.get(name, "#8b5cf6")
        linestyle = "--" if name == "cliper" else "-"
        ax.plot(horizons, [m["mae_24h"], m["mae_48h"]],
                marker="o", label=name.upper(),
                color=color, linestyle=linestyle, linewidth=2)

    ax.set_xticks(horizons); ax.set_xlabel("Horizon (h)")
    ax.set_ylabel("MAE (km)"); ax.set_title("MAE theo Horizon dự báo")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "mae_vs_horizon.png")


def plot_mae_by_type(all_preds: dict, y_test: np.ndarray,
                     traj_labels: np.ndarray, fig_dir: Path):
    """Bar chart MAE 24h theo trajectory type."""
    types = ["westward", "recurving", "southward", "erratic"]
    fig, ax = plt.subplots(figsize=(max(7, len(all_preds) * 2.5), 5))

    x = np.arange(len(types))
    n_models = len(all_preds)
    w = 0.8 / n_models

    for i, (name, pred) in enumerate(all_preds.items()):
        color = COLORS.get(name, "#8b5cf6")
        idx_24, _ = _get_24h_48h_indices(pred.shape[1])
        maes_by_type = []
        for t in types:
            mask = traj_labels == t
            if mask.sum() == 0:
                maes_by_type.append(0.0)
            else:
                err = haversine_km(pred[mask, idx_24], pred[mask, idx_24+1],
                                   y_test[mask, idx_24], y_test[mask, idx_24+1])
                maes_by_type.append(float(np.mean(err)))
        offset = (i - n_models / 2 + 0.5) * w
        ax.bar(x + offset, maes_by_type, w, label=name.upper(),
               color=color, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels([t.capitalize() for t in types])
    ax.set_ylabel("MAE 24h (km)"); ax.set_title("MAE 24h theo loại quỹ đạo bão")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "mae_by_type.png")


def plot_error_histogram(all_metrics: dict, fig_dir: Path):
    """Histogram lỗi 24h cho tất cả models."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for name, m in all_metrics.items():
        if "err_24" not in m:
            continue
        color = COLORS.get(name, "#8b5cf6")
        lw = 2 if name != "cliper" else 1.5
        ax.hist(m["err_24"], bins=40, alpha=0.4, label=name.upper(),
                color=color, density=True, histtype="stepfilled")
        ax.hist(m["err_24"], bins=40, alpha=0.9, label=None,
                color=color, density=True, histtype="step", linewidth=lw)

    ax.set_xlabel("Error 24h (km)"); ax.set_ylabel("Density")
    ax.set_title("Phân phối lỗi dự báo 24h")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "error_histogram.png")


def plot_folium_tracks(X_test: np.ndarray, y_test: np.ndarray,
                       all_preds: dict, scaler, fig_dir: Path,
                       n_samples: int = 3):
    """
    Vẽ Folium map cho n_samples mẫu (interactive HTML).
    Bỏ qua nếu folium không cài.
    """
    try:
        import folium
    except ImportError:
        print("  [info] folium không cài — bỏ qua track HTML maps")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    indices = np.linspace(0, len(X_test) - 1, n_samples, dtype=int)

    for idx in indices:
        # Inverse transform toàn bộ track
        track_scaled = X_test[idx]               # (8, 12)
        track_orig   = scaler.inverse_transform(track_scaled)  # (8, 12)
        lats_hist = track_orig[:, 0] * 14.0 + 8.0
        lons_hist = track_orig[:, 1] * 18.0 + 102.0

        # Thực tế
        _i24, _i48 = _get_24h_48h_indices(y_test.shape[1])
        true_lat24, true_lon24 = y_test[idx, _i24],   y_test[idx, _i24+1]
        true_lat48, true_lon48 = y_test[idx, _i48],   y_test[idx, _i48+1]

        center = [float(lats_hist[-1]), float(lons_hist[-1])]
        m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")

        # Historical track
        coords_hist = list(zip(lats_hist.tolist(), lons_hist.tolist()))
        folium.PolyLine(coords_hist, color="gray", weight=2,
                        tooltip="Historical track").add_to(m)
        folium.Marker(coords_hist[-1], tooltip="Current pos",
                      icon=folium.Icon(color="blue")).add_to(m)

        # Actual future
        folium.PolyLine(
            [coords_hist[-1], [true_lat24, true_lon24], [true_lat48, true_lon48]],
            color="black", weight=2.5, dash_array="5",
            tooltip="Actual").add_to(m)
        folium.CircleMarker([true_lat24, true_lon24], radius=5,
                            color="black", fill=True, tooltip="Actual 24h").add_to(m)
        folium.CircleMarker([true_lat48, true_lon48], radius=5,
                            color="black", fill=True, tooltip="Actual 48h").add_to(m)

        # Model predictions
        for name, pred in all_preds.items():
            clr = COLORS.get(name, "#8b5cf6")
            _i24, _i48 = _get_24h_48h_indices(pred.shape[1])
            lat24p, lon24p = pred[idx, _i24],   pred[idx, _i24+1]
            lat48p, lon48p = pred[idx, _i48],   pred[idx, _i48+1]
            folium.PolyLine(
                [coords_hist[-1], [lat24p, lon24p], [lat48p, lon48p]],
                color=clr, weight=2, dash_array="8",
                tooltip=f"{name.upper()} forecast").add_to(m)
            folium.CircleMarker([lat24p, lon24p], radius=4, color=clr,
                                fill=True, tooltip=f"{name} 24h").add_to(m)
            folium.CircleMarker([lat48p, lon48p], radius=4, color=clr,
                                fill=True, tooltip=f"{name} 48h").add_to(m)

        out_path = fig_dir / f"track_{idx:05d}.html"
        m.save(str(out_path))
        print(f"  [fig] {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="14feat",
                        help="Tag pipeline (vd: '14feat', '') — mặc định '14feat'")
    parser.add_argument("--scs-only", action=argparse.BooleanOptionalAction, default=True,
                        help="Chỉ đánh giá trên sequences có tâm bão trong SCS box (default: True). "
                             "Dùng --no-scs-only để đánh giá trên toàn WP.")
    args = parser.parse_args()
    tag    = args.tag
    suffix = f"_{tag}" if tag else ""

    print("\n" + "=" * 60)
    print(f"  G7 — Evaluate: CLIPER baseline + Model metrics  [tag='{tag}']")
    print(f"  Test domain: {'SCS-only' if args.scs_only else 'WP-full'}")
    print("=" * 60)

    cfg = load_config()

    # Paths
    seq_path    = BASE_DIR / "data" / "features" / f"sequences{suffix}.npz"
    scaler_path = BASE_DIR / cfg["output"]["scaler_path"].replace(".pkl", f"{suffix}.pkl")
    log_path    = BASE_DIR / cfg["output"]["results_log"]
    # Lưu figures vào subfolder riêng theo tag (+ _scs nếu chỉ SCS) — tránh ghi đè
    sub_dir     = (tag if tag else "legacy") + ("_scs" if args.scs_only else "")
    fig_dir     = BASE_DIR / cfg["output"]["figures_dir"] / sub_dir
    ckpt_pct    = cfg["checkpoints"]["g7_min_skill_score_pct"]

    # --- Load data ---
    print("\n[1] Load dữ liệu test...")
    if not seq_path.exists():
        raise FileNotFoundError(f"Không tìm thấy sequences.npz: {seq_path}")
    # allow_pickle=True để đọc được object arrays từ file npz cũ (SID/init_time)
    data = np.load(str(seq_path), allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    # Load sid_test + init_time_test nếu có (cho per-storm breakdown)
    sid_test       = data["sid_test"]       if "sid_test"       in data.files else None
    init_time_test = data["init_time_test"] if "init_time_test" in data.files else None

    # Filter SCS-only nếu in_scs_test có sẵn trong .npz
    if args.scs_only:
        if "in_scs_test" in data.files:
            mask = data["in_scs_test"].astype(bool)
            n_total = len(mask)
            n_scs   = int(mask.sum())
            X_test  = X_test[mask]
            y_test  = y_test[mask]
            if sid_test       is not None: sid_test       = sid_test[mask]
            if init_time_test is not None: init_time_test = init_time_test[mask]
            print(f"  SCS filter: giữ {n_scs:,}/{n_total:,} sequences ({n_scs/max(n_total,1)*100:.1f}%)")
        else:
            print(f"  [warn] sequences.npz không có 'in_scs_test' — chạy lại G3 để có flag. "
                  f"Tạm thời đánh giá trên toàn test set.")

    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    if sid_test is not None:
        print(f"  Per-storm: {len(np.unique(sid_test))} unique storms in test set")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Không tìm thấy scaler.pkl: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler: {scaler_path}")

    # --- CLIPER baseline ---
    anchor_steps = cfg["model"].get("anchor_steps", [4, 8])
    print("\n[2] CLIPER baseline...")
    cliper_pred = cliper_predict(X_test, scaler, anchor_steps=anchor_steps)
    cliper_metrics = compute_metrics(cliper_pred, y_test)
    print(f"  CLIPER MAE 24h = {cliper_metrics['mae_24h']:.1f} km")
    print(f"  CLIPER MAE 48h = {cliper_metrics['mae_48h']:.1f} km")

    log_result(str(log_path), {
        "stage":    "G7_evaluate",
        "model_name": "cliper",
        "mae_24h":  round(cliper_metrics["mae_24h"], 2),
        "rmse_24h": round(cliper_metrics["rmse_24h"], 2),
        "mae_48h":  round(cliper_metrics["mae_48h"], 2),
        "rmse_48h": round(cliper_metrics["rmse_48h"], 2),
        "skill_score_24h": 0.0,
    })

    # --- Load models + predict ---
    print("\n[3] Load & đánh giá models...")
    residual = cfg["model"].get("residual", False)
    if residual:
        print(f"  [residual mode] Model dự đoán delta từ CLIPER")

    model_names = ["lstm", "bilstm", "bigru", "transformer"]
    all_preds   = {"cliper": cliper_pred}
    all_metrics = {"cliper": cliper_metrics}
    skill_scores_dict = {}

    for mname in model_names:
        # --- Multi-seed: tìm tất cả best_{mname}_{tag}_s*.pt ---
        seed_models = load_seed_checkpoints(mname, cfg, tag=tag)
        if seed_models:
            # Mỗi seed → 1 prediction → trung bình
            seed_preds = []
            for sm, sfp in seed_models:
                p_s = model_predict(sm, X_test, cliper_pred=cliper_pred, residual=residual)
                m_s = compute_metrics(p_s, y_test)
                print(f"    seed {sfp.stem.split('_s')[-1]:>4}  MAE 24h={m_s['mae_24h']:.1f} km")
                seed_preds.append(p_s)
            pred       = np.mean(seed_preds, axis=0)
            ckpt_path  = seed_models[0][1].parent / f"best_{mname}_{tag}_seedavg.pt"
            n_seeds    = len(seed_preds)
            print(f"  {mname.upper()}  ({n_seeds}-seed ensemble)")
        else:
            model, ckpt_path = load_model_checkpoint(mname, cfg, tag=tag)
            if model is None:
                print(f"  [skip] {mname.upper()} — không tìm thấy checkpoint")
                continue
            pred = model_predict(model, X_test, cliper_pred=cliper_pred, residual=residual)
            print(f"  {mname.upper()}")

        metrics = compute_metrics(pred, y_test)
        ss_24   = skill_score(cliper_metrics["mae_24h"], metrics["mae_24h"])
        ss_48   = skill_score(cliper_metrics["mae_48h"], metrics["mae_48h"])

        print(f"    MAE 24h = {metrics['mae_24h']:.1f} km  |  "
              f"RMSE 24h = {metrics['rmse_24h']:.1f} km  |  "
              f"Skill 24h = {ss_24:.1f}%")
        print(f"    MAE 48h = {metrics['mae_48h']:.1f} km  |  "
              f"RMSE 48h = {metrics['rmse_48h']:.1f} km  |  "
              f"Skill 48h = {ss_48:.1f}%")

        all_preds[mname]   = pred
        all_metrics[mname] = metrics
        skill_scores_dict[mname] = ss_24   # dùng 24h cho bảng tổng hợp

        log_result(str(log_path), {
            "stage":          "G7_evaluate",
            "model_name":     f"{mname}{suffix}",
            "mae_24h":        round(metrics["mae_24h"], 2),
            "rmse_24h":       round(metrics["rmse_24h"], 2),
            "mae_48h":        round(metrics["mae_48h"], 2),
            "rmse_48h":       round(metrics["rmse_48h"], 2),
            "skill_score_24h": round(ss_24, 2),
            "skill_score_48h": round(ss_48, 2),
            "checkpoint":     str(ckpt_path),
        })

    # --- Ensemble ---
    # Chỉ ensemble các model tốt hơn CLIPER (skill_24h > 0)
    cliper_mae24 = cliper_metrics["mae_24h"]
    good_models  = {k: v for k, v in all_preds.items()
                    if k != "cliper" and all_metrics[k]["mae_24h"] < cliper_mae24}

    def _make_ensemble(preds_dict, label):
        if len(preds_dict) < 2:
            return
        names_str = ", ".join(k.upper() for k in preds_dict)
        print(f"\n  ENSEMBLE-{label} ({len(preds_dict)} models: {names_str})")
        ens_pred  = np.mean(list(preds_dict.values()), axis=0)
        ens_m     = compute_metrics(ens_pred, y_test)
        ens_ss24  = skill_score(cliper_mae24, ens_m["mae_24h"])
        ens_ss48  = skill_score(cliper_metrics["mae_48h"], ens_m["mae_48h"])
        print(f"    MAE 24h = {ens_m['mae_24h']:.1f} km  |  "
              f"RMSE 24h = {ens_m['rmse_24h']:.1f} km  |  "
              f"Skill 24h = {ens_ss24:.1f}%")
        print(f"    MAE 48h = {ens_m['mae_48h']:.1f} km  |  "
              f"RMSE 48h = {ens_m['rmse_48h']:.1f} km  |  "
              f"Skill 48h = {ens_ss48:.1f}%")

        key = f"ensemble_{label.lower()}"
        all_preds[key]   = ens_pred
        all_metrics[key] = ens_m
        skill_scores_dict[key] = ens_ss24
        log_result(str(log_path), {
            "stage":           "G7_evaluate",
            "model_name":      f"{key}{suffix}",
            "mae_24h":         round(ens_m["mae_24h"], 2),
            "rmse_24h":        round(ens_m["rmse_24h"], 2),
            "mae_48h":         round(ens_m["mae_48h"], 2),
            "rmse_48h":        round(ens_m["rmse_48h"], 2),
            "skill_score_24h": round(ens_ss24, 2),
            "skill_score_48h": round(ens_ss48, 2),
            "models":          list(preds_dict.keys()),
        })

    # Ensemble all models (tất cả có checkpoint)
    all_model_preds = {k: v for k, v in all_preds.items() if k != "cliper"}
    _make_ensemble(all_model_preds, "ALL")

    # Ensemble top-2 (2 models tốt nhất theo MAE 24h)
    sorted_models = sorted(all_model_preds.items(),
                           key=lambda x: all_metrics[x[0]]["mae_24h"])
    if len(sorted_models) >= 2:
        top2 = dict(sorted_models[:2])
        _make_ensemble(top2, "TOP2")

    # Ensemble chỉ các models tốt hơn CLIPER
    if len(good_models) >= 2:
        _make_ensemble(good_models, "GOOD")

    # --- Per-storm MAE breakdown ---
    if sid_test is not None:
        print("\n[3b] Per-storm MAE breakdown...")
        # Chọn best model (mae_24h nhỏ nhất, bỏ cliper)
        _model_metrics = {k: v for k, v in all_metrics.items() if k != "cliper"}
        if _model_metrics:
            _best_name = min(_model_metrics, key=lambda k: _model_metrics[k]["mae_24h"])
            _best_pred = all_preds[_best_name]
            print(f"  Using best model: {_best_name.upper()}")

            idx_24, idx_48 = _get_24h_48h_indices(_best_pred.shape[1])
            unique_sids = np.unique(sid_test)
            per_storm = []
            for sid in unique_sids:
                idxs = np.where(sid_test == sid)[0]
                # Best model errors
                err_24 = haversine_km(_best_pred[idxs, idx_24],   _best_pred[idxs, idx_24+1],
                                      y_test[idxs, idx_24],       y_test[idxs, idx_24+1])
                err_48 = haversine_km(_best_pred[idxs, idx_48],   _best_pred[idxs, idx_48+1],
                                      y_test[idxs, idx_48],       y_test[idxs, idx_48+1])
                # CLIPER errors (cho skill score)
                c_err_24 = haversine_km(cliper_pred[idxs, idx_24], cliper_pred[idxs, idx_24+1],
                                        y_test[idxs, idx_24],      y_test[idxs, idx_24+1])
                c_err_48 = haversine_km(cliper_pred[idxs, idx_48], cliper_pred[idxs, idx_48+1],
                                        y_test[idxs, idx_48],      y_test[idxs, idx_48+1])
                mae24, mae48 = float(err_24.mean()), float(err_48.mean())
                cmae24, cmae48 = float(c_err_24.mean()), float(c_err_48.mean())
                season = int(sid[:4]) if len(sid) >= 4 and sid[:4].isdigit() else 0
                per_storm.append({
                    "sid":           str(sid),
                    "season":        season,
                    "n_sequences":   int(len(idxs)),
                    "mae_24h":       round(mae24, 2),
                    "mae_48h":       round(mae48, 2),
                    "cliper_mae_24h": round(cmae24, 2),
                    "cliper_mae_48h": round(cmae48, 2),
                    "skill_24h":     round(skill_score(cmae24, mae24), 2),
                    "skill_48h":     round(skill_score(cmae48, mae48), 2),
                })

            # In bảng sorted theo SID
            per_storm.sort(key=lambda x: x["sid"])
            print(f"\n  {'SID':<16} {'Year':>5} {'N':>4} {'MAE 24h':>8} {'MAE 48h':>9} {'CLIPER 24h':>11} {'Skill 24h':>10} {'Skill 48h':>10}")
            print(f"  {'-'*16} {'-'*5} {'-'*4} {'-'*8} {'-'*9} {'-'*11} {'-'*10} {'-'*10}")
            for s in per_storm:
                print(f"  {s['sid']:<16} {s['season']:>5} {s['n_sequences']:>4} "
                      f"{s['mae_24h']:>7.1f}  {s['mae_48h']:>8.1f}  "
                      f"{s['cliper_mae_24h']:>10.1f}  {s['skill_24h']:>8.1f}%  {s['skill_48h']:>8.1f}%")

            # Storms sorted by error (worst on top)
            worst = sorted(per_storm, key=lambda x: -x["mae_24h"])[:5]
            best = sorted(per_storm, key=lambda x: x["mae_24h"])[:5]
            print(f"\n  Top 5 WORST (MAE 24h cao nhất):")
            for s in worst:
                print(f"    {s['sid']:<16} MAE 24h={s['mae_24h']:.1f} km  Skill={s['skill_24h']:.1f}%  (n={s['n_sequences']})")
            print(f"\n  Top 5 BEST  (MAE 24h thấp nhất):")
            for s in best:
                print(f"    {s['sid']:<16} MAE 24h={s['mae_24h']:.1f} km  Skill={s['skill_24h']:.1f}%  (n={s['n_sequences']})")

            # Lưu JSON cho phân tích sau
            import json as _json
            per_storm_path = fig_dir / "per_storm_metrics.json"
            fig_dir.mkdir(parents=True, exist_ok=True)
            with open(per_storm_path, "w", encoding="utf-8") as f:
                _json.dump({
                    "tag":         tag,
                    "best_model":  _best_name,
                    "n_storms":    len(per_storm),
                    "per_storm":   per_storm,
                }, f, indent=2, ensure_ascii=False)
            print(f"\n  [json] {per_storm_path}")
    else:
        print("\n[3b] Per-storm MAE: skipped (sid_test không có trong .npz — chạy lại G3 để có)")

    # --- Trajectory labels ---
    print("\n[4] Phân loại quỹ đạo...")
    traj_labels = classify_trajectories(X_test, scaler)
    type_counts = {t: int((traj_labels == t).sum())
                   for t in ["westward", "recurving", "southward", "erratic"]}
    print(f"  {type_counts}")

    # --- Plots ---
    print("\n[5] Vẽ biểu đồ...")

    # Best model cho scatter (model có mae_24h nhỏ nhất, bỏ cliper)
    model_only_metrics = {k: v for k, v in all_metrics.items() if k != "cliper"}
    if model_only_metrics:
        best_name = min(model_only_metrics, key=lambda k: model_only_metrics[k]["mae_24h"])
        best_pred = all_preds[best_name]
    else:
        best_name = "cliper"
        best_pred = cliper_pred

    plot_learning_curve(log_path, fig_dir)
    plot_model_comparison(all_metrics, fig_dir)

    if skill_scores_dict:
        plot_skill_scores(skill_scores_dict, float(ckpt_pct), fig_dir)

    plot_scatter(best_pred, y_test, best_name, fig_dir)
    plot_mae_vs_horizon(all_metrics, fig_dir)
    plot_mae_by_type(all_preds, y_test, traj_labels, fig_dir)
    plot_error_histogram(all_metrics, fig_dir)

    # Folium (graceful skip nếu không có folium)
    plot_folium_tracks(X_test, y_test, all_preds, scaler, fig_dir, n_samples=3)

    # --- Checkpoint G7 ---
    print("\n[6] Checkpoint G7...")
    if skill_scores_dict:
        best_skill = max(skill_scores_dict.values())
        best_model_for_skill = max(skill_scores_dict, key=skill_scores_dict.get)
        pass_g7 = best_skill >= float(ckpt_pct)
        status = "PASS" if pass_g7 else "FAIL"
        print(f"  Best Skill Score (24h): {best_skill:.1f}%  "
              f"[{best_model_for_skill.upper()}]  "
              f"| Threshold: {ckpt_pct}%  => {status}")
    else:
        print(f"  Chưa có model checkpoint — chỉ có CLIPER (Skill = 0%)  => FAIL")

    print("\n  G7 hoàn thành.")


if __name__ == "__main__":
    main()
