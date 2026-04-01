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
from src.g5_g6_train.utils import log_result


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

def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    pred, true: (N, 4) = [lat_24h, lon_24h, lat_48h, lon_48h] in degrees.
    Returns dict với mae_24h, rmse_24h, mae_48h, rmse_48h.
    """
    err_24 = haversine_km(pred[:, 0], pred[:, 1], true[:, 0], true[:, 1])
    err_48 = haversine_km(pred[:, 2], pred[:, 3], true[:, 2], true[:, 3])
    return {
        "mae_24h":  float(np.mean(err_24)),
        "rmse_24h": float(np.sqrt(np.mean(err_24 ** 2))),
        "mae_48h":  float(np.mean(err_48)),
        "rmse_48h": float(np.sqrt(np.mean(err_48 ** 2))),
        "err_24":   err_24,   # giữ lại để vẽ histogram
        "err_48":   err_48,
    }


def skill_score(cliper_mae: float, model_mae: float) -> float:
    """Skill Score (%) = (cliper - model) / cliper * 100."""
    if cliper_mae == 0:
        return 0.0
    return (cliper_mae - model_mae) / cliper_mae * 100.0


# ─── CLIPER baseline ──────────────────────────────────────────────────────────

def cliper_predict(X_test: np.ndarray, scaler) -> np.ndarray:
    """
    CLIPER: persistence extrapolation từ bước cuối của X_test.
    X_test: (N, 8, 12) — đã StandardScaler.
    Trả về (N, 4) = [lat_24h, lon_24h, lat_48h, lon_48h] in degrees.
    """
    N = X_test.shape[0]
    # Inverse transform last time-step
    last_step_scaled = X_test[:, -1, :]           # (N, 12)
    last_step_orig   = scaler.inverse_transform(last_step_scaled)  # (N, 12)

    # Denormalize position
    lat_current = last_step_orig[:, 0] * 14.0 + 8.0    # lat_norm → lat
    lon_current = last_step_orig[:, 1] * 18.0 + 102.0  # lon_norm → lon

    # Last displacement (degrees / 6h)
    dlat_last = last_step_orig[:, 2]  # dlat
    dlon_last = last_step_orig[:, 3]  # dlon

    # CLIPER predictions
    lat_24h = lat_current + 4 * dlat_last
    lon_24h = lon_current + 4 * dlon_last
    lat_48h = lat_current + 8 * dlat_last
    lon_48h = lon_current + 8 * dlon_last

    return np.stack([lat_24h, lon_24h, lat_48h, lon_48h], axis=1)  # (N, 4)


# ─── Trajectory classification ────────────────────────────────────────────────

def classify_trajectories(X_test: np.ndarray, scaler) -> np.ndarray:
    """
    Phân loại trajectory từ 4 bước cuối của X_test.
    Returns: array of str (N,) — "westward" | "recurving" | "southward" | "erratic"
    """
    N = X_test.shape[0]
    # Lấy 4 bước cuối, inverse transform
    last4_scaled = X_test[:, -4:, :]  # (N, 4, 12)
    last4_flat   = last4_scaled.reshape(N * 4, 12)
    last4_orig   = scaler.inverse_transform(last4_flat).reshape(N, 4, 12)

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

def load_model_checkpoint(model_name: str, cfg: dict):
    """
    Thử load từ models/checkpoints/best_{name}.pt trước,
    rồi models/final/{name}.pt.
    Trả về (model, ckpt_path) hoặc (None, None) nếu không tìm thấy.
    """
    candidates = [
        BASE_DIR / cfg["output"]["checkpoint_dir"] / f"best_{model_name}.pt",
        BASE_DIR / "models" / "final" / f"{model_name}.pt",
    ]
    found_path = None
    for p in candidates:
        if p.exists():
            found_path = p
            break

    if found_path is None:
        return None, None

    try:
        model = build_model(model_name, cfg)
        state_dict = torch.load(str(found_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"  [load] {model_name.upper()} <- {found_path}")
        return model, found_path
    except Exception as e:
        print(f"  [warn] Không load được {model_name}: {e}")
        return None, None


def model_predict(model, X_test: np.ndarray) -> np.ndarray:
    """
    Chạy inference, trả về (N, 4) degrees.
    """
    model.eval()
    with torch.no_grad():
        X_t  = torch.from_numpy(X_test).float()
        pred = model(X_t).cpu().numpy()  # (N, 4)
    return pred


# ─── Plots ────────────────────────────────────────────────────────────────────

COLORS = {
    "cliper":      "#6b7280",
    "lstm":        "#3b82f6",
    "bilstm":      "#10b981",
    "transformer": "#f59e0b",
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

    for ax, h_idx, label in [(axes[0], 0, "24h"), (axes[1], 2, "48h")]:
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
        maes_by_type = []
        for t in types:
            mask = traj_labels == t
            if mask.sum() == 0:
                maes_by_type.append(0.0)
            else:
                err = haversine_km(pred[mask, 0], pred[mask, 1],
                                   y_test[mask, 0], y_test[mask, 1])
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
        true_lat24, true_lon24 = y_test[idx, 0], y_test[idx, 1]
        true_lat48, true_lon48 = y_test[idx, 2], y_test[idx, 3]

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
            lat24p, lon24p = pred[idx, 0], pred[idx, 1]
            lat48p, lon48p = pred[idx, 2], pred[idx, 3]
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
    print("\n" + "=" * 60)
    print("  G7 — Evaluate: CLIPER baseline + Model metrics")
    print("=" * 60)

    cfg = load_config()

    # Paths
    seq_path    = BASE_DIR / "data" / "features" / "sequences.npz"
    scaler_path = BASE_DIR / cfg["output"]["scaler_path"]
    log_path    = BASE_DIR / cfg["output"]["results_log"]
    fig_dir     = BASE_DIR / cfg["output"]["figures_dir"]
    ckpt_pct    = cfg["checkpoints"]["g7_min_skill_score_pct"]

    # --- Load data ---
    print("\n[1] Load dữ liệu test...")
    if not seq_path.exists():
        raise FileNotFoundError(f"Không tìm thấy sequences.npz: {seq_path}")
    data = np.load(str(seq_path))
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")

    if not scaler_path.exists():
        raise FileNotFoundError(f"Không tìm thấy scaler.pkl: {scaler_path}")
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"  Scaler: {scaler_path}")

    # --- CLIPER baseline ---
    print("\n[2] CLIPER baseline...")
    cliper_pred = cliper_predict(X_test, scaler)
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
    model_names = ["lstm", "bilstm", "transformer"]
    all_preds   = {"cliper": cliper_pred}
    all_metrics = {"cliper": cliper_metrics}
    skill_scores_dict = {}

    for mname in model_names:
        model, ckpt_path = load_model_checkpoint(mname, cfg)
        if model is None:
            print(f"  [skip] {mname.upper()} — không tìm thấy checkpoint")
            continue

        pred    = model_predict(model, X_test)
        metrics = compute_metrics(pred, y_test)
        ss_24   = skill_score(cliper_metrics["mae_24h"], metrics["mae_24h"])
        ss_48   = skill_score(cliper_metrics["mae_48h"], metrics["mae_48h"])

        print(f"  {mname.upper()}")
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
            "model_name":     mname,
            "mae_24h":        round(metrics["mae_24h"], 2),
            "rmse_24h":       round(metrics["rmse_24h"], 2),
            "mae_48h":        round(metrics["mae_48h"], 2),
            "rmse_48h":       round(metrics["rmse_48h"], 2),
            "skill_score_24h": round(ss_24, 2),
            "skill_score_48h": round(ss_48, 2),
            "checkpoint":     str(ckpt_path),
        })

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
