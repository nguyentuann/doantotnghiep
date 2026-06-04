"""
analyze.py — G7 Analysis: Stratified error + Failure mode + Attention
----------------------------------------------------------------------
Phân tích sâu kết quả best model (Transformer scs_v9 mặc định):
  C1. MAE theo trajectory type / horizon / position / intensity / age
  C2. Attention weight visualization (Transformer only)
  C3. Top-N failure cases với pattern analysis

Chạy:
    python -m src.g7_evaluate.analyze --tag scs_v9 --model transformer

Outputs:
    results/figures/{tag}_analysis/
        stratified_mae.png
        mae_by_horizon.png
        mae_by_position.png
        mae_by_intensity.png
        attention_avg.png
        worst_cases.png
        analysis_report.md
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.g4_models import build_model, load_config
from src.g5_g6_train.trainer import cliper_from_X
from src.g7_evaluate.evaluate import (
    _build_model_from_state,
    classify_trajectories,
    haversine_km,
    load_seed_checkpoints,
    model_predict,
    skill_score,
)


BASE_DIR = Path(__file__).parent.parent.parent


# ─── Helper: compute per-step error ──────────────────────────────────────────

def per_step_mae(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """pred, true: (N, n_steps*2). Returns array (n_steps,) of MAE in km."""
    n_steps = pred.shape[1] // 2
    p = pred.reshape(-1, n_steps, 2)
    t = true.reshape(-1, n_steps, 2)
    maes = []
    for s in range(n_steps):
        err = haversine_km(p[:, s, 0], p[:, s, 1], t[:, s, 0], t[:, s, 1])
        maes.append(float(err.mean()))
    return np.array(maes)


def per_sample_err_24h(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-sample Haversine error at 24h step."""
    n_steps = pred.shape[1] // 2
    idx_24 = 6 if n_steps == 8 else 0  # step 4 (24h) flat index = 6
    return haversine_km(pred[:, idx_24], pred[:, idx_24+1],
                        true[:, idx_24], true[:, idx_24+1])


# ─── C1: Stratified MAE ──────────────────────────────────────────────────────

def stratified_mae_by_type(err_24: np.ndarray, traj_labels: np.ndarray) -> dict:
    """MAE 24h grouped by trajectory type."""
    types = ["westward", "recurving", "southward", "erratic"]
    out = {}
    for t in types:
        mask = traj_labels == t
        if mask.sum() == 0:
            out[t] = {"mae": float("nan"), "n": 0}
        else:
            out[t] = {"mae": float(err_24[mask].mean()), "n": int(mask.sum())}
    return out


def mae_by_intensity(err_24: np.ndarray, X_test: np.ndarray, scaler) -> dict:
    """Bin theo vmax (feature index 6). Returns dict {bin_label: {mae, n}}."""
    last_step = scaler.inverse_transform(X_test[:, -1, :])
    vmax = last_step[:, 6]  # vmax index = 6 trong 27 features

    bins  = [0, 35, 64, 96, 200]   # TS/TD, Cat1, Cat2-3, Cat4-5 (kt)
    labels = ["TS/TD (<35kt)", "Cat1 (35-64)", "Cat2-3 (64-96)", "Cat4-5 (>96)"]
    out = {}
    for i, lab in enumerate(labels):
        mask = (vmax >= bins[i]) & (vmax < bins[i+1])
        if mask.sum() == 0:
            out[lab] = {"mae": float("nan"), "n": 0}
        else:
            out[lab] = {"mae": float(err_24[mask].mean()), "n": int(mask.sum())}
    return out


def mae_by_position(err_24: np.ndarray, X_test: np.ndarray, scaler) -> tuple:
    """2D grid: MAE theo (lat, lon) bin của initial position."""
    last_step = scaler.inverse_transform(X_test[:, -1, :])
    lat = last_step[:, 0] * 14.0 + 8.0      # lat_norm → [8, 22]
    lon = last_step[:, 1] * 18.0 + 102.0    # lon_norm → [102, 120]

    lat_bins = np.linspace(8, 22, 5)        # 4 bins
    lon_bins = np.linspace(102, 120, 5)
    grid_mae = np.full((4, 4), np.nan)
    grid_n   = np.zeros((4, 4), dtype=int)

    for i in range(4):
        for j in range(4):
            mask = ((lat >= lat_bins[i]) & (lat < lat_bins[i+1])
                    & (lon >= lon_bins[j]) & (lon < lon_bins[j+1]))
            if mask.sum() > 0:
                grid_mae[i, j] = err_24[mask].mean()
                grid_n[i, j]   = int(mask.sum())
    return grid_mae, grid_n, lat_bins, lon_bins


def mae_by_storm_age(err_24: np.ndarray, X_test: np.ndarray, scaler) -> dict:
    """Bin theo storm_age_h (feature 11)."""
    last_step = scaler.inverse_transform(X_test[:, -1, :])
    age_h = last_step[:, 11]
    bins   = [0, 24, 72, 168, 1000]
    labels = ["new (<24h)", "young (1-3d)", "mature (3-7d)", "old (>7d)"]
    out = {}
    for i, lab in enumerate(labels):
        mask = (age_h >= bins[i]) & (age_h < bins[i+1])
        if mask.sum() == 0:
            out[lab] = {"mae": float("nan"), "n": 0}
        else:
            out[lab] = {"mae": float(err_24[mask].mean()), "n": int(mask.sum())}
    return out


# ─── C2: Attention extraction (Transformer) ──────────────────────────────────

def extract_attention(model, X_test: np.ndarray, lookback: int) -> np.ndarray:
    """
    Trả về attention weights trung bình: shape (n_layers, lookback, lookback).
    Mỗi cell [i, j] = mean attention từ query_pos_i → key_pos_j.
    """
    model.eval()
    attn_maps = []

    # Hook để capture attention output
    def make_hook(layer_idx):
        def hook(module, input, output):
            # MultiheadAttention forward: input = (query, key, value, ...)
            # Need to recompute với need_weights=True
            pass
        return hook

    # Cách đơn giản: chạy thủ công từng layer với need_weights=True
    with torch.no_grad():
        X_t = torch.from_numpy(X_test[:200]).float()  # subset cho tốc độ
        x = model.input_proj(X_t) + model.pos_embed   # [B, L, D]

        for layer in model.encoder.layers:
            # Pre-LN: norm trước attention
            x_norm = layer.norm1(x)
            attn_out, attn_w = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=True,  # average across heads
            )
            attn_maps.append(attn_w.mean(dim=0).cpu().numpy())  # avg across batch
            # Tiếp tục forward để layer sau có input đúng
            x = x + layer.dropout1(attn_out)
            x = x + layer.dropout2(layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x))))))

    return np.stack(attn_maps)  # (n_layers, L, L)


# ─── C3: Worst-case analysis ─────────────────────────────────────────────────

def worst_cases(err_24: np.ndarray, pred: np.ndarray, true: np.ndarray,
                X_test: np.ndarray, scaler, top_n: int = 20) -> dict:
    """Lấy top-N worst predictions, trả về thông tin chi tiết."""
    worst_idx = np.argsort(err_24)[-top_n:][::-1]
    last_step = scaler.inverse_transform(X_test[:, -1, :])

    out = []
    for rank, idx in enumerate(worst_idx, 1):
        out.append({
            "rank": rank,
            "idx":  int(idx),
            "err_24h_km":   float(err_24[idx]),
            "init_lat":     float(last_step[idx, 0] * 14.0 + 8.0),
            "init_lon":     float(last_step[idx, 1] * 18.0 + 102.0),
            "vmax":         float(last_step[idx, 6]),
            "storm_age_h":  float(last_step[idx, 11]),
            "dlat":         float(last_step[idx, 2]),
            "dlon":         float(last_step[idx, 3]),
        })
    return {"top_n": top_n, "cases": out, "indices": worst_idx.tolist()}


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_stratified_by_type(stats: dict, fig_dir: Path):
    types  = list(stats.keys())
    maes   = [stats[t]["mae"] for t in types]
    counts = [stats[t]["n"]   for t in types]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
    bars = ax.bar([t.capitalize() for t in types], maes, color=colors, alpha=0.85)
    ax.set_ylabel("MAE 24h (km)")
    ax.set_title("MAE 24h theo loại quỹ đạo (Transformer scs_v9)")
    ax.grid(axis="y", alpha=0.3)
    for bar, m, n in zip(bars, maes, counts):
        if not np.isnan(m):
            ax.annotate(f"{m:.1f}\nn={n}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9)
    fig.tight_layout()
    out = fig_dir / "stratified_mae.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_mae_by_horizon(per_step: np.ndarray, fig_dir: Path):
    hours = [(i+1) * 6 for i in range(len(per_step))]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hours, per_step, "o-", linewidth=2, color="#f59e0b", markersize=8)
    for h, m in zip(hours, per_step):
        ax.annotate(f"{m:.0f}", (h, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("Horizon (giờ)")
    ax.set_ylabel("MAE (km)")
    ax.set_title("MAE theo từng horizon (6h → 48h)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "mae_by_horizon.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_mae_by_intensity(stats: dict, fig_dir: Path):
    labels = list(stats.keys())
    maes   = [stats[l]["mae"] for l in labels]
    counts = [stats[l]["n"]   for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, maes, color="#8b5cf6", alpha=0.85)
    ax.set_ylabel("MAE 24h (km)")
    ax.set_title("MAE 24h theo cường độ bão (vmax tại thời điểm dự đoán)")
    ax.grid(axis="y", alpha=0.3)
    for bar, m, n in zip(bars, maes, counts):
        if not np.isnan(m):
            ax.annotate(f"{m:.1f}\nn={n}",
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=9)
    fig.tight_layout()
    out = fig_dir / "mae_by_intensity.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_mae_by_position(grid_mae, grid_n, lat_bins, lon_bins, fig_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(grid_mae, origin="lower", aspect="auto", cmap="RdYlGn_r",
                   extent=[lon_bins[0], lon_bins[-1], lat_bins[0], lat_bins[-1]],
                   vmin=50, vmax=200)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("MAE 24h theo vị trí khởi tạo (Biển Đông)")
    plt.colorbar(im, ax=ax, label="MAE 24h (km)")

    # Annotate
    for i in range(4):
        for j in range(4):
            if grid_n[i, j] > 0:
                lat_c = (lat_bins[i] + lat_bins[i+1]) / 2
                lon_c = (lon_bins[j] + lon_bins[j+1]) / 2
                ax.text(lon_c, lat_c, f"{grid_mae[i,j]:.0f}\nn={grid_n[i,j]}",
                        ha="center", va="center", fontsize=9,
                        color="white" if grid_mae[i,j] > 130 else "black",
                        fontweight="bold")
    fig.tight_layout()
    out = fig_dir / "mae_by_position.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_attention(attn: np.ndarray, fig_dir: Path):
    n_layers, L, _ = attn.shape
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 4.5))
    if n_layers == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.imshow(attn[i], cmap="viridis", vmin=0, vmax=attn.max())
        ax.set_title(f"Layer {i+1}: Attention pattern")
        ax.set_xlabel("Key position (timestep)")
        ax.set_ylabel("Query position")
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out = fig_dir / "attention_avg.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_attention_distribution(attn: np.ndarray, fig_dir: Path):
    """Distribution attention từ last query token (token cuối — token dùng để predict)."""
    n_layers, L, _ = attn.shape
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(L)
    width = 0.8 / n_layers
    for i in range(n_layers):
        last_q_attn = attn[i, -1, :]  # query = last token
        ax.bar(x + i*width - 0.4 + width/2, last_q_attn, width,
               label=f"Layer {i+1}", alpha=0.85)
    ax.set_xlabel("Key timestep (1=cũ nhất, 8=hiện tại)")
    ax.set_ylabel("Attention weight")
    ax.set_title("Attention từ token cuối (token dùng để predict)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"t-{L-i-1}" if i < L-1 else "t" for i in range(L)])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = fig_dir / "attention_last_query.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_worst_cases(worst: dict, X_test, y_test, pred, scaler, fig_dir: Path):
    """Plot top 6 worst predictions trên cùng 1 figure."""
    cases = worst["cases"][:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for ax, c in zip(axes.flat, cases):
        idx = c["idx"]
        last8_orig = scaler.inverse_transform(X_test[idx])
        lat_hist = last8_orig[:, 0] * 14.0 + 8.0
        lon_hist = last8_orig[:, 1] * 18.0 + 102.0
        # True future
        n_steps = y_test.shape[1] // 2
        lat_true = y_test[idx, 0::2]
        lon_true = y_test[idx, 1::2]
        lat_pred = pred[idx, 0::2]
        lon_pred = pred[idx, 1::2]

        ax.plot(lon_hist, lat_hist, "o-", color="gray", label="Lịch sử", markersize=5)
        ax.plot(lon_true, lat_true, "s-", color="black", label="Thực tế", markersize=5)
        ax.plot(lon_pred, lat_pred, "x--", color="red", label="Dự đoán", markersize=8)
        ax.scatter([lon_hist[-1]], [lat_hist[-1]], color="blue", s=80, zorder=5, label="Init")
        ax.set_title(f"#{c['rank']}: err 24h={c['err_24h_km']:.0f}km | vmax={c['vmax']:.0f}kt")
        ax.set_xlabel("Lon"); ax.set_ylabel("Lat")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
        ax.set_xlim(102, 120); ax.set_ylim(8, 22)

    fig.suptitle("Top-6 worst predictions", fontsize=14)
    fig.tight_layout()
    out = fig_dir / "worst_cases.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


# ─── Markdown report ─────────────────────────────────────────────────────────

def write_report(tag: str, model_name: str, overall: dict, by_type: dict,
                 by_horizon: np.ndarray, by_intensity: dict, by_age: dict,
                 by_position, attn: np.ndarray, worst: dict, fig_dir: Path):
    grid_mae, grid_n, lat_bins, lon_bins = by_position
    md = []
    md.append(f"# Analysis Report — {model_name.upper()} ({tag})\n")
    md.append("## Tổng quan\n")
    md.append(f"- N test samples (SCS): **{overall['n_test']}**")
    md.append(f"- MAE 24h: **{overall['mae_24h']:.1f} km**")
    md.append(f"- MAE 48h: **{overall['mae_48h']:.1f} km**")
    md.append(f"- Skill 24h: **{overall['skill_24h']:.1f}%**")
    md.append(f"- Skill 48h: **{overall['skill_48h']:.1f}%**\n")

    md.append("## C1.1 — MAE theo loại quỹ đạo\n")
    md.append("| Loại | N | MAE 24h | Note |")
    md.append("|------|---|---------|------|")
    for t, s in by_type.items():
        note = ""
        if s["n"] >= 20 and not np.isnan(s["mae"]):
            if s["mae"] > overall["mae_24h"] * 1.2:
                note = "⚠️ Cao hơn TB"
            elif s["mae"] < overall["mae_24h"] * 0.8:
                note = "✅ Tốt"
        md.append(f"| {t.capitalize()} | {s['n']} | "
                  f"{s['mae']:.1f} km |" if not np.isnan(s['mae'])
                  else f"| {t.capitalize()} | {s['n']} | N/A | quá ít mẫu |")
        if s["n"] >= 20:
            md[-1] = md[-1].rstrip("|") + f" {note} |"
    md.append("")

    md.append("## C1.2 — MAE theo horizon\n")
    md.append("| Step | Horizon | MAE (km) |")
    md.append("|------|---------|----------|")
    for i, m in enumerate(by_horizon):
        md.append(f"| {i+1} | {(i+1)*6}h | {m:.1f} |")
    md.append("")
    # Growth rate
    g_24_48 = by_horizon[7] / by_horizon[3] if len(by_horizon) >= 8 else 1.0
    md.append(f"**Tỷ lệ tăng MAE từ 24h → 48h**: {g_24_48:.2f}× "
              f"(CLIPER thường 2.5-3×, model nên < 2.5×)\n")

    md.append("## C1.3 — MAE theo cường độ (vmax)\n")
    md.append("| Cường độ | N | MAE 24h |")
    md.append("|----------|---|---------|")
    for lab, s in by_intensity.items():
        m = f"{s['mae']:.1f} km" if not np.isnan(s['mae']) else "N/A"
        md.append(f"| {lab} | {s['n']} | {m} |")
    md.append("")

    md.append("## C1.4 — MAE theo storm age\n")
    md.append("| Age | N | MAE 24h |")
    md.append("|-----|---|---------|")
    for lab, s in by_age.items():
        m = f"{s['mae']:.1f} km" if not np.isnan(s['mae']) else "N/A"
        md.append(f"| {lab} | {s['n']} | {m} |")
    md.append("")

    md.append("## C2 — Attention pattern (Transformer)\n")
    md.append(f"Số layer: {attn.shape[0]}, lookback: {attn.shape[1]}\n")
    md.append("**Attention từ token cuối (token dùng để predict) tới các bước quá khứ:**\n")
    md.append("| Layer | t-7 | t-6 | t-5 | t-4 | t-3 | t-2 | t-1 | t |")
    md.append("|-------|-----|-----|-----|-----|-----|-----|-----|---|")
    for li in range(attn.shape[0]):
        row = attn[li, -1, :]
        cells = " | ".join(f"{v:.3f}" for v in row)
        md.append(f"| {li+1} | {cells} |")
    md.append("")

    # Interpretation
    last_layer_attn = attn[-1, -1, :]
    top_3_steps = np.argsort(last_layer_attn)[-3:][::-1]
    interp = []
    for s in top_3_steps:
        interp.append(f"t-{attn.shape[1]-s-1}" if s < attn.shape[1]-1 else "t")
    md.append(f"**Top 3 timesteps được attend nhiều nhất (last layer):** {', '.join(interp)}\n")
    if last_layer_attn[-3:].sum() > 0.7:
        md.append("→ Model chủ yếu dùng 3 bước CUỐI. Lookback dài hơn có thể không cần thiết.\n")
    elif last_layer_attn[:3].sum() > 0.5:
        md.append("→ Model attend vào bước XA quá khứ. Lookback dài có giá trị.\n")
    else:
        md.append("→ Model phân bố attention khá đều. Có thể giảm dropout.\n")

    md.append("## C3 — Top-20 worst cases\n")
    md.append("| Rank | err_24h | Init Lat | Init Lon | vmax | age (h) |")
    md.append("|------|---------|----------|----------|------|---------|")
    for c in worst["cases"]:
        md.append(f"| {c['rank']} | {c['err_24h_km']:.0f} km | "
                  f"{c['init_lat']:.2f} | {c['init_lon']:.2f} | "
                  f"{c['vmax']:.0f} kt | {c['storm_age_h']:.0f} |")
    md.append("")

    # Stats về worst cases
    worst_lats = np.array([c["init_lat"] for c in worst["cases"]])
    worst_lons = np.array([c["init_lon"] for c in worst["cases"]])
    worst_vmax = np.array([c["vmax"] for c in worst["cases"]])
    md.append(f"**Worst cases stats:** mean init_lat={worst_lats.mean():.1f}°N, "
              f"mean init_lon={worst_lons.mean():.1f}°E, mean vmax={worst_vmax.mean():.0f} kt\n")

    md.append("## Kết luận đề xuất\n")
    md.append("Dựa trên phân tích trên, các hướng cải tiến có cơ sở:\n")

    out_md = fig_dir / "analysis_report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  [report] {out_md}")
    return md


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="scs_v9")
    parser.add_argument("--model", default="transformer",
                        help="Model để phân tích: lstm/bilstm/bigru/transformer")
    args = parser.parse_args()
    tag = args.tag
    model_name = args.model
    suffix = f"_{tag}" if tag else ""

    print("=" * 60)
    print(f"  G7 ANALYZE — {model_name.upper()} ({tag})")
    print("=" * 60)

    cfg = load_config()

    # Paths
    seq_path    = BASE_DIR / "data" / "features" / f"sequences{suffix}.npz"
    scaler_path = BASE_DIR / cfg["output"]["scaler_path"].replace(".pkl", f"{suffix}.pkl")
    fig_dir     = BASE_DIR / cfg["output"]["figures_dir"] / f"{tag}_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load test data (SCS-only)
    data = np.load(str(seq_path), allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)
    if "in_scs_test" in data.files:
        mask = data["in_scs_test"].astype(bool)
        X_test, y_test = X_test[mask], y_test[mask]
    print(f"[1] X_test: {X_test.shape}, y_test: {y_test.shape}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # CLIPER baseline
    anchor_steps = cfg["model"].get("anchor_steps", [4, 8])
    cliper_pred = cliper_from_X(X_test, scaler, anchor_steps=anchor_steps)
    cliper_err_24 = per_sample_err_24h(cliper_pred, y_test)
    cliper_mae_24 = cliper_err_24.mean()
    cliper_mae_48 = haversine_km(cliper_pred[:, 14], cliper_pred[:, 15],
                                 y_test[:, 14], y_test[:, 15]).mean()
    print(f"[2] CLIPER MAE 24h = {cliper_mae_24:.1f} km")

    # Load multi-seed Transformer
    print(f"[3] Load {model_name} seed checkpoints...")
    seed_models = load_seed_checkpoints(model_name, cfg, tag=tag)
    if not seed_models:
        raise FileNotFoundError(f"Không tìm thấy seed checkpoints cho {model_name}_{tag}")
    print(f"    Loaded {len(seed_models)} seeds")

    residual = cfg["model"].get("residual", False)
    seed_preds = [model_predict(m, X_test, cliper_pred=cliper_pred, residual=residual)
                  for m, _ in seed_models]
    pred = np.mean(seed_preds, axis=0)

    err_24 = per_sample_err_24h(pred, y_test)
    mae_24 = err_24.mean()
    mae_48 = haversine_km(pred[:, 14], pred[:, 15],
                          y_test[:, 14], y_test[:, 15]).mean()
    print(f"[4] Ensemble MAE 24h = {mae_24:.1f} km")

    overall = {
        "n_test":    int(len(X_test)),
        "mae_24h":   float(mae_24),
        "mae_48h":   float(mae_48),
        "skill_24h": float(skill_score(cliper_mae_24, mae_24)),
        "skill_48h": float(skill_score(cliper_mae_48, mae_48)),
    }

    # --- C1: Stratified ---
    print("[5] C1 — Stratified analysis...")
    traj_labels = classify_trajectories(X_test, scaler)
    by_type      = stratified_mae_by_type(err_24, traj_labels)
    by_horizon   = per_step_mae(pred, y_test)
    by_intensity = mae_by_intensity(err_24, X_test, scaler)
    by_position  = mae_by_position(err_24, X_test, scaler)
    by_age       = mae_by_storm_age(err_24, X_test, scaler)

    print("  MAE by type:")
    for t, s in by_type.items():
        print(f"    {t:<12} n={s['n']:>4} mae={s['mae']:>6.1f} km"
              if not np.isnan(s['mae']) else
              f"    {t:<12} n={s['n']:>4} (no data)")
    print(f"  MAE by horizon (km): {[f'{m:.0f}' for m in by_horizon]}")
    print("  MAE by intensity:")
    for lab, s in by_intensity.items():
        if not np.isnan(s['mae']):
            print(f"    {lab:<22} n={s['n']:>4} mae={s['mae']:>6.1f} km")

    plot_stratified_by_type(by_type, fig_dir)
    plot_mae_by_horizon(by_horizon, fig_dir)
    plot_mae_by_intensity(by_intensity, fig_dir)
    plot_mae_by_position(*by_position, fig_dir)

    # --- C2: Attention (Transformer only) ---
    attn = None
    if model_name == "transformer":
        print("[6] C2 — Extract attention weights...")
        # Build a single Transformer (use first seed model)
        first_model = seed_models[0][0]
        attn = extract_attention(first_model, X_test, lookback=X_test.shape[1])
        print(f"    Attention shape: {attn.shape}")
        plot_attention(attn, fig_dir)
        plot_attention_distribution(attn, fig_dir)
    else:
        print("[6] C2 — Skipped (chỉ áp dụng cho Transformer)")
        attn = np.zeros((1, X_test.shape[1], X_test.shape[1]))

    # --- C3: Worst cases ---
    print("[7] C3 — Failure mode analysis (top-20)...")
    worst = worst_cases(err_24, pred, y_test, X_test, scaler, top_n=20)
    plot_worst_cases(worst, X_test, y_test, pred, scaler, fig_dir)

    print("\n  Top 5 worst:")
    for c in worst["cases"][:5]:
        print(f"    #{c['rank']} err={c['err_24h_km']:.0f}km "
              f"lat={c['init_lat']:.1f} lon={c['init_lon']:.1f} "
              f"vmax={c['vmax']:.0f}kt age={c['storm_age_h']:.0f}h")

    # --- Report ---
    print("[8] Write analysis report...")
    write_report(tag, model_name, overall, by_type, by_horizon,
                 by_intensity, by_age, by_position, attn, worst, fig_dir)

    # Save raw stats as JSON
    json_out = fig_dir / "analysis_stats.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "tag": tag,
            "model": model_name,
            "overall": overall,
            "by_type": by_type,
            "by_horizon": by_horizon.tolist(),
            "by_intensity": by_intensity,
            "by_age": by_age,
            "worst_cases": worst["cases"],
            "attention_last_query": attn[-1, -1, :].tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"  [json] {json_out}")

    print("\n" + "=" * 60)
    print(f"  Analysis hoàn thành — output: {fig_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
