"""
feature_importance.py — Permutation Importance cho champion model
------------------------------------------------------------------
Mỗi feature, shuffle giá trị across test set → đo MAE 24h tăng bao nhiêu.
Feature càng quan trọng, MAE tăng càng nhiều khi shuffle.

Chạy:
    python -m src.g7_evaluate.feature_importance --tag scs_v12_lb6

Output:
    results/figures/{tag}_analysis/feature_importance.png
    results/figures/{tag}_analysis/feature_importance.json
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

from src.g4_models import load_config
from src.g5_g6_train.trainer import cliper_from_X
from src.g7_evaluate.evaluate import (
    load_seed_checkpoints,
    model_predict,
    haversine_km,
    _get_24h_48h_indices,
)


BASE_DIR = Path(__file__).parent.parent.parent


# Feature groups cho group-level importance
FEATURE_GROUPS_SCS_V12 = {
    "Storm-state":      list(range(0, 13)),     # lat_norm..dist2land
    "Wind-shear":       [13],                    # wind_shear
    "Mid-level ERA5":   list(range(14, 19)),    # u500, v500, u700, v700, z500
    "Upper-low ERA5":   list(range(19, 23)),    # u200, v200, u850, v850
    "Vorticity":        [23, 24],                # vort500, vort850
    "Beta drift":       [25, 26],                # beta_lat/lon_drift
    "DLM Annulus":      [27, 28],                # asteer_u, asteer_v
    "850 hPa Annulus":  [29, 30],                # asteer_u850, asteer_v850
}


def compute_mae_24h(pred, y_test):
    """Compute MAE 24h từ pred + true (Haversine km)."""
    idx_24, _ = _get_24h_48h_indices(pred.shape[1])
    err = haversine_km(pred[:, idx_24], pred[:, idx_24+1],
                       y_test[:, idx_24], y_test[:, idx_24+1])
    return float(err.mean())


def predict_with_ensemble(seed_models, X, cliper_pred, residual):
    """Predict bằng 3-seed ensemble (average)."""
    preds = []
    for model, _ in seed_models:
        p = model_predict(model, X, cliper_pred=cliper_pred, residual=residual)
        preds.append(p)
    return np.mean(preds, axis=0)


def permute_feature(X, feature_idx, rng):
    """Shuffle 1 feature across ALL sequences and timesteps (preserve structure)."""
    X_perm = X.copy()
    # Shuffle theo sample axis (axis 0), giữ nguyên temporal structure trong mỗi sample
    perm = rng.permutation(X.shape[0])
    X_perm[:, :, feature_idx] = X[perm, :, feature_idx]
    return X_perm


def permute_group(X, feature_indices, rng):
    """Shuffle CẢ NHÓM features cùng lúc (preserve correlation trong group)."""
    X_perm = X.copy()
    perm = rng.permutation(X.shape[0])
    for idx in feature_indices:
        X_perm[:, :, idx] = X[perm, :, idx]
    return X_perm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="scs_v12_lb6")
    parser.add_argument("--model", default="transformer")
    parser.add_argument("--n-repeats", type=int, default=10,
                        help="Số lần lặp lại permutation cho stability (default 10)")
    parser.add_argument("--scs-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  FEATURE IMPORTANCE — {args.model.upper()} {args.tag}")
    print(f"  Permutation repeats: {args.n_repeats}")
    print("=" * 60)

    cfg = load_config()
    suffix = f"_{args.tag}" if args.tag else ""

    # Paths
    seq_path    = BASE_DIR / "data" / "features" / f"sequences{suffix}.npz"
    scaler_path = BASE_DIR / cfg["output"]["scaler_path"].replace(".pkl", f"{suffix}.pkl")
    sub_dir     = (args.tag if args.tag else "legacy") + ("_scs" if args.scs_only else "")
    fig_dir     = BASE_DIR / cfg["output"]["figures_dir"] / f"{args.tag}_analysis"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Feature names
    feat_names = cfg["features"]["names"]

    # --- Load data ---
    print("\n[1] Load test data...")
    data = np.load(str(seq_path), allow_pickle=True)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.float32)

    if args.scs_only and "in_scs_test" in data.files:
        mask = data["in_scs_test"].astype(bool)
        X_test = X_test[mask]
        y_test = y_test[mask]
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Features: {len(feat_names)}")

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    # --- Load 3-seed ensemble ---
    print(f"\n[2] Load {args.model} seed checkpoints...")
    seed_models = load_seed_checkpoints(args.model, cfg, tag=args.tag)
    if not seed_models:
        raise FileNotFoundError(f"Không tìm thấy seeds cho {args.model}_{args.tag}")
    print(f"  Loaded {len(seed_models)} seeds")

    # --- Compute baseline MAE ---
    print("\n[3] Compute baseline MAE 24h...")
    anchor_steps = cfg["model"].get("anchor_steps", [4, 8])
    residual = cfg["model"].get("residual", False)
    cliper_pred = cliper_from_X(X_test, scaler, anchor_steps=anchor_steps) if residual else None

    pred_baseline = predict_with_ensemble(seed_models, X_test, cliper_pred, residual)
    baseline_mae = compute_mae_24h(pred_baseline, y_test)
    print(f"  Baseline MAE 24h: {baseline_mae:.2f} km")

    # --- Per-feature permutation importance ---
    print(f"\n[4] Per-feature permutation importance ({args.n_repeats} repeats)...")
    rng = np.random.default_rng(42)
    n_features = X_test.shape[2]
    per_feat_results = []

    for feat_idx in range(n_features):
        feat_name = feat_names[feat_idx] if feat_idx < len(feat_names) else f"feat_{feat_idx}"
        mae_increases = []
        for rep in range(args.n_repeats):
            X_perm = permute_feature(X_test, feat_idx, rng)
            # CLIPER cũng phải tính lại nếu permute lat_norm/lon_norm/dlat/dlon (features 0-3)
            if feat_idx in [0, 1, 2, 3] and residual:
                cliper_pred_perm = cliper_from_X(X_perm, scaler, anchor_steps=anchor_steps)
            else:
                cliper_pred_perm = cliper_pred
            pred_perm = predict_with_ensemble(seed_models, X_perm, cliper_pred_perm, residual)
            mae_perm = compute_mae_24h(pred_perm, y_test)
            mae_increases.append(mae_perm - baseline_mae)

        mean_inc = float(np.mean(mae_increases))
        std_inc  = float(np.std(mae_increases))
        per_feat_results.append({
            "feature":     feat_name,
            "index":       feat_idx,
            "mae_increase": round(mean_inc, 2),
            "mae_std":     round(std_inc, 2),
        })
        print(f"  {feat_idx:2d}. {feat_name:<25} +{mean_inc:>7.2f} km  (±{std_inc:.2f})")

    # Sort by importance
    per_feat_results.sort(key=lambda x: -x["mae_increase"])

    # --- Per-group permutation importance ---
    print(f"\n[5] Per-GROUP permutation importance...")
    per_group_results = []
    for group_name, indices in FEATURE_GROUPS_SCS_V12.items():
        if max(indices) >= n_features:
            continue  # skip if features beyond n_features
        mae_increases = []
        for rep in range(args.n_repeats):
            X_perm = permute_group(X_test, indices, rng)
            need_cliper_recompute = any(i in [0,1,2,3] for i in indices) and residual
            if need_cliper_recompute:
                cliper_pred_perm = cliper_from_X(X_perm, scaler, anchor_steps=anchor_steps)
            else:
                cliper_pred_perm = cliper_pred
            pred_perm = predict_with_ensemble(seed_models, X_perm, cliper_pred_perm, residual)
            mae_perm = compute_mae_24h(pred_perm, y_test)
            mae_increases.append(mae_perm - baseline_mae)

        mean_inc = float(np.mean(mae_increases))
        per_group_results.append({
            "group":         group_name,
            "n_features":    len(indices),
            "feature_idxs":  indices,
            "mae_increase":  round(mean_inc, 2),
            "mae_per_feat":  round(mean_inc / len(indices), 2),
        })
        print(f"  {group_name:<20} ({len(indices)} feat) +{mean_inc:>7.2f} km  "
              f"(per-feat: +{mean_inc/len(indices):.2f} km)")

    per_group_results.sort(key=lambda x: -x["mae_increase"])

    # --- Plots ---
    print(f"\n[6] Generate plots...")
    _plot_feature_importance(per_feat_results, baseline_mae, fig_dir)
    _plot_group_importance(per_group_results, baseline_mae, fig_dir)

    # --- Save JSON ---
    json_out = fig_dir / "feature_importance.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "tag":           args.tag,
            "model":         args.model,
            "n_repeats":     args.n_repeats,
            "baseline_mae_24h": round(baseline_mae, 2),
            "per_feature":   per_feat_results,
            "per_group":     per_group_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  [json] {json_out}")

    print(f"\n  ✓ Feature importance analysis done")
    print(f"    Baseline MAE 24h = {baseline_mae:.1f} km")
    print(f"    Top 3 features:")
    for r in per_feat_results[:3]:
        print(f"      {r['feature']:<25} +{r['mae_increase']:.1f} km")
    print(f"    Top 3 groups:")
    for r in per_group_results[:3]:
        print(f"      {r['group']:<20} +{r['mae_increase']:.1f} km")


def _plot_feature_importance(results, baseline_mae, fig_dir):
    """Bar chart per-feature importance."""
    # Limit to top 20 for readability
    top = results[:20]
    names = [r["feature"] for r in top][::-1]
    incs  = [r["mae_increase"] for r in top][::-1]
    stds  = [r["mae_std"] for r in top][::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(top)))
    bars = ax.barh(names, incs, xerr=stds, color=colors, alpha=0.85, edgecolor="black")
    ax.set_xlabel(f"MAE 24h increase khi shuffle (km)\nBaseline = {baseline_mae:.1f} km")
    ax.set_ylabel("Feature")
    ax.set_title(f"Permutation Feature Importance — Top 20")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5)

    for bar, inc in zip(bars, incs):
        ax.annotate(f"+{inc:.1f}", xy=(inc, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=8)
    fig.tight_layout()
    out = fig_dir / "feature_importance.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def _plot_group_importance(results, baseline_mae, fig_dir):
    """Bar chart per-group importance."""
    names = [r["group"] for r in results]
    incs  = [r["mae_increase"] for r in results]
    per_feat = [r["mae_per_feat"] for r in results]
    n_feats = [r["n_features"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))

    # Left: total importance per group
    ax = axes[0]
    bars = ax.barh([f"{n} ({nf})" for n, nf in zip(names, n_feats)][::-1],
                   incs[::-1], color=colors[::-1], alpha=0.85, edgecolor="black")
    ax.set_xlabel("MAE 24h increase khi shuffle CẢ NHÓM (km)")
    ax.set_title("Per-Group Importance (Total)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    for bar, inc in zip(bars, incs[::-1]):
        ax.annotate(f"+{inc:.1f}", xy=(inc, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=9)

    # Right: per-feature importance
    sorted_by_perfeat = sorted(zip(names, per_feat, n_feats, colors),
                                key=lambda x: x[1])
    names_pf  = [x[0] for x in sorted_by_perfeat]
    perfeat_vals = [x[1] for x in sorted_by_perfeat]
    nf_pf = [x[2] for x in sorted_by_perfeat]
    colors_pf = [x[3] for x in sorted_by_perfeat]

    ax = axes[1]
    bars = ax.barh([f"{n} ({nf})" for n, nf in zip(names_pf, nf_pf)],
                   perfeat_vals, color=colors_pf, alpha=0.85, edgecolor="black")
    ax.set_xlabel("MAE increase per-feature trong nhóm (km / feature)")
    ax.set_title("Per-Group Importance (Normalized by N features)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, perfeat_vals):
        ax.annotate(f"+{val:.1f}", xy=(val, bar.get_y() + bar.get_height()/2),
                    xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=9)

    fig.suptitle(f"Per-Group Feature Importance (Baseline MAE = {baseline_mae:.1f} km)",
                 fontsize=12)
    fig.tight_layout()
    out = fig_dir / "feature_importance_groups.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


if __name__ == "__main__":
    main()
