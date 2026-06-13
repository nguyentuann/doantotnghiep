"""
plot_per_storm.py — Visualization cho per-storm metrics
--------------------------------------------------------
Đọc per_storm_metrics.json từ G7 evaluate, sinh các biểu đồ:
  1. Histogram MAE 24h per storm
  2. Histogram Skill Score per storm
  3. Scatter Skill vs N sequences (xem n nhỏ có variance lớn)
  4. Bar chart: model vs CLIPER cho top-10 storms (theo MAE)
  5. Cumulative distribution function (CDF) of MAE

Chạy:
    python -m src.g7_evaluate.plot_per_storm --tag scs_v12_lb6
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).parent.parent.parent  # source/model_ai/


def _savefig(fig, path: Path, name: str):
    path.mkdir(parents=True, exist_ok=True)
    out = path / name
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {out}")


def plot_mae_histogram(per_storm: list, mean_mae: float, fig_dir: Path):
    maes = np.array([s["mae_24h"] for s in per_storm])
    fig, ax = plt.subplots(figsize=(9, 5))
    n, bins, patches = ax.hist(maes, bins=15, color="#3b82f6", alpha=0.7, edgecolor="black")
    ax.axvline(mean_mae, color="red", linestyle="--", linewidth=2,
               label=f"Mean = {mean_mae:.1f} km")
    ax.axvline(np.median(maes), color="green", linestyle="--", linewidth=2,
               label=f"Median = {np.median(maes):.1f} km")
    ax.set_xlabel("MAE 24h per storm (km)")
    ax.set_ylabel("Số bão")
    ax.set_title(f"Phân phối MAE 24h trên {len(per_storm)} bão test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "per_storm_mae_histogram.png")


def plot_skill_histogram(per_storm: list, fig_dir: Path):
    skills = np.array([s["skill_24h"] for s in per_storm])
    fig, ax = plt.subplots(figsize=(9, 5))
    n_pos = (skills > 0).sum()
    n_neg = (skills < 0).sum()
    n_50  = (skills > 50).sum()

    colors = ["#ef4444" if s < 0 else "#10b981" if s > 50 else "#3b82f6" for s in sorted(skills)]
    ax.hist(skills, bins=20, color="#3b82f6", alpha=0.7, edgecolor="black")
    ax.axvline(0, color="red", linestyle="--", linewidth=2, label="CLIPER (0%)")
    ax.axvline(skills.mean(), color="orange", linestyle="-", linewidth=2,
               label=f"Mean = {skills.mean():.1f}%")
    ax.axvline(20, color="purple", linestyle=":", linewidth=1.5,
               label="Thesis target 20%")

    ax.set_xlabel("Skill Score 24h per storm (%)")
    ax.set_ylabel("Số bão")
    ax.set_title(f"Phân phối Skill Score — {n_pos}/{len(per_storm)} bão beat CLIPER, "
                 f"{n_50} bão > 50% skill")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "per_storm_skill_histogram.png")


def plot_skill_vs_n(per_storm: list, fig_dir: Path):
    """Scatter skill vs N — xem storm có ít sequence có variance cao không."""
    n_seq = np.array([s["n_sequences"] for s in per_storm])
    skill = np.array([s["skill_24h"] for s in per_storm])
    mae   = np.array([s["mae_24h"]   for s in per_storm])

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(n_seq, skill, c=mae, cmap="RdYlGn_r", s=80,
                    alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.set_xlabel("Số sequences trong test set (N)")
    ax.set_ylabel("Skill Score 24h (%)")
    ax.set_title("Skill vs sample size per storm — color = MAE (km)")
    plt.colorbar(sc, ax=ax, label="MAE 24h (km)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "per_storm_skill_vs_n.png")


def plot_model_vs_cliper_topN(per_storm: list, fig_dir: Path, top_n: int = 15):
    """Bar chart so sánh model vs CLIPER MAE cho top-N storms (sorted by MAE)."""
    # Sort theo MAE 24h từ thấp đến cao
    sorted_storms = sorted(per_storm, key=lambda x: x["mae_24h"])
    # Lấy đều: 7 best + 1 middle + 7 worst để có distribution
    if len(sorted_storms) > top_n:
        n_each = top_n // 2
        selected = sorted_storms[:n_each] + sorted_storms[-(top_n - n_each):]
    else:
        selected = sorted_storms

    sids = [s["sid"] for s in selected]
    model_mae  = [s["mae_24h"]        for s in selected]
    cliper_mae = [s["cliper_mae_24h"] for s in selected]

    x = np.arange(len(sids))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(11, len(sids) * 0.7), 6))
    b1 = ax.bar(x - width/2, model_mae,  width, label="Model (Transformer)", color="#3b82f6")
    b2 = ax.bar(x + width/2, cliper_mae, width, label="CLIPER baseline",      color="#6b7280")

    ax.set_xticks(x)
    ax.set_xticklabels(sids, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("MAE 24h (km)")
    ax.set_title(f"Model vs CLIPER cho {len(sids)} bão (sorted by Model MAE)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, rotation=90)

    fig.tight_layout()
    _savefig(fig, fig_dir, "per_storm_model_vs_cliper.png")


def plot_cdf(per_storm: list, fig_dir: Path):
    """CDF — fraction of storms with MAE ≤ x."""
    maes = np.sort(np.array([s["mae_24h"] for s in per_storm]))
    cdf  = np.arange(1, len(maes) + 1) / len(maes)

    cliper_maes = np.sort(np.array([s["cliper_mae_24h"] for s in per_storm]))
    cdf_cliper  = np.arange(1, len(cliper_maes) + 1) / len(cliper_maes)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(maes, cdf * 100, drawstyle="steps-post",
            linewidth=2, color="#3b82f6", label="Model")
    ax.plot(cliper_maes, cdf_cliper * 100, drawstyle="steps-post",
            linewidth=2, color="#6b7280", linestyle="--", label="CLIPER")
    ax.axhline(50, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(150, color="purple", linestyle=":", alpha=0.5, label="Thesis target 150 km")

    ax.set_xlabel("MAE 24h (km)")
    ax.set_ylabel("Cumulative % of storms")
    ax.set_title("CDF — Phần trăm bão có MAE ≤ x km")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "per_storm_cdf.png")


def plot_yearly(per_storm: list, fig_dir: Path):
    """Box plot MAE per year — xem model generalize sang năm mới thế nào."""
    by_year = {}
    for s in per_storm:
        y = s["season"]
        by_year.setdefault(y, []).append(s["mae_24h"])

    years  = sorted(by_year.keys())
    data   = [by_year[y] for y in years]
    counts = [len(d) for d in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(data, labels=[f"{y}\n(n={c})" for y, c in zip(years, counts)],
                    patch_artist=True)
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6"]
    for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("MAE 24h (km)")
    ax.set_title("MAE 24h theo năm — verify generalization sang năm mới")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _savefig(fig, fig_dir, "per_storm_yearly.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="scs_v12_lb6",
                        help="Tag (mặc định scs_v12_lb6 — champion)")
    parser.add_argument("--scs-only", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    suffix  = f"_{args.tag}" if args.tag else ""
    sub_dir = (args.tag if args.tag else "legacy") + ("_scs" if args.scs_only else "")
    fig_dir = BASE_DIR / "results" / "figures" / sub_dir
    json_path = fig_dir / "per_storm_metrics.json"

    if not json_path.exists():
        print(f"[error] Không tìm thấy {json_path}")
        print("  Chạy: python -m src.g7_evaluate.evaluate --tag {tag}")
        return

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    per_storm  = data["per_storm"]
    best_model = data["best_model"]
    tag        = data["tag"]
    n_storms   = data["n_storms"]
    mean_mae   = float(np.mean([s["mae_24h"] for s in per_storm]))
    mean_skill = float(np.mean([s["skill_24h"] for s in per_storm]))

    print("=" * 60)
    print(f"  Per-Storm Visualization — {tag}")
    print(f"  Best model: {best_model.upper()} | {n_storms} storms")
    print(f"  Mean MAE 24h = {mean_mae:.1f} km | Mean Skill 24h = {mean_skill:.1f}%")
    print("=" * 60)

    print(f"\n[1] MAE histogram...")
    plot_mae_histogram(per_storm, mean_mae, fig_dir)
    print(f"[2] Skill histogram...")
    plot_skill_histogram(per_storm, fig_dir)
    print(f"[3] Skill vs N scatter...")
    plot_skill_vs_n(per_storm, fig_dir)
    print(f"[4] Model vs CLIPER bar chart...")
    plot_model_vs_cliper_topN(per_storm, fig_dir, top_n=14)
    print(f"[5] CDF plot...")
    plot_cdf(per_storm, fig_dir)
    print(f"[6] Yearly box plot...")
    plot_yearly(per_storm, fig_dir)

    print(f"\n  ✓ 6 figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
