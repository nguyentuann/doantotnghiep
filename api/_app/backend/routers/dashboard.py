"""
dashboard.py — Endpoints cho Model Dashboard (trực quan hóa thông số model)
---------------------------------------------------------------------------
  GET /dashboard/models                 — danh sách tag SCS + 4 kiến trúc mỗi tag
  GET /dashboard/per-storm/{tag}         — MAE per-storm (34 bão)
  GET /dashboard/feature-importance/{tag} — permutation importance
  GET /dashboard/ablation                — tiến trình cải tiến v7 → v12

Đọc trực tiếp từ results_log.json + per_storm_metrics.json + feature_importance.json.
"""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

from _artifacts import model_ai_dir
_BASE = model_ai_dir()
_LOG = _BASE / "results/results_log.json"
_FIG = _BASE / "results/figures"

# ─── Metadata cho từng tag SCS (kèm thứ tự hiển thị, champion) ────────────
# results_log.json không lưu n_features/lookback → khai báo ở đây
TAG_META = {
    "scs_v12_lb6":   {"display": "scs_v12_lb6 (Champion)", "n_features": 31, "lookback": 6,
                      "desc": "Champion — thêm annulus 850 hPa low-level steering", "champion": True, "order": 1},
    "scs_v11_lb6":   {"display": "scs_v11_lb6", "n_features": 29, "lookback": 6,
                      "desc": "+ DLM annulus steering (500+700 hPa) — best 48h", "champion": False, "order": 2},
    "scs_v12_lean_lb6": {"display": "scs_v12_lean_lb6 (Lean)", "n_features": 24, "lookback": 6,
                      "desc": "Bản gọn — bỏ 7 đặc trưng importance thấp", "champion": False, "order": 3},
    "scs_v9_lb6":    {"display": "scs_v9_lb6", "n_features": 27, "lookback": 6,
                      "desc": "+ lookback 8→6 (sweet spot)", "champion": False, "order": 4},
    "scs_v9":        {"display": "scs_v9", "n_features": 27, "lookback": 8,
                      "desc": "+ beta drift + SWA + multi-seed", "champion": False, "order": 5},
    "scs_v7":        {"display": "scs_v7", "n_features": 25, "lookback": 8,
                      "desc": "SCS-only training baseline", "champion": False, "order": 6},
    "scs_lite15_lb6": {"display": "scs_lite15_lb6 (Lite)", "n_features": 15, "lookback": 6,
                      "desc": "Bản nhẹ — chỉ storm-state, không cần ERA5", "champion": False, "order": 7},
}

# Số tham số mỗi kiến trúc (xấp xỉ)
ARCH_PARAMS = {"lstm": 146_000, "bilstm": 280_000, "bigru": 220_000, "transformer": 110_000}
ARCH_ORDER = ["lstm", "bilstm", "bigru", "transformer"]
ARCH_DISPLAY = {"lstm": "LSTM", "bilstm": "BiLSTM+Attn",
                "bigru": "BiGRU+Attn", "transformer": "Transformer"}


def _load_log():
    if not _LOG.exists():
        return []
    with open(_LOG, "r", encoding="utf-8") as f:
        return json.load(f)


def _best_entry(logs, model_name):
    """Lấy entry MAE 24h thấp nhất cho model_name."""
    cands = [e for e in logs if e.get("model_name") == model_name and "mae_24h" in e]
    if not cands:
        return None
    return min(cands, key=lambda e: e["mae_24h"])


@router.get("/models")
def list_dashboard_models():
    """Danh sách tag SCS, mỗi tag kèm 4 kiến trúc (nếu có) + CLIPER baseline."""
    logs = _load_log()
    result = []

    # CLIPER baseline — tính từ per_storm của champion (đúng test set SCS-only),
    # KHÔNG dùng global min (sẽ nhầm sang CLIPER của tag 14feat test set khác)
    cliper_24h, cliper_48h = 174.6, 431.5  # fallback giá trị champion đã biết
    champ_ps = _FIG / "scs_v12_lb6_scs/per_storm_metrics.json"
    if champ_ps.exists():
        try:
            with open(champ_ps, "r", encoding="utf-8") as f:
                ps = json.load(f)["per_storm"]
            tot_n = sum(s["n_sequences"] for s in ps)
            if tot_n > 0:
                cliper_24h = round(sum(s["cliper_mae_24h"] * s["n_sequences"] for s in ps) / tot_n, 1)
                cliper_48h = round(sum(s["cliper_mae_48h"] * s["n_sequences"] for s in ps) / tot_n, 1)
        except Exception:
            pass

    for tag, meta in sorted(TAG_META.items(), key=lambda kv: kv[1]["order"]):
        archs = []
        for arch in ARCH_ORDER:
            e = _best_entry(logs, f"{arch}_{tag}")
            if e is None:
                continue
            archs.append({
                "arch": arch,
                "display": ARCH_DISPLAY[arch],
                "params": ARCH_PARAMS[arch],
                "mae_24h": e.get("mae_24h"),
                "mae_48h": e.get("mae_48h"),
                "rmse_24h": e.get("rmse_24h"),
                "skill_24h": e.get("skill_score_24h"),
                "skill_48h": e.get("skill_score_48h"),
            })
        if not archs:
            continue
        # Best arch theo skill_24h
        best_arch = max(archs, key=lambda a: a.get("skill_24h") or -999)
        result.append({
            "tag": tag,
            "display": meta["display"],
            "n_features": meta["n_features"],
            "lookback": meta["lookback"],
            "description": meta["desc"],
            "is_champion": meta["champion"],
            "architectures": archs,
            "best_arch": best_arch["arch"],
            "best_skill_24h": best_arch.get("skill_24h"),
            "best_mae_24h": best_arch.get("mae_24h"),
            "has_per_storm": (_FIG / f"{tag}_scs/per_storm_metrics.json").exists(),
            "has_feature_importance": (_FIG / f"{tag}_analysis/feature_importance.json").exists(),
        })

    return {
        "cliper": {"mae_24h": cliper_24h, "mae_48h": cliper_48h},
        "models": result,
    }


@router.get("/per-storm/{tag}")
def per_storm(tag: str):
    """MAE per-storm cho tag (từ per_storm_metrics.json)."""
    path = _FIG / f"{tag}_scs/per_storm_metrics.json"
    if not path.exists():
        raise HTTPException(404, f"Không có per-storm data cho tag '{tag}'")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@router.get("/feature-importance/{tag}")
def feature_importance(tag: str):
    """Permutation feature importance cho tag (nếu có)."""
    path = _FIG / f"{tag}_analysis/feature_importance.json"
    if not path.exists():
        raise HTTPException(404, f"Không có feature-importance cho tag '{tag}'")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Lọc bỏ lat_norm/lon_norm (CLIPER artifact) khỏi top per_feature để hiển thị đúng
    if "per_feature" in data:
        data["per_feature_clean"] = [
            f for f in data["per_feature"]
            if f["feature"] not in ("lat_norm", "lon_norm")
        ]
    return data


@router.get("/ablation")
def ablation():
    """Tiến trình cải tiến qua các phiên bản (Transformer 3-seed)."""
    logs = _load_log()
    progression = [
        ("scs_v7", "SCS baseline (25 feat)"),
        ("scs_v9", "+ beta drift + SWA"),
        ("scs_v9_lb6", "+ lookback 6"),
        ("scs_v11_lb6", "+ DLM annulus"),
        ("scs_v12_lb6", "+ 850 hPa annulus (champion)"),
    ]
    steps = []
    for tag, label in progression:
        e = _best_entry(logs, f"transformer_{tag}")
        if e:
            steps.append({
                "tag": tag,
                "label": label,
                "skill_24h": e.get("skill_score_24h"),
                "skill_48h": e.get("skill_score_48h"),
                "mae_24h": e.get("mae_24h"),
            })
    return {"progression": steps}
