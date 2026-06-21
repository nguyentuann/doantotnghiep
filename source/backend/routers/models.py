import json
from pathlib import Path
from typing import Dict, List
from fastapi import APIRouter
from schemas.response import ModelInfo

router = APIRouter(prefix="/models", tags=["models"])

_LOG_PATH = Path(__file__).parent.parent.parent / "model_ai/results/results_log.json"


@router.get("", response_model=List[ModelInfo])
def list_models():
    if not _LOG_PATH.exists():
        return []

    with open(_LOG_PATH, "r", encoding="utf-8") as f:
        logs = json.load(f)

    # Lấy entry G7_evaluate hoặc G5/G6 train
    candidates = [
        e for e in logs
        if e.get("model_name", "").lower() != "cliper" and "mae_24h" in e
    ]

    # Dedup: giữ entry MAE thấp nhất cho mỗi model
    best: Dict[str, dict] = {}
    for e in candidates:
        name = e["model_name"].lower()
        if name not in best or e["mae_24h"] < best[name]["mae_24h"]:
            best[name] = e

    if not best:
        return []

    best_model = min(best.values(), key=lambda e: e["mae_24h"])["model_name"].lower()

    return [
        ModelInfo(
            name=name,
            mae_24h=e.get("mae_24h"),
            mae_48h=e.get("mae_48h"),
            skill_score_24h=e.get("skill_score_24h"),
            is_best=(name == best_model),
        )
        for name, e in best.items()
    ]
