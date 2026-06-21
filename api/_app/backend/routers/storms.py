import json
from pathlib import Path
from fastapi import APIRouter, HTTPException, Query
from schemas.response import StormListItem, StormDetailResponse
from services.predictor import rolling_predict, is_ready, available_archs, DEFAULT_ARCH

router = APIRouter(prefix="/storms", tags=["storms"])

_DATA_PATH = Path(__file__).parent.parent / "data/historical_tracks.json"
_cache: list[dict] | None = None


def _load():
    global _cache
    if _cache is not None:
        return _cache
    if not _DATA_PATH.exists():
        _cache = []
        return _cache
    with open(_DATA_PATH, "r", encoding="utf-8") as f:
        _cache = json.load(f)
    return _cache


@router.get("", response_model=list[StormListItem])
def list_storms():
    storms = _load()
    return [
        {"sid": s["sid"], "name": s["name"], "season": s["season"], "basin": s.get("basin", "WP")}
        for s in storms
    ]


@router.get("/archs")
def list_archs():
    return {"archs": available_archs(), "default": DEFAULT_ARCH}


@router.get("/{sid}", response_model=StormDetailResponse)
def get_storm(sid: str, model: str | None = Query(None)):
    storms = _load()
    storm = next((s for s in storms if s["sid"] == sid), None)
    if storm is None:
        raise HTTPException(status_code=404, detail=f"Không tìm thấy storm: {sid}")

    arch = model if model in ("transformer", "lstm", "bilstm", "bigru") else DEFAULT_ARCH
    cutoff = storm.get("cutoff_index", 8)
    predicted_track = []

    if is_ready(arch) and len(storm["track"]) > cutoff:
        try:
            predicted_track = rolling_predict(storm["track"], cutoff, arch=arch)
        except Exception:
            predicted_track = []

    return {**storm, "predicted_track": predicted_track}
