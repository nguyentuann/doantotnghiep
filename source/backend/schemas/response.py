from pydantic import BaseModel, Field
from typing import List, Optional


class ForecastPoint(BaseModel):
    lat: float
    lon: float
    hour: int = Field(..., description="Giờ dự báo: 24 hoặc 48")
    mae_km: Optional[float] = Field(None, description="MAE tham chiếu (km)")


class ForecastResponse(BaseModel):
    origin_lat: float
    origin_lon: float
    points: List[ForecastPoint]
    model_used: str


class PredictResponse(BaseModel):
    forecast: ForecastResponse


class StormPoint(BaseModel):
    iso_time: str
    lat: float
    lon: float
    vmax: Optional[float] = None
    pmin: Optional[float] = None


class StormListItem(BaseModel):
    sid: str
    name: str
    season: int
    basin: str


class StormDetailResponse(BaseModel):
    sid: str
    name: str
    season: int
    basin: str
    track: List[StormPoint]
    cutoff_index: int = 8
    predicted_track: List[StormPoint] = []   # rolling prediction từ cutoff trở đi


class ModelInfo(BaseModel):
    name: str
    mae_24h: Optional[float] = None
    mae_48h: Optional[float] = None
    skill_score_24h: Optional[float] = None
    is_best: bool = False
