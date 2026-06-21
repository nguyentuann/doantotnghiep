from pydantic import BaseModel, Field
from typing import List, Optional


class TrackPoint(BaseModel):
    lat: float = Field(..., description="Vĩ độ (°N)")
    lon: float = Field(..., description="Kinh độ (°E)")
    vmax: Optional[float] = Field(None, description="Tốc độ gió cực đại (kt)")
    pmin: Optional[float] = Field(None, description="Áp suất cực tiểu (hPa)")
    iso_time: Optional[str] = Field(None, description="Thời điểm ISO 8601, vd: 2023-09-05T06:00:00")


class PredictRequest(BaseModel):
    track: List[TrackPoint] = Field(
        ...,
        min_length=8,
        description="Tối thiểu 8 điểm (48h lịch sử, mỗi bước 6h)"
    )
    model_name: Optional[str] = Field(
        None,
        description="Tên model: lstm | bilstm | transformer. Mặc định: best model"
    )
