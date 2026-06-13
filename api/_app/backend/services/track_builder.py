"""
track_builder.py — Nội suy track 6h steps + tính uncertainty cone
"""

import math
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def build_forecast(origin_lat: float, origin_lon: float,
                   lat_24h: float, lon_24h: float,
                   lat_48h: float, lon_48h: float,
                   model_name: str,
                   mae_24h: float = None,
                   mae_48h: float = None) -> dict:
    """
    Tạo forecast response với 2 điểm dự báo: +24h và +48h.
    """
    points = [
        {
            "lat":    round(lat_24h, 4),
            "lon":    round(lon_24h, 4),
            "hour":   24,
            "mae_km": round(mae_24h, 1) if mae_24h else None,
        },
        {
            "lat":    round(lat_48h, 4),
            "lon":    round(lon_48h, 4),
            "hour":   48,
            "mae_km": round(mae_48h, 1) if mae_48h else None,
        },
    ]

    return {
        "origin_lat": round(origin_lat, 4),
        "origin_lon": round(origin_lon, 4),
        "points":     points,
        "model_used": model_name,
    }
