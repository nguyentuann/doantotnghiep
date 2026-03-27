"""
loss.py — HaversineLoss
------------------------
Loss = w_24h × Haversine(pred_24h, true_24h)
     + w_48h × Haversine(pred_48h, true_48h)

Dùng Haversine thay vì MSE vì 1° kinh độ ≠ 1° vĩ độ về km.
Ở vĩ độ 20°N: 1° kinh ≈ 103 km, 1° vĩ ≈ 111 km.
"""

import torch
import torch.nn as nn


class HaversineLoss(nn.Module):
    """
    Input  pred, true : [B, 4] = [lat_24h, lon_24h, lat_48h, lon_48h]
    Output             : scalar (km)
    """

    R_KM = 6371.0

    def __init__(self, weight_24h: float = 0.6, weight_48h: float = 0.4):
        super().__init__()
        self.w24 = weight_24h
        self.w48 = weight_48h

    def _haversine_km(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """pred, true: [B, 2] = [lat_deg, lon_deg] → [B] km"""
        lat1 = torch.deg2rad(pred[:, 0])
        lon1 = torch.deg2rad(pred[:, 1])
        lat2 = torch.deg2rad(true[:, 0])
        lon2 = torch.deg2rad(true[:, 1])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (torch.sin(dlat / 2) ** 2
             + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
        c = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))
        return self.R_KM * c

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        dist_24h = self._haversine_km(pred[:, :2], true[:, :2])
        dist_48h = self._haversine_km(pred[:, 2:], true[:, 2:])
        return self.w24 * dist_24h.mean() + self.w48 * dist_48h.mean()

    def mae_km(self, pred: torch.Tensor, true: torch.Tensor):
        """Trả về (mae_24h_km, mae_48h_km) — dùng cho logging, không backprop."""
        with torch.no_grad():
            mae_24 = self._haversine_km(pred[:, :2], true[:, :2]).mean().item()
            mae_48 = self._haversine_km(pred[:, 2:], true[:, 2:]).mean().item()
        return mae_24, mae_48
