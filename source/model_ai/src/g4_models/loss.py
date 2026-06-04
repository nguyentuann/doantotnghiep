"""
loss.py — HaversineLoss + MultiHorizonLoss
------------------------------------------
HaversineLoss (legacy):
  Loss = w_24h × Hav(pred_24h, true_24h) + w_48h × Hav(pred_48h, true_48h)
  Input: [B, 4] = [lat_24h, lon_24h, lat_48h, lon_48h]

MultiHorizonLoss (Sprint 1):
  Input: [B, n_steps*2] — e.g. [B, 16] for 8 steps
  Loss = Σ step_weight[t] × Hav(pred_t, true_t)
       + λ_smooth × ||Δ²(pred_track)||²   ← 2nd derivative penalty
       + λ_dir    × mean(1 - cos(Δbearing)) ← directional consistency
"""

import torch
import torch.nn as nn


def _haversine_km(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """pred, true: [B, 2] = [lat_deg, lon_deg] → [B] km"""
    R = 6371.0
    lat1 = torch.deg2rad(pred[:, 0])
    lon1 = torch.deg2rad(pred[:, 1])
    lat2 = torch.deg2rad(true[:, 0])
    lon2 = torch.deg2rad(true[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (torch.sin(dlat / 2) ** 2
         + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2)
    c = 2 * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))
    return R * c


class HaversineLoss(nn.Module):
    """Legacy loss cho 2-anchor output [B, 4]."""

    def __init__(self, weight_24h: float = 0.6, weight_48h: float = 0.4):
        super().__init__()
        self.w24 = weight_24h
        self.w48 = weight_48h

    def _haversine_km(self, pred, true):
        return _haversine_km(pred, true)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        dist_24h = _haversine_km(pred[:, :2], true[:, :2])
        dist_48h = _haversine_km(pred[:, 2:], true[:, 2:])
        return self.w24 * dist_24h.mean() + self.w48 * dist_48h.mean()

    def mae_km(self, pred: torch.Tensor, true: torch.Tensor):
        """Trả về (mae_24h_km, mae_48h_km) — dùng cho logging."""
        with torch.no_grad():
            mae_24 = _haversine_km(pred[:, :2], true[:, :2]).mean().item()
            mae_48 = _haversine_km(pred[:, 2:], true[:, 2:]).mean().item()
        return mae_24, mae_48


class MultiHorizonLoss(nn.Module):
    """
    Multi-horizon loss cho Sprint 1.

    pred, true : [B, n_steps*2] — flat array [lat_s1, lon_s1, lat_s2, lon_s2, ...]
    n_steps    : số bước dự đoán (8 cho Sprint 1)
    idx_24h    : index 0-based của step 24h trong anchor_steps (mặc định 3 = step 4)
    idx_48h    : index 0-based của step 48h trong anchor_steps (mặc định 7 = step 8)

    Loss = primary_haversine + λ_smooth * smooth + λ_dir * directional
    """

    def __init__(self, anchor_steps: list, weight_24h: float = 0.50,
                 weight_48h: float = 0.20, lambda_smooth: float = 1.0,
                 lambda_dir: float = 0.1):
        super().__init__()
        self.anchor_steps  = anchor_steps
        self.n_steps       = len(anchor_steps)
        self.lambda_smooth = lambda_smooth
        self.lambda_dir    = lambda_dir

        # Step weights: extra weight cho 24h và 48h, còn lại chia đều
        n_other = self.n_steps - 2
        other_w = max(0.0, 1.0 - weight_24h - weight_48h) / max(n_other, 1)
        weights = []
        for step in anchor_steps:
            if step == 4:
                weights.append(weight_24h)
            elif step == 8:
                weights.append(weight_48h)
            else:
                weights.append(other_w)
        self.register_buffer("step_weights", torch.tensor(weights, dtype=torch.float32))

        # Chỉ số 24h và 48h trong mảng flat (mỗi step chiếm 2 vị trí)
        self._idx_24h = anchor_steps.index(4) if 4 in anchor_steps else None
        self._idx_48h = anchor_steps.index(8) if 8 in anchor_steps else None

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        B = pred.shape[0]
        pred_t = pred.view(B, self.n_steps, 2)  # [B, n_steps, 2]
        true_t = true.view(B, self.n_steps, 2)

        # --- Primary: weighted Haversine per step ---
        primary = torch.zeros(1, device=pred.device)
        for t in range(self.n_steps):
            d = _haversine_km(pred_t[:, t, :], true_t[:, t, :]).mean()
            primary = primary + self.step_weights[t] * d

        # --- Smoothness: L2 of 2nd derivative in degrees space ---
        smooth = torch.zeros(1, device=pred.device)
        if self.n_steps >= 3:
            delta2 = pred_t[:, 2:, :] - 2.0 * pred_t[:, 1:-1, :] + pred_t[:, :-2, :]
            smooth = delta2.pow(2).mean()

        # --- Directional: 1 - cos(angle between consecutive directions) ---
        dir_loss = torch.zeros(1, device=pred.device)
        if self.n_steps >= 2:
            pred_dirs = pred_t[:, 1:, :] - pred_t[:, :-1, :]   # [B, n-1, 2]
            true_dirs = true_t[:, 1:, :] - true_t[:, :-1, :]
            pred_norm = pred_dirs / (pred_dirs.norm(dim=-1, keepdim=True) + 1e-8)
            true_norm = true_dirs / (true_dirs.norm(dim=-1, keepdim=True) + 1e-8)
            cos_sim   = (pred_norm * true_norm).sum(dim=-1)      # [B, n-1]
            dir_loss  = (1.0 - cos_sim).mean()

        return primary + self.lambda_smooth * smooth + self.lambda_dir * dir_loss

    def mae_km(self, pred: torch.Tensor, true: torch.Tensor):
        """Trả về (mae_24h_km, mae_48h_km) — dùng cho logging."""
        with torch.no_grad():
            B = pred.shape[0]
            pred_t = pred.view(B, self.n_steps, 2)
            true_t = true.view(B, self.n_steps, 2)

            if self._idx_24h is not None:
                mae_24 = _haversine_km(pred_t[:, self._idx_24h, :],
                                       true_t[:, self._idx_24h, :]).mean().item()
            else:
                mae_24 = float("nan")

            if self._idx_48h is not None:
                mae_48 = _haversine_km(pred_t[:, self._idx_48h, :],
                                       true_t[:, self._idx_48h, :]).mean().item()
            else:
                mae_48 = float("nan")

        return mae_24, mae_48
