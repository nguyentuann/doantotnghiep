"""
bigru.py — BiGRU + Self-Attention (Model 4)
--------------------------------------------
Kiến trúc theo Song et al. (2022) JAMC — BiGRU với Attention mechanism
đã được chứng minh vượt BiLSTM trên WNP track prediction.

GRU ít tham số hơn LSTM (3 gates thay vì 4), hội tụ nhanh hơn.

  Input [B, lookback, n_features]
    → BiGRU(hidden, bidirectional=True, layers) → [B, lookback, hidden*2]
    → Attention: Linear(hidden*2 → 1) → Softmax → [B, lookback, 1]
    → Weighted sum → context [B, hidden*2]
    → LayerNorm
    → Linear(hidden*2 → hidden) → GELU → Dropout → Linear(hidden → output_size)
  Output [B, output_size]
"""

import torch
import torch.nn as nn

from .lstm import _init_head_bias_scs


class BiGRUAttentionModel(nn.Module):

    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float, output_size: int = 16):
        super().__init__()
        self.bigru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        bigru_out = hidden_size * 2  # bidirectional

        self.attn = nn.Linear(bigru_out, 1, bias=False)
        self.norm = nn.LayerNorm(bigru_out)
        self.head = nn.Sequential(
            nn.Linear(bigru_out, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        _init_head_bias_scs(self.head[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, lookback, n_features] → [B, output_size]"""
        out, _ = self.bigru(x)                           # [B, lookback, hidden*2]
        scores  = self.attn(out).squeeze(-1)             # [B, lookback]
        weights = torch.softmax(scores, dim=1)           # [B, lookback]
        context = (weights.unsqueeze(-1) * out).sum(1)  # [B, hidden*2]
        context = self.norm(context)
        return self.head(context)                        # [B, output_size]
