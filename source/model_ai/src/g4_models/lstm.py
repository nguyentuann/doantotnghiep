"""
lstm.py — LSTM Baseline (Model 1)
-----------------------------------
Kiến trúc đơn giản nhất, dùng làm reference point cho BiLSTM và Transformer.

  Input [B, 8, 12]
    → LSTM(hidden=128, layers=2, dropout=0.3)
    → lấy hidden state bước cuối [B, 128]
    → LayerNorm(128)
    → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→4)
  Output [B, 4]

Tham số: 213,572
Checkpoint G5: Val MAE 24h < 200 km
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):

    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float, output_size: int = 4):
        """
        Args:
            n_features  : số features đầu vào (12)
            hidden_size : kích thước hidden state (128)
            num_layers  : số lớp LSTM xếp chồng (2)
            dropout     : xác suất dropout — tắt giữa các LSTM layer (0.3)
            output_size : số đầu ra (4 = lat_24h, lon_24h, lat_48h, lon_48h)
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, lookback, n_features]
        Hidden state bước cuối đã "đọc" toàn bộ chuỗi qua forget/input gate.
        """
        out, _ = self.lstm(x)    # [B, lookback, hidden]
        last = out[:, -1, :]     # [B, hidden] — bước cuối
        last = self.norm(last)
        return self.head(last)   # [B, 4]
