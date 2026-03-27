"""
bilstm.py — BiLSTM + Self-Attention (Model 2)
-----------------------------------------------
Cải tiến so với LSTM theo 2 hướng:
  1. Bidirectional: đọc cả xuôi lẫn ngược → tại bước t biết cả t-1 lẫn t+1
  2. Attention: tự động học trọng số quan trọng của từng bước thời gian
     (vd: bước bão đổi hướng đột ngột quan trọng hơn bước đi thẳng)

  Input [B, 8, 12]
    → BiLSTM(hidden=128, bidirectional=True, layers=2) → [B, 8, 256]
    → Attention: Linear(256→1) → Softmax → trọng số [B, 8]
    → Weighted sum → context vector [B, 256]
    → LayerNorm(256)
    → Linear(256→128) → GELU → Dropout(0.3) → Linear(128→4)
  Output [B, 4]

Tham số: 574,852
Checkpoint G6: cải thiện ≥ 10% Val MAE so với LSTM
"""

import torch
import torch.nn as nn


class BiLSTMAttentionModel(nn.Module):

    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float, output_size: int = 4):
        """
        Args:
            n_features  : số features đầu vào (12)
            hidden_size : hidden size mỗi chiều — output sẽ là 2×hidden (128 → 256)
            num_layers  : số lớp BiLSTM (2)
            dropout     : xác suất dropout (0.3)
            output_size : số đầu ra (4)
        """
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        bilstm_out = hidden_size * 2   # bidirectional → 256

        # Additive attention (Bahdanau-style)
        # Tính 1 scalar score cho mỗi bước thời gian → softmax → trọng số
        self.attn = nn.Linear(bilstm_out, 1, bias=False)

        self.norm = nn.LayerNorm(bilstm_out)
        self.head = nn.Sequential(
            nn.Linear(bilstm_out, hidden_size),
            nn.GELU(),                  # mềm hơn ReLU, phù hợp context vector
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, lookback, n_features] → [B, 4]"""
        out, _ = self.bilstm(x)                          # [B, lookback, 256]
        scores  = self.attn(out).squeeze(-1)             # [B, lookback]
        weights = torch.softmax(scores, dim=1)           # [B, lookback] — sum=1
        context = (weights.unsqueeze(-1) * out).sum(1)  # [B, 256]
        context = self.norm(context)
        return self.head(context)                        # [B, 4]
