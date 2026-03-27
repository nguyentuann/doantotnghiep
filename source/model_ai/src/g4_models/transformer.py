"""
transformer.py — Temporal Transformer (Model 3)
-------------------------------------------------
Self-Attention song song trên tất cả 8 bước — khác LSTM là tuần tự.
Learnable Positional Encoding thay vì sin/cos cố định (tốt hơn cho chuỗi ngắn 8 bước).
Pre-LayerNorm (norm_first=True) ổn định gradient hơn Post-LN với batch nhỏ.

  Input [B, 8, 12]
    → Linear(12→64)                           # chiếu vào không gian d_model
    → + Learnable PE(1, 8, 64)               # thêm thông tin vị trí
    → TransformerEncoder(d=64, heads=4, ff=256, layers=2, norm_first=True)
    → lấy token cuối [B, 64]                 # bước 8 = thời điểm hiện tại
    → Linear(64→32) → ReLU → Linear(32→4)
  Output [B, 4]

Tham số: 103,524
Checkpoint G6: cải thiện ≥ 10% Val MAE so với LSTM
"""

import torch
import torch.nn as nn


class TemporalTransformerModel(nn.Module):

    def __init__(self, n_features: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int,
                 dropout: float, lookback: int, output_size: int = 4):
        """
        Args:
            n_features        : số features đầu vào (12)
            d_model           : chiều không gian embedding (64)
            nhead             : số attention heads — d_model phải chia hết cho nhead (4)
            num_encoder_layers: số lớp TransformerEncoder (2)
            dim_feedforward   : chiều FFN bên trong encoder = 4×d_model (256)
            dropout           : xác suất dropout (0.2 — nhỏ hơn LSTM vì model đã nhỏ)
            lookback          : độ dài chuỗi đầu vào (8) — cần cho PE
            output_size       : số đầu ra (4)
        """
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) phải chia hết cho nhead ({nhead})"
        )

        self.input_proj = nn.Linear(n_features, d_model)

        # Learnable PE: mỗi trong 8 bước thời gian có vector vị trí riêng
        # Khởi tạo nhỏ (×0.01) để không lấn át input projection ban đầu
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,        # Pre-LN: normalize trước Attention/FFN
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,  # tắt warning khi norm_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, lookback, n_features] → [B, 4]"""
        x = self.input_proj(x) + self.pos_embed   # [B, lookback, d_model]
        x = self.encoder(x)                        # [B, lookback, d_model]
        last = x[:, -1, :]                         # token cuối = bước hiện tại
        return self.head(last)                     # [B, 4]
