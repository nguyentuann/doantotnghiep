"""
factory.py — Hàm khởi tạo model và loss từ config.yaml
---------------------------------------------------------
Import từ đây khi cần dùng trong train.py hoặc evaluate.py.
"""

import yaml
import torch.nn as nn
from pathlib import Path

from .loss import HaversineLoss, MultiHorizonLoss
from .lstm import LSTMModel
from .bilstm import BiLSTMAttentionModel
from .bigru import BiGRUAttentionModel
from .transformer import TemporalTransformerModel


def load_config() -> dict:
    cfg_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_name: str, cfg: dict = None) -> nn.Module:
    """
    Khởi tạo model theo tên, đọc hyperparameter từ config.yaml.

    Args:
        model_name : 'lstm' | 'bilstm' | 'transformer'
        cfg        : dict config (None → tự load)

    Returns:
        nn.Module chưa train
    """
    if cfg is None:
        cfg = load_config()

    m           = cfg["model"]
    n_features  = cfg["features"]["n_features"]  # 19 (sprint1) / 14 / 12
    hidden_size = m["hidden_size"]               # 128
    num_layers  = m["num_layers"]               # 2
    dropout     = m["dropout"]                  # 0.3
    lookback    = m["lookback"]                 # 8
    output_size = m.get("output_size", 4)       # 16 (sprint1) / 4 (legacy)

    name = model_name.lower().strip()

    if name == "lstm":
        return LSTMModel(n_features, hidden_size, num_layers, dropout, output_size)

    elif name in ("bilstm", "bilstm_attention"):
        return BiLSTMAttentionModel(n_features, hidden_size, num_layers, dropout, output_size)

    elif name in ("bigru", "bigru_attention"):
        return BiGRUAttentionModel(n_features, hidden_size, num_layers, dropout, output_size)

    elif name in ("transformer", "temporal_transformer"):
        nhead = m.get("nhead", 4)
        return TemporalTransformerModel(
            n_features=n_features,
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            lookback=lookback,
            output_size=output_size,
        )

    else:
        raise ValueError(
            f"model_name không hợp lệ: '{name}'. Chọn: lstm | bilstm | bigru | transformer"
        )


def build_loss(cfg: dict = None):
    """
    Khởi tạo loss function từ config.yaml.
    - Nếu config có anchor_steps với nhiều hơn 2 bước → MultiHorizonLoss (Sprint 1)
    - Ngược lại → HaversineLoss (legacy)
    """
    if cfg is None:
        cfg = load_config()
    anchor_steps = cfg["model"].get("anchor_steps", [4, 8])
    if len(anchor_steps) > 2:
        return MultiHorizonLoss(
            anchor_steps=anchor_steps,
            weight_24h=cfg["loss"]["weight_24h"],
            weight_48h=cfg["loss"]["weight_48h"],
            lambda_smooth=cfg["loss"].get("lambda_smooth", 1.0),
            lambda_dir=cfg["loss"].get("lambda_dir", 0.1),
        )
    return HaversineLoss(
        weight_24h=cfg["loss"]["weight_24h"],
        weight_48h=cfg["loss"]["weight_48h"],
    )


def count_parameters(model: nn.Module) -> int:
    """Đếm số tham số có thể train."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
