"""
factory.py — Hàm khởi tạo model và loss từ config.yaml
---------------------------------------------------------
Import từ đây khi cần dùng trong train.py hoặc evaluate.py.
"""

import yaml
import torch.nn as nn
from pathlib import Path

from .loss import HaversineLoss
from .lstm import LSTMModel
from .bilstm import BiLSTMAttentionModel
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
    n_features  = cfg["features"]["n_features"]  # 12
    hidden_size = m["hidden_size"]               # 128
    num_layers  = m["num_layers"]               # 2
    dropout     = m["dropout"]                  # 0.3
    lookback    = m["lookback"]                 # 8

    name = model_name.lower().strip()

    if name == "lstm":
        return LSTMModel(n_features, hidden_size, num_layers, dropout)

    elif name in ("bilstm", "bilstm_attention"):
        return BiLSTMAttentionModel(n_features, hidden_size, num_layers, dropout)

    elif name in ("transformer", "temporal_transformer"):
        return TemporalTransformerModel(
            n_features=n_features,
            d_model=64,
            nhead=4,
            num_encoder_layers=num_layers,
            dim_feedforward=256,
            dropout=0.2,
            lookback=lookback,
        )

    else:
        raise ValueError(
            f"model_name không hợp lệ: '{name}'. Chọn: lstm | bilstm | transformer"
        )


def build_loss(cfg: dict = None) -> HaversineLoss:
    """Khởi tạo HaversineLoss với trọng số từ config.yaml."""
    if cfg is None:
        cfg = load_config()
    return HaversineLoss(
        weight_24h=cfg["loss"]["weight_24h"],
        weight_48h=cfg["loss"]["weight_48h"],
    )


def count_parameters(model: nn.Module) -> int:
    """Đếm số tham số có thể train."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
