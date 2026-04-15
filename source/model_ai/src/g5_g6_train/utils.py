"""
utils.py — Công cụ hỗ trợ training
-------------------------------------
  EarlyStopping  — dừng sớm khi val loss không cải thiện
  save_checkpoint / load_checkpoint — lưu/đọc model
  log_result     — ghi kết quả vào results_log.json
"""

import json
import torch
from datetime import datetime
from pathlib import Path


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """
    Dừng training khi val_loss không cải thiện sau `patience` epoch liên tiếp.
    Lưu best model vào `checkpoint_path`.

    Args:
        patience        : số epoch chờ trước khi dừng (từ config: 20)
        min_delta       : cải thiện tối thiểu để tính là "tốt hơn" (km)
        checkpoint_path : đường dẫn lưu best model
    """

    def __init__(self, patience: int, checkpoint_path: str, min_delta: float = 0.1):
        self.patience   = patience
        self.min_delta  = min_delta
        self.path       = Path(checkpoint_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.best_loss  = float("inf")
        self.counter    = 0
        self.best_epoch = 0

    def step(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Gọi sau mỗi epoch.
        Returns True nếu nên dừng training.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.counter    = 0
            torch.save(model.state_dict(), self.path)
            return False   # tiếp tục train
        else:
            self.counter += 1
            return self.counter >= self.patience   # True → dừng

    def load_best(self, model: torch.nn.Module) -> torch.nn.Module:
        """Đọc lại weight của epoch tốt nhất."""
        model.load_state_dict(torch.load(self.path, map_location="cpu"))
        return model


# ─── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    """Lưu state_dict của model."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)
    print(f"  [ckpt] Đã lưu: {p}")


def load_checkpoint(path: str, model: torch.nn.Module) -> torch.nn.Module:
    """Đọc state_dict vào model (in-place)."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


# ─── Scaler loading ───────────────────────────────────────────────────────────

def load_scaler(cfg: dict, base_dir: Path, tag: str = ""):
    """Load fitted StandardScaler. Nếu tag != '' thì đọc scaler_{tag}.pkl."""
    import pickle
    suffix      = f"_{tag}" if tag else ""
    scaler_dir  = (base_dir / cfg["output"]["scaler_path"]).parent
    scaler_path = scaler_dir / f"scaler{suffix}.pkl"
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def load_lstm_baseline_mae(base_dir: Path, cfg: dict, tag: str = ""):
    """Đọc MAE 24h của LSTM từ results_log.json. Trả về float hoặc None.
    tag: cùng tag với lần train LSTM (để lấy đúng phiên bản)."""
    import json
    log_path = base_dir / cfg["output"]["results_log"]
    if not log_path.exists():
        return None
    with open(log_path, "r", encoding="utf-8") as f:
        logs = json.load(f)
    model_key    = f"lstm_{tag}" if tag else "lstm"
    lstm_entries = [e for e in logs
                    if e.get("model_name") == model_key and "best_mae_24h" in e]
    if not lstm_entries:
        return None
    return lstm_entries[-1]["best_mae_24h"]


# ─── Result logging ───────────────────────────────────────────────────────────

def log_result(log_path: str, entry: dict) -> None:
    """
    Ghi thêm một entry vào results_log.json.
    Entry nên có: model_name, mae_24h, mae_48h, skill_score, epochs_trained, ...
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Đọc log cũ nếu có
    if log_path.exists():
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []

    entry["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logs.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"  [log] Đã ghi: {log_path}")
