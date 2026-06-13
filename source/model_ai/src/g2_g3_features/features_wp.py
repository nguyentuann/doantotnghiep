"""
features_wp.py
--------------
G2/G3 (WP-full variant): Pipeline features + sequences cho wp_full tag.

Chiến lược:
  - TRAIN  : Tất cả WP storms 1960–2016  (học pattern chung của xoáy thuận NTĐ)
  - VAL    : Chỉ SCS storms 2017–2020    (đánh giá đúng target domain)
  - TEST   : Chỉ SCS storms 2021–2024    (so sánh fair với các tag khác)

Output (không ghi đè file hiện có):
  data/features/feature_matrix_wp_full.csv
  data/features/sequences_wp_full.npz
  models/scaler_wp_full.pkl

Chạy:
  cd source/model_ai
  python -m src.g2_g3_features.features_wp
"""

import math
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.g1_data.data_loader_wp import load_all_wp, save_wp_data, load_config
from src.g2_g3_features.features import build_features

TAG = "wp_full"


# ---------------------------------------------------------------------------
# G3: make_sequences với is_scs_storm trong meta
# ---------------------------------------------------------------------------

def make_sequences_wp(feat_df: pd.DataFrame, config: dict, scs_storm_sids: set):
    """
    Tạo sequences giống make_sequences() nhưng thêm `is_scs_storm` vào meta.

    Parameters
    ----------
    scs_storm_sids : set — tập SID của các storm có ít nhất 1 điểm trong SCS box
    """
    feat_names = config["features"]["names"]
    lookback   = config["model"]["lookback"]
    anchor     = config["model"]["anchor_steps"]
    step_24h   = anchor[0]
    step_48h   = anchor[1]

    X_list, y_list, meta_list = [], [], []

    for sid, grp in feat_df.groupby("SID"):
        grp  = grp.sort_values("ISO_TIME").reset_index(drop=True)
        n    = len(grp)
        min_len = lookback + step_48h
        if n < min_len:
            continue

        vals   = grp[feat_names].values.astype(np.float32)
        lats   = grp["LAT"].values
        lons   = grp["LON"].values
        season = grp["SEASON"].iloc[0]
        times  = grp["ISO_TIME"].values
        is_scs = sid in scs_storm_sids

        for i in range(n - min_len + 1):
            x_seq   = vals[i : i + lookback]
            lat_24h = lats[i + lookback + step_24h - 1]
            lon_24h = lons[i + lookback + step_24h - 1]
            lat_48h = lats[i + lookback + step_48h - 1]
            lon_48h = lons[i + lookback + step_48h - 1]

            X_list.append(x_seq)
            y_list.append([lat_24h, lon_24h, lat_48h, lon_48h])
            meta_list.append({
                "SID":          sid,
                "SEASON":       season,
                "init_time":    times[i + lookback - 1],
                "is_scs_storm": is_scs,
            })

    X    = np.array(X_list, dtype=np.float32)
    y    = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)

    print(f"[G3-WP] Sequences: X{X.shape}, y{y.shape}")
    n_scs = meta["is_scs_storm"].sum()
    print(f"        SCS sequences: {n_scs:,} / {len(meta):,}  ({n_scs/len(meta)*100:.1f}%)")
    return X, y, meta


# ---------------------------------------------------------------------------
# G3: split + scale — train=all WP, val/test=SCS only
# ---------------------------------------------------------------------------

def split_and_scale_wp(X, y, meta, config, tag: str = TAG):
    """
    Chia split với chiến lược WP-full:
      train : năm trong [train_start, train_end]  — ALL WP storms
      val   : năm trong [val_start, val_end]      — chỉ SCS storms
      test  : năm trong [test_start, test_end]    — chỉ SCS storms
    """
    split_cfg = config["split"]
    base_dir  = Path(__file__).parent.parent.parent
    suffix    = f"_{tag}"
    scaler_dir  = (base_dir / config["output"]["scaler_path"]).parent
    scaler_path = scaler_dir / f"scaler{suffix}.pkl"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    season = meta["SEASON"].values
    is_scs = meta["is_scs_storm"].values

    tr_mask  = (season >= split_cfg["train"][0]) & (season <= split_cfg["train"][1])
    val_mask = (season >= split_cfg["val"][0])   & (season <= split_cfg["val"][1]) & is_scs
    te_mask  = (season >= split_cfg["test"][0])  & (season <= split_cfg["test"][1]) & is_scs

    print(f"\n[G3-WP] Split:")
    print(f"  train (all WP {split_cfg['train'][0]}–{split_cfg['train'][1]}): {tr_mask.sum():,} seq")
    print(f"  val   (SCS    {split_cfg['val'][0]}–{split_cfg['val'][1]}):  {val_mask.sum():,} seq")
    print(f"  test  (SCS    {split_cfg['test'][0]}–{split_cfg['test'][1]}):  {te_mask.sum():,} seq")

    X_train, y_train = X[tr_mask],  y[tr_mask]
    X_val,   y_val   = X[val_mask], y[val_mask]
    X_test,  y_test  = X[te_mask],  y[te_mask]

    # StandardScaler: fit CHỈ trên train
    n_feat = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_feat))

    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(X_train.shape)
    X_val   = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test  = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"[G3-WP] Scaler lưu: {scaler_path}")

    # Checkpoint G3
    min_seq = config["checkpoints"]["g3_min_sequences"]
    total   = tr_mask.sum() + val_mask.sum() + te_mask.sum()
    if total < min_seq:
        raise ValueError(f"[G3-WP FAIL] Chỉ có {total} sequences, cần >= {min_seq}.")
    print(f"[G3-WP] PASS: {total:,} sequences >= {min_seq}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "meta_train": meta[tr_mask].reset_index(drop=True),
        "meta_val":   meta[val_mask].reset_index(drop=True),
        "meta_test":  meta[te_mask].reset_index(drop=True),
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="G2/G3 WP-full pipeline")
    parser.add_argument("--tag",        default=None,
                        help="Tag output (default: wp_full hoặc wp_steering nếu --steering). "
                             "Ví dụ: wp_1945")
    parser.add_argument("--year-start", type=int, default=None,
                        help="Override year_start trong config (ví dụ: 1945).")
    parser.add_argument("--steering",   action="store_true",
                        help="Thêm steering_u và steering_v từ ERA5 (14 features). "
                             "Tag mặc định: wp_steering.")
    args = parser.parse_args()

    # Tag mặc định phụ thuộc vào --steering
    if args.tag is not None:
        tag = args.tag
    elif args.steering:
        tag = "wp_steering"
    else:
        tag = "wp_full"

    cfg      = load_config()
    base_dir = Path(__file__).parent.parent.parent
    suffix   = f"_{tag}"

    # Override year_start nếu được chỉ định
    if args.year_start is not None:
        old_year = cfg["geography"]["year_start"]
        cfg["geography"]["year_start"] = args.year_start
        cfg["split"]["train"][0]       = args.year_start
        print(f"[G1-WP] Override year_start: {old_year} → {args.year_start}")

    year_start = cfg["geography"]["year_start"]

    # --- G1-WP: Load toàn bộ WP storms ---
    # File CSV đặt tên theo year_start để tránh ghi đè
    wp_csv = base_dir / f"data/processed/bao_bien_dong_wp_{year_start}.csv"
    if wp_csv.exists():
        print(f"[G1-WP] File đã có, đọc lại: {wp_csv.name}")
        df_wp = pd.read_csv(wp_csv, low_memory=False, parse_dates=["ISO_TIME"])
        print(f"        {len(df_wp):,} rows, {df_wp['SID'].nunique():,} storms")
    else:
        print(f"[G1-WP] Tạo mới: {wp_csv.name}")
        df_wp = load_all_wp(config=cfg)
        df_wp.to_csv(wp_csv, index=False)
        size_mb = wp_csv.stat().st_size / 1024 / 1024
        print(f"[G1-WP] Đã lưu: {wp_csv} ({size_mb:.1f} MB)")

    # Tập SCS storm SIDs (để đánh dấu val/test)
    scs_storm_sids = set(df_wp[df_wp["is_scs_storm"]]["SID"].unique())
    print(f"[G1-WP] SCS storm SIDs: {len(scs_storm_sids):,}")

    # --- G2: Tính features ---
    feat_names = list(cfg["features"]["names"])
    if "wind_shear" in feat_names or "sst_actual" in feat_names:
        raise ValueError(
            "features_wp.py không hỗ trợ ERA5/SST. "
            "Dùng features.py với --tag 14feat nếu cần."
        )
    print("[G2-WP] Bỏ qua ERA5/SST — không có trong feature list")
    feat_df = build_features(df_wp, cfg)

    # --- Steering flow (tuỳ chọn) ---
    if args.steering:
        from src.g2_g3_features.era5_extractor import extract_steering_features
        print("[G2-WP] Trích xuất steering flow từ ERA5 (±5°)...")
        feat_df = extract_steering_features(feat_df, cfg, radius_deg=5.0)
        feat_names = feat_names + ["steering_u", "steering_v"]
        cfg["features"]["names"]     = feat_names
        cfg["features"]["n_features"] = len(feat_names)
        print(f"[G2-WP] Feature list ({len(feat_names)}): {feat_names}")

    # Lưu feature_matrix_{tag}.csv
    feat_stem = Path(cfg["data"]["feature_file"])
    feat_path = base_dir / feat_stem.parent / f"{feat_stem.stem}{suffix}{feat_stem.suffix}"
    feat_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_csv(feat_path, index=False)
    size_mb = feat_path.stat().st_size / 1024 / 1024
    print(f"[G2-WP] Đã lưu: {feat_path} ({size_mb:.1f} MB)")

    # --- G3: Tạo sequences ---
    X, y, meta = make_sequences_wp(feat_df, cfg, scs_storm_sids)

    # --- G3: Split + Scale ---
    data = split_and_scale_wp(X, y, meta, cfg, tag=tag)

    # Lưu sequences_{tag}.npz
    seq_path = base_dir / f"data/features/sequences{suffix}.npz"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        seq_path,
        X_train=data["X_train"], y_train=data["y_train"],
        X_val=data["X_val"],     y_val=data["y_val"],
        X_test=data["X_test"],   y_test=data["y_test"],
    )
    print(f"[G3-WP] Sequences lưu: {seq_path}")
    print(f"\n[DONE] Tag '{tag}' sẵn sàng train:")
    print(f"  python -m src.g5_g6_train.train_lstm        --tag {tag}")
    print(f"  python -m src.g5_g6_train.train_bilstm      --tag {tag}")
    print(f"  python -m src.g5_g6_train.train_transformer --tag {tag}")
