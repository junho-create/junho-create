"""
Step5 attribution: feature-group permutation importance for a completed
`run_DoubleAdapt.py` run. Reloads `<logdir>/args.yaml` (exact panel/backbone
config), `<logdir>/factor_cols.json` (which flattened-window columns belong
to which factor), and `<logdir>/checkpoint.pt` (trained weights, no retrain).

For each feature-level group (price / macro / sector / target / related),
shuffles that group's values across test-period samples -- destroying only
that group's cross-sample signal while leaving row order intact for every
other column -- reruns online inference, and reports the Weighted Hit Rate
drop. This answers "how much did each fund-manager step actually move the
needle" per Step5, as opposed to a static coefficient/attention weight.

Usage
-----
    python -m forecasting_task.attribution --logdir logs/doubleadapt_full
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

FORECASTING_DIR = Path(__file__).resolve().parent
REPO_ROOT = FORECASTING_DIR.parent
DOUBLEADAPT_SRC = REPO_ROOT / "DoubleAdapt" / "src"
for _p in (str(REPO_ROOT), str(DOUBLEADAPT_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model as da_model  # noqa: E402
import utils as da_utils  # noqa: E402

from forecasting_task.backbones import build_backbone  # noqa: E402
from forecasting_task.preprocessing.levels import PRICE_FACTOR_NAMES  # noqa: E402
from forecasting_task.preprocessing.panel import PanelConfig, build_panel  # noqa: E402
from forecasting_task.run_DoubleAdapt import weighted_hit_rate  # noqa: E402


def _factor_level(name: str) -> str:
    if name in PRICE_FACTOR_NAMES:
        return "price"
    for level in ("macro", "sector", "related", "target"):
        if name.startswith(level):
            return level
    return "other"


def _permute_group(panel_df: pd.DataFrame, cols, mask: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    feat = panel_df["feature"].copy()
    sub = feat.loc[mask, cols]
    permuted = sub.iloc[rng.permutation(len(sub))]
    permuted.index = sub.index
    feat.loc[mask, cols] = permuted
    return pd.concat({"feature": feat, "label": panel_df["label"]}, axis=1).sort_index(
        level=["datetime", "instrument"]
    )


def main():
    p = argparse.ArgumentParser(description="Step5 feature-group permutation importance")
    p.add_argument("--logdir", required=True, help="Completed run_DoubleAdapt.py logdir "
                                                     "(needs args.yaml, factor_cols.json, checkpoint.pt).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_csv", default=None, help="Default: <logdir>/attribution.csv")
    cli_args = p.parse_args()

    with open(os.path.join(cli_args.logdir, "args.yaml")) as f:
        args = argparse.Namespace(**yaml.safe_load(f))
    with open(os.path.join(cli_args.logdir, "factor_cols.json")) as f:
        factor_cols = json.load(f)

    df = pd.read_parquet(args.data_csv) if str(args.data_csv).endswith(".parquet") else pd.read_csv(args.data_csv)
    df["date"] = pd.to_datetime(df["date"])
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        df = df[df["ticker"].isin(tickers)]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    cfg = PanelConfig(
        seq_len=args.seq_len, horizon=args.horizon, n_text_components=args.n_text_components,
        train_start=args.train_start, train_end=args.train_end, primary_emb_model=args.primary_emb_model,
        use_macro=args.use_macro, use_sector=args.use_sector,
        use_target_novelty=args.use_target_novelty, use_related_novelty=args.use_related_novelty,
        use_sector_novelty=getattr(args, "use_sector_novelty", False),
        novelty_window=args.novelty_window, novelty_quantile=args.novelty_quantile,
        novelty_consensus_frac=args.novelty_consensus_frac,
        sector_agreement_threshold=args.sector_agreement_threshold,
        macro_similarity_threshold=args.macro_similarity_threshold, macro_drift_window=args.macro_drift_window,
        use_text_signal=getattr(args, "use_text_signal", False),
        text_signal_hidden=getattr(args, "text_signal_hidden", 128),
        text_signal_layers=getattr(args, "text_signal_layers", 2),
        text_signal_heads=getattr(args, "text_signal_heads", 4),
        text_signal_epochs=getattr(args, "text_signal_epochs", 15),
    )
    panel_cache_dir = getattr(args, "panel_cache_dir", str(REPO_ROOT / "data" / ".panel_cache")) or None
    panel_df, raw_close, label_mean, label_std, factor_num, seq_len, factor_cols_check = build_panel(
        df, args.data_dir, cfg, cache_dir=panel_cache_dir
    )
    assert factor_cols_check == factor_cols, "factor_cols.json doesn't match rebuilt panel -- data/config drifted."

    calendar = pd.Series(sorted(panel_df.index.get_level_values("datetime").unique()))
    ta = da_utils.TimeAdjuster(calendar)
    segments = {
        "train": (args.train_start, args.train_end),
        "valid": (args.valid_start, args.valid_end),
        "test": (args.test_start, args.test_end),
    }
    horizon_param = args.horizon - 1
    rolling_tasks = da_utils.organize_all_tasks(
        segments, ta, step=args.step, trunc_days=args.horizon, rtype=da_utils.TimeAdjuster.SHIFT_SD,
    )
    test_slice = slice(ta.align_time(args.test_start, tp_type="start"), ta.align_time(args.test_end, tp_type="end"))

    net_model = build_backbone(
        args.backbone, factor_num, seq_len,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=args.backbone_dropout,
    )
    aux_weight = getattr(args, "aux_weight", 0.0)
    dir_scale = getattr(args, "dir_scale", 5.0)
    freeze_online = bool(getattr(args, "freeze_online", 0))
    if args.naive:
        framework = da_model.IncrementalManager(
            net_model, x_dim=factor_num * seq_len, lr_model=args.lr, weight_decay=args.weight_decay,
            need_permute=False, begin_valid_epoch=0,
            aux_weight=aux_weight, dir_scale=dir_scale, label_mean=label_mean, label_std=label_std,
            freeze_online=freeze_online,
        )
    else:
        framework = da_model.DoubleAdaptManager(
            net_model, x_dim=factor_num * seq_len, lr_model=args.lr, weight_decay=args.weight_decay,
            first_order=True, begin_valid_epoch=0, factor_num=factor_num,
            aux_weight=aux_weight, dir_scale=dir_scale, label_mean=label_mean, label_std=label_std,
            freeze_online=freeze_online,
            lr_da=args.lr_da, lr_ma=args.lr_ma, adapt_x=bool(args.adapt_x), adapt_y=bool(args.adapt_y),
            reg=args.reg, num_head=args.num_head, temperature=args.tau, need_permute=False,
        )
    framework.load_state_dict(torch.load(os.path.join(cli_args.logdir, "checkpoint.pt")))

    def run_inference(panel):
        test_data = da_utils.get_rolling_data(
            rolling_tasks["test"], data=panel, factor_num=factor_num, horizon=horizon_param,
            not_sequence=False, sequence_last_dim=True, to_tensor=True,
        )
        pred_y_all = framework.inference(test_data, date_slice=test_slice)
        raw_pred = pred_y_all["pred"] * label_std + label_mean
        raw_true = pred_y_all["label"] * label_std + label_mean
        anchor_close = raw_close.reindex(pred_y_all.index)
        pred_close = anchor_close * np.exp(raw_pred)
        true_close = anchor_close * np.exp(raw_true)
        return weighted_hit_rate(pred_close, true_close, anchor_close)

    baseline_hit_rate = run_inference(panel_df)
    print(f"[Attribution] baseline Weighted Hit Rate = {baseline_hit_rate:.4f}")

    test_start, test_end = pd.Timestamp(args.test_start), pd.Timestamp(args.test_end)
    dt = panel_df.index.get_level_values("datetime")
    test_mask = np.asarray((dt >= test_start) & (dt <= test_end))

    level_to_factor_idx = {}
    for fi, name in enumerate(factor_cols):
        level_to_factor_idx.setdefault(_factor_level(name), []).append(fi)

    rng = np.random.default_rng(cli_args.seed)
    rows = [{"group": "baseline", "n_factors": 0, "hit_rate": baseline_hit_rate, "delta": 0.0}]
    for level, factor_idxs in level_to_factor_idx.items():
        cols = [c for fi in factor_idxs for c in range(fi * seq_len, (fi + 1) * seq_len)]
        perturbed = _permute_group(panel_df, cols, test_mask, rng)
        hit_rate = run_inference(perturbed)
        delta = hit_rate - baseline_hit_rate
        print(f"[Attribution] group={level:8s} n_factors={len(factor_idxs):3d} "
              f"permuted_hit_rate={hit_rate:.4f} delta={delta:+.4f}")
        rows.append({"group": level, "n_factors": len(factor_idxs), "hit_rate": hit_rate, "delta": delta})

    out_csv = cli_args.out_csv or os.path.join(cli_args.logdir, "attribution.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[Attribution] wrote {out_csv}")


if __name__ == "__main__":
    main()
