"""
Self-contained Kaggle submission generator for the DLinear + DoubleAdapt
"dlinear_final_compliant" model (best config, self-scored 2023 weighted hit
rate = 0.5900, public LB equal-weight 2022 = 0.54738).

Unlike forecasting_task/generate_kaggle_submission.py, this script does NOT
rebuild the feature panel from the raw 6x16GB embedding parquets (that peaks at
~76GB RAM and would OOM a Kaggle notebook). Instead it loads a pre-built panel
pickle -- the exact byte-for-byte panel the checkpoint was trained on -- so the
submission is reproduced deterministically in ~2 minutes with no large-memory
step. The panel is itself derived only from competition-provided data (prices +
the six allowed precomputed embeddings), so this uses no external data/models.

Bundle layout expected (all paths overridable via CLI):
    <bundle>/code/               (this repo: forecasting_task/, DoubleAdapt/)
    <bundle>/checkpoint.pt
    <bundle>/args.yaml
    <bundle>/panel.pkl           (pickled build_panel() output tuple)
    <bundle>/test.parquet        (prices + text ids; competition-provided)

Usage on Kaggle (see the notebook cells):
    python code/forecasting_task/kaggle_submit.py \
        --bundle /kaggle/input/<your-code-dataset> \
        --out /kaggle/working/submission.csv
"""
import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description="Kaggle submission from a pre-built panel + trained checkpoint")
    p.add_argument("--bundle", required=True, help="Directory holding code/, checkpoint.pt, args.yaml, panel.pkl, test.parquet")
    p.add_argument("--code_dir", default=None, help="Override path to the repo code root (default: <bundle>/code)")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint.pt path")
    p.add_argument("--args_yaml", default=None, help="Override args.yaml path")
    p.add_argument("--panel_pkl", default=None, help="Override panel.pkl path")
    p.add_argument("--test_parquet", default=None, help="Override test.parquet (prices) path")
    p.add_argument("--out", default="/kaggle/working/submission.csv")
    p.add_argument("--sub_start", default="2022-01-01")
    p.add_argument("--sub_end", default="2023-12-31")
    cli = p.parse_args()

    bundle = Path(cli.bundle)
    code_dir = Path(cli.code_dir) if cli.code_dir else bundle / "code"
    checkpoint = Path(cli.checkpoint) if cli.checkpoint else bundle / "checkpoint.pt"
    args_yaml = Path(cli.args_yaml) if cli.args_yaml else bundle / "args.yaml"
    panel_pkl = Path(cli.panel_pkl) if cli.panel_pkl else bundle / "panel.pkl"
    test_parquet = Path(cli.test_parquet) if cli.test_parquet else bundle / "test.parquet"

    # Repo imports: DoubleAdapt/src holds model.py/utils.py/net.py/higher_optim.py
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir / "DoubleAdapt" / "src"))
    import model as da_model  # noqa: E402
    import utils as da_utils  # noqa: E402
    from forecasting_task.backbones import build_backbone  # noqa: E402

    with open(args_yaml) as f:
        args = argparse.Namespace(**yaml.safe_load(f))
    set_seed(getattr(args, "seed", 7))

    print(f"[Submit] loading pre-built panel: {panel_pkl}")
    with open(panel_pkl, "rb") as f:
        panel_df, raw_close, label_mean, label_std, factor_num, seq_len, factor_cols = pickle.load(f)
    print(f"[Submit] panel shape={panel_df.shape} factor_num={factor_num} seq_len={seq_len}")

    df = pd.read_parquet(test_parquet)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    all_tickers = sorted(df["ticker"].unique().tolist())

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
    rolling_tasks_data = {
        k: da_utils.get_rolling_data(
            rolling_tasks[k], data=panel_df, factor_num=factor_num, horizon=horizon_param,
            not_sequence=False, sequence_last_dim=True, to_tensor=True,
        )
        for k in ["valid", "test"]
    }

    net_model = build_backbone(
        args.backbone, factor_num, seq_len,
        hidden_size=args.hidden_size, num_layers=args.num_layers,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_ff=args.d_ff,
        dropout=args.backbone_dropout,
    )
    aux_weight = getattr(args, "aux_weight", 0.0)
    dir_scale = getattr(args, "dir_scale", 5.0)
    freeze_online = bool(getattr(args, "freeze_online", 1))
    framework = da_model.DoubleAdaptManager(
        net_model, x_dim=factor_num * seq_len, lr_model=args.lr, weight_decay=args.weight_decay,
        first_order=True, begin_valid_epoch=0, factor_num=factor_num,
        lr_da=args.lr_da, lr_ma=args.lr_ma, adapt_x=bool(args.adapt_x), adapt_y=bool(args.adapt_y),
        reg=args.reg, num_head=args.num_head, temperature=args.tau, need_permute=False,
        freeze_online=freeze_online,
        aux_weight=aux_weight, dir_scale=dir_scale, label_mean=label_mean, label_std=label_std,
    )
    map_loc = None if torch.cuda.is_available() else "cpu"
    framework.load_state_dict(torch.load(str(checkpoint), map_location=map_loc))
    print(f"[Submit] checkpoint loaded ({'cuda' if torch.cuda.is_available() else 'cpu'})")

    def run_inference(seg_name, start, end):
        date_slice = slice(ta.align_time(start, tp_type="start"), ta.align_time(end, tp_type="end"))
        pred_y_all = framework.inference(rolling_tasks_data[seg_name], date_slice=date_slice)
        raw_pred = pred_y_all["pred"] * label_std + label_mean
        anchor_close = raw_close.reindex(pred_y_all.index)
        pred_close = anchor_close * np.exp(raw_pred)
        out = pred_close.rename("pred_close").to_frame()
        out.index.names = ["anchor_date", "ticker"]
        return out

    print("[Submit] inference on valid (2022) segment...")
    preds_2022 = run_inference("valid", args.valid_start, args.valid_end)
    print("[Submit] inference on test (2023) segment...")
    preds_2023 = run_inference("test", args.test_start, args.test_end)
    anchor_preds = pd.concat([preds_2022, preds_2023]).sort_index()

    # Map each anchor-date prediction to its target date (anchor + horizon trading days)
    horizon = args.horizon
    target_rows = []
    for ticker, g in df.groupby("ticker", sort=False):
        dates = g["date"].sort_values().reset_index(drop=True)
        pos_of_date = pd.Series(dates.index.values, index=dates.values)
        if ticker not in anchor_preds.index.get_level_values("ticker"):
            continue
        sub = anchor_preds.xs(ticker, level="ticker", drop_level=False)
        for (anchor_date, _), row in sub.iterrows():
            pos = pos_of_date.get(anchor_date)
            if pos is not None and pos + horizon < len(dates):
                target_date = dates.iloc[pos + horizon]
            else:
                target_date = anchor_date + pd.tseries.offsets.BDay(horizon)
            target_rows.append((target_date, ticker, row["pred_close"]))

    pred_df = pd.DataFrame(target_rows, columns=["date", "ticker", "pred_close"])
    pred_df = pred_df.groupby(["date", "ticker"], as_index=False)["pred_close"].mean()
    pred_lookup = pred_df.set_index(["date", "ticker"])["pred_close"]

    sub_dates = pd.date_range(cli.sub_start, cli.sub_end, freq="B")
    rows = []
    for ticker in all_tickers:
        actual = df[df["ticker"] == ticker].set_index("date")["close"].sort_index()
        last_val = actual[actual.index < cli.sub_start]
        last_val = float(last_val.iloc[-1]) if len(last_val) else (float(actual.iloc[0]) if len(actual) else 100.0)
        for d in sub_dates:
            if (d, ticker) in pred_lookup.index:
                val = float(pred_lookup.loc[(d, ticker)]); last_val = val
            elif d in actual.index:
                val = float(actual.loc[d]); last_val = val
            else:
                val = last_val
            rows.append({"ID": f"{ticker}_{d.strftime('%Y-%m-%d')}", "Close": val})

    sub_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    sub_df.to_csv(cli.out, index=False)
    print(f"[Submit] wrote {len(sub_df)} rows to {cli.out}")
    print(sub_df.head())
    print(sub_df.tail())


if __name__ == "__main__":
    main()
