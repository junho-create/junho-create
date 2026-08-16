"""
Chronos-2 rolling-forecast baseline with news-derived covariates.

Chronos-2 is one of the four pretrained models the competition explicitly
allows. Unlike TabPFN (tabular, permutation-invariant, no native notion of
time), Chronos-2 is a real time-series foundation model: it forecasts each
ticker's close price autoregressively from its own price history, and can
additionally condition on "past-only" covariates -- so we feed it the same
text-PCA + novelty/drift columns build_panel() already computes, satisfying
the "no numerical-only forecasting" rule (news actually enters the model,
not just the price series).

At each rolling anchor date (every --step trading days, matching the
DoubleAdapt run's cadence), it forecasts `horizon` trading days ahead for
every ticker in one predict_df() call (context = all history up to that
anchor date), and we score the median forecast's direction against the
realized direction.

Usage
-----
    python -m forecasting_task.run_chronos --step 25 --horizon 20
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from forecasting_task.preprocessing.panel import PanelConfig, build_panel  # noqa: E402


def build_long_frame(panel_df, raw_close, factor_cols, seq_len, cov_cols=None):
    """One row per (date, ticker): close (target) + text/novelty covariates (last-value-in-window,
    i.e. "as of that date"). Excludes the raw price factors (log_ret_close etc.) -- those are
    redundant with the target itself and we want the covariate set to be unambiguously the
    text-derived signal for rule-compliance purposes.

    Each extra covariate is itself a separate series Chronos-2 has to encode per ticker, so cost
    scales roughly linearly with len(cov_cols) * n_tickers (measured: 100 tickers took ~3s with 0
    covariates, ~12s with 2, ~124s with all 30 PCA+novelty columns on CPU). Default is the same
    two "news matters here" proxy columns run_DoubleAdapt.py's --news_weight_mult uses."""
    if cov_cols is None:
        cov_cols = [c for c in ("target_novelty_consensus", "related_drift_flag") if c in factor_cols]
    feat = panel_df["feature"]
    last_idx = {c: (factor_cols.index(c) + 1) * seq_len - 1 for c in cov_cols}
    cov_df = pd.DataFrame({c: feat.iloc[:, last_idx[c]].to_numpy() for c in cov_cols}, index=panel_df.index)
    wide = raw_close.rename("target").to_frame().join(cov_df)
    wide = wide.reset_index().rename(columns={"datetime": "timestamp", "instrument": "id"})
    return wide.sort_values(["id", "timestamp"]).reset_index(drop=True), cov_cols


def main():
    p = argparse.ArgumentParser(description="Chronos-2 rolling direction-forecast baseline")
    p.add_argument("--reference_logdir", default=str(REPO_ROOT / "logs" / "dlinear_final_compliant"))
    p.add_argument("--horizon", type=int, default=20)
    p.add_argument("--step", type=int, default=25, help="Trading days between rolling anchor dates.")
    p.add_argument("--valid_start", default="2022-01-01")
    p.add_argument("--valid_end", default="2022-12-31")
    p.add_argument("--test_start", default="2023-01-01")
    p.add_argument("--test_end", default="2023-12-01")
    p.add_argument("--device", default="cpu")
    p.add_argument("--covariates", default=None,
                   help="Comma-separated covariate column names (default: target_novelty_consensus,"
                        "related_drift_flag -- see build_long_frame docstring re: runtime cost).")
    p.add_argument("--out_dir", default=str(REPO_ROOT / "logs" / "chronos_direction"))
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    import yaml
    with open(os.path.join(args.reference_logdir, "args.yaml")) as f:
        ref = argparse.Namespace(**yaml.safe_load(f))

    df = pd.read_parquet(ref.data_csv)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    cfg = PanelConfig(
        seq_len=ref.seq_len, horizon=args.horizon, n_text_components=ref.n_text_components,
        train_start=ref.train_start, train_end=ref.train_end, primary_emb_model=ref.primary_emb_model,
        use_target_novelty=ref.use_target_novelty, use_related_novelty=ref.use_related_novelty,
        novelty_window=ref.novelty_window, novelty_quantile=ref.novelty_quantile,
        novelty_consensus_frac=ref.novelty_consensus_frac,
    )
    panel_df, raw_close, _, _, factor_num, seq_len, factor_cols = build_panel(
        df, ref.data_dir, cfg, cache_dir=ref.panel_cache_dir
    )
    cov_override = [c.strip() for c in args.covariates.split(",")] if args.covariates else None
    wide, cov_cols = build_long_frame(panel_df, raw_close, factor_cols, seq_len, cov_cols=cov_override)
    print(f"[Data] {len(wide)} rows, {wide['id'].nunique()} tickers, covariates={cov_cols}")

    all_dates = sorted(wide["timestamp"].unique())

    def anchors_in(start, end):
        rng = [d for d in all_dates if pd.Timestamp(start) <= d <= pd.Timestamp(end)]
        return rng[::args.step]

    valid_anchors = anchors_in(args.valid_start, args.valid_end)
    test_anchors = anchors_in(args.test_start, args.test_end)
    print(f"[Anchors] valid={len(valid_anchors)} test={len(test_anchors)}")

    from chronos import Chronos2Pipeline
    t0 = time.time()
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=args.device)
    print(f"[Model] loaded in {time.time()-t0:.1f}s")

    close_lookup = wide.set_index(["id", "timestamp"])["target"]

    def target_date_for(ticker, anchor_date):
        dates = wide.loc[wide["id"] == ticker, "timestamp"]
        pos = dates.searchsorted(anchor_date)
        idx = pos + args.horizon
        return dates.iloc[idx] if idx < len(dates) else None

    def run_anchor(anchor_date):
        context_df = wide[wide["timestamp"] <= anchor_date]
        pred = pipeline.predict_df(
            context_df, prediction_length=args.horizon, quantile_levels=[0.5],
            id_column="id", timestamp_column="timestamp", target="target", freq="B",
        )
        # last predicted step per id == horizon trading days ahead
        last_step = pred.sort_values(["id", "timestamp"]).groupby("id").tail(1)
        rows = []
        for _, r in last_step.iterrows():
            ticker = r["id"]
            tgt_date = target_date_for(ticker, anchor_date)
            if tgt_date is None:
                continue
            anchor_close = close_lookup.get((ticker, anchor_date))
            true_close = close_lookup.get((ticker, tgt_date))
            if anchor_close is None or true_close is None:
                continue
            rows.append({
                "ticker": ticker, "anchor_date": anchor_date, "target_date": tgt_date,
                "anchor_close": anchor_close, "pred_close": r["predictions"], "true_close": true_close,
            })
        return rows

    def score(anchors, name):
        all_rows = []
        for i, a in enumerate(anchors):
            t1 = time.time()
            rows = run_anchor(a)
            all_rows.extend(rows)
            print(f"[{name}] anchor {i+1}/{len(anchors)} ({a.date()}) -> {len(rows)} preds in {time.time()-t1:.1f}s")
        out = pd.DataFrame(all_rows)
        if len(out) == 0:
            print(f"[{name}] no scoreable rows")
            return out, float("nan")
        hit = (np.sign(out["pred_close"] - out["anchor_close"]) == np.sign(out["true_close"] - out["anchor_close"]))
        hit_rate = float(hit.mean())
        print(f"[{name}] n={len(out)} hit_rate={hit_rate:.4f}")
        return out, hit_rate

    valid_df, valid_hit = score(valid_anchors, "valid_2022")
    test_df, test_hit = score(test_anchors, "test_2023")

    valid_df.to_csv(os.path.join(args.out_dir, "valid_preds.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir, "test_preds.csv"), index=False)
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write(f"valid_2022_hit_rate={valid_hit:.6f} (n={len(valid_df)})\n")
        f.write(f"test_2023_hit_rate={test_hit:.6f} (n={len(test_df)})\n")
    print(f"[Done] wrote {args.out_dir}/results.txt")


if __name__ == "__main__":
    main()
