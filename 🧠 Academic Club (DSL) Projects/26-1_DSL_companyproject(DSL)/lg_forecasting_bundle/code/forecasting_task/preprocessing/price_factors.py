"""Step1: per-ticker stationary price factors (the time-series backbone's raw input)."""
import numpy as np
import pandas as pd

from .levels import PRICE_FACTOR_NAMES

__all__ = ["PRICE_FACTOR_NAMES", "build_price_factors"]


def build_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    """`df` sorted by (ticker, date)."""
    g = df.groupby("ticker", sort=False)
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    out["log_ret_close"] = np.log(close / g["close"].shift(1))
    out["hl_range"] = (df["high"] - df["low"]) / close
    out["oc_range"] = (df["open"] - close) / close
    out["log_ret_volume"] = np.log(df["volume"].clip(lower=1) / g["volume"].shift(1).clip(lower=1))
    out["mom_5d"] = np.log(close / g["close"].shift(5))
    return out
