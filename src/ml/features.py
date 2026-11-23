"""Leakage-safe features and labels for daily crypto bars.

Rules that keep this honest:

* A feature row for day *t* uses only OHLCV up to and including day *t*'s close.
  Everything is built from backward-looking ``rolling``/``shift(+k)``/``pct_change``
  operations; there is no ``shift(-k)`` anywhere in :func:`make_features`.
* All features are scale-free (returns, ratios, z-scores), so one model can be
  trained on BTC, ETH and SOL together.
* The label for day *t* looks *forward* ``horizon`` days. When training, the
  last ``horizon`` rows before a test window must be dropped ("purged"),
  because their labels overlap the test period. See :mod:`src.ml.walk_forward`.

The rows are indexed by the bar's **close** time (open time + 1 day), the same
timestamp the Nautilus strategy sees as ``bar.ts_event``, so predictions line
up with bars exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "ret_1", "ret_3", "ret_5", "ret_10", "ret_20", "ret_60",
    "vol_10", "vol_30", "vol_ratio",
    "sma_10_50", "close_sma20", "close_sma200",
    "rsi_14", "dist_high_90", "dist_low_90",
    "volume_z_30", "range_10",
]
WARMUP_BARS = 200  # longest lookback (SMA 200)


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI, scaled to [-0.5, 0.5] (0 = neutral)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(loss.notna()) / 100 - 0.5


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for one symbol's daily OHLCV (index = open time, UTC).

    Returns a DataFrame indexed by bar close time with columns :data:`FEATURES`.
    The first :data:`WARMUP_BARS` rows contain NaNs and should be dropped.
    """
    close, high, low = df["close"], df["high"], df["low"]
    logret = np.log(close).diff()
    f = pd.DataFrame(index=df.index)

    for k in (1, 3, 5, 10, 20, 60):
        f[f"ret_{k}"] = np.log(close / close.shift(k))

    f["vol_10"] = logret.rolling(10).std()
    f["vol_30"] = logret.rolling(30).std()
    f["vol_ratio"] = f["vol_10"] / f["vol_30"]

    sma10, sma20 = close.rolling(10).mean(), close.rolling(20).mean()
    sma50, sma200 = close.rolling(50).mean(), close.rolling(200).mean()
    f["sma_10_50"] = sma10 / sma50 - 1
    f["close_sma20"] = close / sma20 - 1
    f["close_sma200"] = close / sma200 - 1

    f["rsi_14"] = _rsi(close)
    f["dist_high_90"] = close / high.rolling(90).max() - 1
    f["dist_low_90"] = close / low.rolling(90).min() - 1

    logvol = np.log(df["volume"].replace(0, np.nan))
    vol_std = logvol.rolling(30).std()
    z = (logvol - logvol.rolling(30).mean()) / vol_std.replace(0, np.nan)
    f["volume_z_30"] = z.where(vol_std != 0, 0.0).where(vol_std.notna())  # flat volume -> 0
    f["range_10"] = ((high - low) / close).rolling(10).mean()

    f.index = _close_index(df)
    return f[FEATURES]


def make_labels(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Forward log return over ``horizon`` days and its sign, indexed by close time.

    ``fwd_ret`` at day *t* is ``log(close[t + horizon] / close[t])``: the return
    from acting at *t*'s close. The last ``horizon`` rows are NaN.
    """
    close = df["close"]
    fwd = np.log(close.shift(-horizon) / close)  # the only forward-looking line in this module
    out = pd.DataFrame({"fwd_ret": fwd, "label": (fwd > 0).astype(float).where(fwd.notna())})
    out.index = _close_index(df)
    return out


def _close_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    step = pd.Timedelta(df.index[1] - df.index[0]) if len(df) > 1 else pd.Timedelta(days=1)
    return pd.DatetimeIndex(df.index + step, name="close_time")


def build_dataset(frames: dict[str, pd.DataFrame], horizon: int = 5) -> pd.DataFrame:
    """Stack features + labels for several symbols into one long table.

    Columns: ``symbol``, :data:`FEATURES`, ``fwd_ret``, ``label``. Index: close
    time (not unique across symbols). Warm-up rows are dropped; rows whose
    label is not yet known (the last ``horizon`` days) are kept with NaN labels
    so they can still be predicted.
    """
    parts = []
    for symbol, df in frames.items():
        part = make_features(df).join(make_labels(df, horizon))
        part = part.iloc[WARMUP_BARS:].dropna(subset=FEATURES)
        part.insert(0, "symbol", symbol)
        parts.append(part)
    return pd.concat(parts).sort_index(kind="stable")
