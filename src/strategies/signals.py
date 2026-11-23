"""A library of classic long/flat trading rules, expressed as daily target positions.

Each rule takes a daily OHLCV DataFrame (one row per session) and returns a
boolean Series: ``True`` means "be long after this session's close", ``False``
means "be in cash". The value for day *t* uses only data up to and including
day *t* (tests check this), and every trade it implies is executed at the next
session's open by :class:`~src.strategies.target_position.TargetPositionStrategy`.

The rules and their parameters are the standard published versions. They are
deliberately **not** tuned on this data, so comparing them is fair.

=====================  ======================================================================
buy_and_hold           always long (the baseline: what holding the stock does)
sma_10_50              long while SMA10 > SMA50 (this project's original rule)
golden_cross           long while SMA50 > SMA200 (the classic "golden cross")
trend_200              long while close > SMA200, checked on the first session of each month
                       (Faber 2007, "A Quantitative Approach to Tactical Asset Allocation")
momentum_12_1          long while the 12-month return, skipping the latest month, is > 0,
                       checked on the first session of each month (time-series momentum;
                       Moskowitz, Ooi & Pedersen 2012)
donchian_55_20         buy a close above the prior 55-day high, sell a close below the prior
                       20-day low (the Turtle-trader breakout)
macd                   long while MACD(12, 26) > its 9-day signal line
rsi2_reversion         buy when RSI(2) < 10 while above SMA200, sell when close > SMA5
                       (short-term mean reversion; Connors & Alvarez)
=====================  ======================================================================
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Signal = Callable[[pd.DataFrame], pd.Series]


def _month_start_mask(index: pd.DatetimeIndex) -> np.ndarray:
    """True on the first session of each calendar month (and the first row).

    Strictly causal: whether day *t* starts a new month depends only on day *t-1*.
    """
    months = index.tz_localize(None).to_period("M") if index.tz is not None else index.to_period("M")
    return np.r_[True, months[1:] != months[:-1]]


def _hold_between(decisions: pd.Series, mask: np.ndarray) -> pd.Series:
    """Apply ``decisions`` only on masked days and carry the position in between."""
    out = decisions.where(mask).ffill().fillna(False)
    return out.astype(bool)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=df.index)


def sma_cross(fast: int, slow: int) -> Signal:
    def rule(df: pd.DataFrame) -> pd.Series:
        c = df["close"]
        f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
        return (f > s).fillna(False)

    rule.__name__ = f"sma_{fast}_{slow}"
    return rule


def trend_200(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    above = c > c.rolling(200).mean()
    return _hold_between(above.where(c.rolling(200).count() == 200), _month_start_mask(df.index))


def momentum_12_1(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    ret = c.shift(21) / c.shift(252) - 1  # 12-month return excluding the most recent month
    return _hold_between((ret > 0).where(ret.notna()), _month_start_mask(df.index))


def donchian(entry: int = 55, exit_: int = 20) -> Signal:
    def rule(df: pd.DataFrame) -> pd.Series:
        c = df["close"].to_numpy()
        hi = df["high"].rolling(entry).max().shift(1).to_numpy()  # prior N-day high (excludes today)
        lo = df["low"].rolling(exit_).min().shift(1).to_numpy()
        pos = np.zeros(len(c), dtype=bool)
        long = False
        for i in range(len(c)):
            if not long and not np.isnan(hi[i]) and c[i] > hi[i]:
                long = True
            elif long and not np.isnan(lo[i]) and c[i] < lo[i]:
                long = False
            pos[i] = long
        return pd.Series(pos, index=df.index)

    rule.__name__ = f"donchian_{entry}_{exit_}"
    return rule


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    c = df["close"]
    line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    ready = np.arange(len(c)) >= slow + signal  # let the EMAs settle
    return pd.Series((line > sig).to_numpy() & ready, index=df.index)


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0).where(loss.notna())


def rsi2_reversion(df: pd.DataFrame) -> pd.Series:
    c = df["close"]
    rsi = _rsi(c, 2).to_numpy()
    sma200 = c.rolling(200).mean().to_numpy()
    sma5 = c.rolling(5).mean().to_numpy()
    cv = c.to_numpy()
    pos = np.zeros(len(c), dtype=bool)
    long = False
    for i in range(len(c)):
        if not long and not np.isnan(sma200[i]) and cv[i] > sma200[i] and rsi[i] < 10:
            long = True
        elif long and cv[i] > sma5[i]:
            long = False
        pos[i] = long
    return pd.Series(pos, index=df.index)


STRATEGIES: dict[str, tuple[str, Signal]] = {
    "buy_and_hold": ("Buy & hold (what you do now)", buy_and_hold),
    "sma_10_50": ("SMA 10/50 crossover", sma_cross(10, 50)),
    "golden_cross": ("Golden cross (SMA 50/200)", sma_cross(50, 200)),
    "trend_200": ("Above 200-day SMA, monthly", trend_200),
    "momentum_12_1": ("12-1 month momentum, monthly", momentum_12_1),
    "donchian_55_20": ("Donchian breakout 55/20", donchian(55, 20)),
    "macd": ("MACD 12/26/9", macd),
    "rsi2_reversion": ("RSI(2) mean reversion", rsi2_reversion),
}
WARMUP_BARS = 252  # the longest lookback (12-month momentum); all rules start trading after it
