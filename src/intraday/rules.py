"""Classic day-trading rules as per-minute long/cash targets.

Each rule returns a boolean array aligned with the minute bars: ``True`` means
"be long after this bar's close". Trades implied by a change are filled at the
**next** bar's open. Every day-trading rule is flat by the close: its target is
forced to ``False`` on each session's last two bars, so the exit fills at the
open of the final minute. Nothing is held overnight. All rules are long only,
using standard published settings (nothing is tuned to this data).

==================  =====================================================================
hold_intraday       long from the first minute to the close, every day (intraday baseline)
buy_and_hold        long throughout, overnight included (what holding the stock does)
orb_30              opening-range breakout: after 10:00, buy a close above the first 30
                    minutes' high; sell on a close below its low or at the close
                    (one trade per day; Crabel / Zarattini & Aziz)
intraday_momentum   if the return from the prior close to 10:00 is positive, hold the last
                    30 minutes (Gao, Han, Li & Zhou 2018, "Market intraday momentum")
vwap_reversion      buy a close more than 2 standard deviations (30-minute) below the day's
                    VWAP between 10:00 and the last half-hour; sell on a close back at VWAP
gap_fade            if the open gaps down more than 0.5% from the prior close, buy after
                    the first minute; sell when the gap closes, at noon, or at the close
==================  =====================================================================
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Rule = Callable[[pd.DataFrame], np.ndarray]
FLAT_TAIL = 2  # last two bars of each session are forced flat


def _sessions(df: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each session, in order."""
    codes = pd.factorize(df["session"])[0]
    bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])
    return [np.arange(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]


def close_minute(df: pd.DataFrame) -> np.ndarray:
    """Official close (minutes after midnight New York) for each row: 16:00, or 13:00 on early-close days.

    Taken from the exchange calendar, not from the data, so a missing bar can
    never make a rule hold through the close.
    """
    from src.intraday.data import CLOSE_MINUTE, EARLY_CLOSE_MINUTE, is_early_close

    early = {d: is_early_close(d) for d in pd.unique(df["session"])}
    return np.array([EARLY_CLOSE_MINUTE if early[d] else CLOSE_MINUTE for d in df["session"]])


def _flat_at_close(target: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Force cash for the final ``FLAT_TAIL`` minutes before the official close."""
    out = target.copy()
    out[df["minute"].to_numpy() >= close_minute(df) - FLAT_TAIL] = False
    return out


def _prior_close(df: pd.DataFrame) -> dict[int, float]:
    """Session index -> previous session's last close."""
    closes = df["close"].to_numpy()
    sess = _sessions(df)
    return {k: float(closes[sess[k - 1][-1]]) for k in range(1, len(sess))}


def hold_intraday(df: pd.DataFrame) -> np.ndarray:
    return _flat_at_close(np.ones(len(df), dtype=bool), df)


def buy_and_hold(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df), dtype=bool)


def orb_30(df: pd.DataFrame) -> np.ndarray:
    c, h, lo, m = (df[k].to_numpy() for k in ("close", "high", "low", "minute"))
    out = np.zeros(len(df), dtype=bool)
    for rows in _sessions(df):
        opening = rows[m[rows] < 600]
        if len(opening) < 25:  # need (most of) the first 30 minutes
            continue
        hi, low = h[opening].max(), lo[opening].min()
        long = entered = False
        for i in rows[m[rows] >= 600]:
            if not long and not entered and c[i] > hi:
                long = entered = True
            elif long and c[i] < low:
                long = False
            out[i] = long
    return _flat_at_close(out, df)


def intraday_momentum(df: pd.DataFrame) -> np.ndarray:
    c, m = df["close"].to_numpy(), df["minute"].to_numpy()
    close = close_minute(df)
    prior = _prior_close(df)
    out = np.zeros(len(df), dtype=bool)
    for k, rows in enumerate(_sessions(df)):
        if k not in prior:
            continue
        first_half_hour = rows[m[rows] < 600]
        if len(first_half_hour) == 0 or m[first_half_hour[-1]] < 595:
            continue
        r = c[first_half_hour[-1]] / prior[k] - 1
        if r > 0:
            # Decide on the 15:29 close (31 minutes before the close) -> hold from 15:30 to 15:59.
            out[rows[m[rows] >= close[rows] - 31]] = True
    return _flat_at_close(out, df)


def vwap_reversion(df: pd.DataFrame, k_std: float = 2.0, window: int = 30) -> np.ndarray:
    c, m = df["close"].to_numpy(), df["minute"].to_numpy()
    tp = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy()
    v = df["volume"].to_numpy()
    out = np.zeros(len(df), dtype=bool)
    for rows in _sessions(df):
        pv, vv = np.cumsum(tp[rows] * v[rows]), np.cumsum(v[rows])
        vwap = np.where(vv > 0, pv / np.maximum(vv, 1e-12), c[rows])
        std = pd.Series(c[rows]).rolling(window).std().to_numpy()
        close = close_minute(df.iloc[rows[:1]])[0]
        long = False
        for j, i in enumerate(rows):
            if not long and 600 <= m[i] < close - 30 and not np.isnan(std[j]) and c[i] < vwap[j] - k_std * std[j]:
                long = True
            elif long and c[i] >= vwap[j]:
                long = False
            out[i] = long
    return _flat_at_close(out, df)


def gap_fade(df: pd.DataFrame, min_gap: float = 0.005) -> np.ndarray:
    o, c, m = df["open"].to_numpy(), df["close"].to_numpy(), df["minute"].to_numpy()
    prior = _prior_close(df)
    out = np.zeros(len(df), dtype=bool)
    for k, rows in enumerate(_sessions(df)):
        if k not in prior:
            continue
        pc = prior[k]
        if o[rows[0]] / pc - 1 >= -min_gap:
            continue
        for i in rows:
            if c[i] >= pc or m[i] >= 719:  # gap closed, or noon
                break
            out[i] = True
    return _flat_at_close(out, df)


INTRADAY_RULES: dict[str, tuple[str, Rule]] = {
    "buy_and_hold": ("Buy & hold (overnight included)", buy_and_hold),
    "hold_intraday": ("Long every day, open to close", hold_intraday),
    "orb_30": ("Opening-range breakout (30 min)", orb_30),
    "intraday_momentum": ("Intraday momentum (last 30 min)", intraday_momentum),
    "vwap_reversion": ("VWAP mean reversion", vwap_reversion),
    "gap_fade": ("Gap-down fade", gap_fade),
}
