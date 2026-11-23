"""Performance statistics for an equity curve and a list of trades.

Kept independent of Nautilus so the maths is transparent and unit-testable.
Crypto trades every day, so annualisation uses 365 periods per year.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 365


@dataclass(frozen=True)
class PerformanceStats:
    total_return: float  # fraction, e.g. 0.12 == +12%
    num_trades: int  # closed round trips
    win_rate: float  # fraction of round trips with positive net PnL (NaN if none)
    max_drawdown: float  # fraction, negative, e.g. -0.25 == -25%
    sharpe_ratio: float  # annualised, risk-free rate 0
    final_equity: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def total_return(equity: pd.Series, starting_capital: float | None = None) -> float:
    """Final equity relative to ``starting_capital`` (default: first equity point)."""
    start = float(equity.iloc[0]) if starting_capital is None else starting_capital
    return float(equity.iloc[-1] / start - 1.0)


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline, as a negative fraction (0 if none)."""
    running_peak = equity.cummax()
    return float((equity / running_peak - 1.0).min())


def sharpe_ratio(equity: pd.Series, periods_per_year: int = PERIODS_PER_YEAR) -> float:
    """Annualised Sharpe of per-period simple returns (risk-free rate = 0)."""
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return float("nan")
    std = returns.std(ddof=1)
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return float("nan")
    return sum(1 for p in trade_pnls if p > 0) / len(trade_pnls)


def compute_stats(
    equity: pd.Series,
    trade_pnls: list[float],
    starting_capital: float | None = None,
) -> PerformanceStats:
    return PerformanceStats(
        total_return=total_return(equity, starting_capital),
        num_trades=len(trade_pnls),
        win_rate=win_rate(trade_pnls),
        max_drawdown=max_drawdown(equity),
        sharpe_ratio=sharpe_ratio(equity),
        final_equity=float(equity.iloc[-1]),
    )


def buy_and_hold_equity(
    closes: pd.Series,
    starting_capital: float,
    fee_rate: float,
    entry_price: float | None = None,
) -> pd.Series:
    """Equity of investing all capital at the first close and holding.

    The entry pays ``fee_rate`` on the notional; the final point also deducts
    the exit fee so the result is comparable with a strategy that ends flat.
    ``entry_price`` defaults to the first close.
    """
    entry = float(closes.iloc[0] if entry_price is None else entry_price)
    qty = starting_capital / (entry * (1.0 + fee_rate))
    equity = closes.astype(float) * qty
    equity.iloc[-1] = equity.iloc[-1] * (1.0 - fee_rate)
    return equity.rename("buy_and_hold")
