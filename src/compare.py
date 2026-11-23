"""Compare trading rules on the same symbols, period, costs and execution.

For each symbol, every rule in :data:`src.strategies.signals.STRATEGIES` is run
through Nautilus with :class:`TargetPositionStrategy`. All rules start trading
on the same session (after the longest warm-up), so buy & hold and every
trading rule cover exactly the same days.

Picking "the best" of several rules on one history is itself a form of
overfitting, so this also reports whether the ranking is stable: each run's
equity curve is split into first and second halves, and the rank correlation
of the rules' Sharpe ratios across the two halves is measured. Rankings that
flip between halves mean the "winner" was mostly luck.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import bar_close_times
from src.data_loader import bar_type_for
from src.data_loader import equity_instrument
from src.data_loader import open_quotes
from src.data_loader import to_nautilus_bars
from src.metrics import PERIODS_PER_YEAR
from src.metrics import max_drawdown
from src.metrics import sharpe_ratio
from src.research import ResearchSettings
from src.strategies.signals import STRATEGIES
from src.strategies.signals import WARMUP_BARS
from src.strategies.target_position import TargetPositionConfig
from src.strategies.target_position import TargetPositionStrategy

MIN_EVAL_BARS = 120  # sessions of trading needed after warm-up to compare at all


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / PERIODS_PER_YEAR
    if years <= 0 or equity.iloc[0] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def _stats(equity: pd.Series, capital: float) -> dict[str, float]:
    return {
        "total_return": float(equity.iloc[-1] / capital - 1),
        "cagr": _cagr(pd.concat([pd.Series([capital]), equity.reset_index(drop=True)])),
        "max_drawdown": max_drawdown(pd.concat([pd.Series([capital]), equity.reset_index(drop=True)])),
        "sharpe_ratio": sharpe_ratio(equity),
    }


@dataclass
class SymbolComparison:
    symbol: str
    start: pd.Timestamp
    end: pd.Timestamp
    stats: pd.DataFrame  # index = strategy key
    equity: dict[str, pd.Series]


def compare_symbol(
    df: pd.DataFrame,
    symbol: str,
    settings: ResearchSettings = ResearchSettings(),
    keys: list[str] | None = None,
    warmup: int = WARMUP_BARS,
) -> SymbolComparison:
    """Run every rule on ``df`` from session ``warmup`` onward."""
    keys = keys or list(STRATEGIES)
    if len(df) < warmup + MIN_EVAL_BARS:
        raise ValueError(f"{symbol}: {len(df)} sessions; needs {warmup + MIN_EVAL_BARS}")

    instrument = equity_instrument(symbol)
    bar_type = bar_type_for(symbol)
    bars = to_nautilus_bars(df, instrument, bar_type)
    quotes = open_quotes(df, instrument)
    close_ts = bar_close_times(df)
    close_ns = [int(t.value) for t in close_ts]
    start_ts = close_ts[warmup]
    half = close_ts[warmup + (len(df) - warmup) // 2]

    rows, curves = [], {}
    for key in keys:
        _, rule = STRATEGIES[key]
        target = rule(df)
        cfg = TargetPositionConfig(
            bar_type=bar_type,
            targets=dict(zip(close_ns, target.to_numpy().tolist())),
            position_fraction=settings.position_fraction,
            trade_start_ns=close_ns[warmup],
        )
        out = run_backtest(bars, instrument, cfg, starting_capital=settings.capital,
                           cost_rate=settings.cost_rate, strategy_cls=TargetPositionStrategy, quotes=quotes)
        if out.bars_seen != len(bars):
            raise RuntimeError(f"{symbol}/{key}: strategy saw {out.bars_seen} of {len(bars)} bars")
        eq = out.equity[out.equity.index >= start_ts]
        curves[key] = eq
        first, second = eq[eq.index < half], eq[eq.index >= half]
        rows.append({
            "strategy": key,
            **_stats(eq, settings.capital),
            "trades": len(out.positions),
            "time_in_market": float(target.iloc[warmup:].mean()),
            "sharpe_first_half": sharpe_ratio(first),
            "sharpe_second_half": sharpe_ratio(second),
        })
    return SymbolComparison(symbol, start_ts, close_ts[-1], pd.DataFrame(rows).set_index("strategy"), curves)


def summarise(results: list[SymbolComparison]) -> pd.DataFrame:
    """One row per strategy: averages and head-to-head counts across symbols."""
    keys = list(results[0].stats.index)
    table = []
    sharpe = pd.DataFrame({r.symbol: r.stats["sharpe_ratio"] for r in results})  # strategy x symbol
    ranks = sharpe.rank(ascending=False)
    for k in keys:
        per = pd.DataFrame([r.stats.loc[k] for r in results])
        bh = pd.DataFrame([r.stats.loc["buy_and_hold"] for r in results])
        table.append({
            "strategy": k,
            "avg_sharpe": float(per["sharpe_ratio"].mean()),
            "median_sharpe": float(per["sharpe_ratio"].median()),
            "avg_cagr": float(per["cagr"].mean()),
            "avg_max_drawdown": float(per["max_drawdown"].mean()),
            "avg_time_in_market": float(per["time_in_market"].mean()),
            "avg_trades": float(per["trades"].mean()),
            "avg_rank": float(ranks.loc[k].mean()),
            "beats_bh_sharpe": int((per["sharpe_ratio"].values > bh["sharpe_ratio"].values).sum()),
            "beats_bh_return": int((per["total_return"].values > bh["total_return"].values).sum()),
            "shallower_dd_than_bh": int((per["max_drawdown"].values > bh["max_drawdown"].values).sum()),
            "symbols": len(results),
        })
    return pd.DataFrame(table).set_index("strategy").sort_values("avg_rank")


def rank_stability(results: list[SymbolComparison]) -> float:
    """Average Spearman correlation of strategy Sharpe ranks, first half vs second half."""
    rhos = []
    for r in results:
        a, b = r.stats["sharpe_first_half"], r.stats["sharpe_second_half"]
        if a.notna().sum() > 2 and b.notna().sum() > 2:
            rho = a.rank().corr(b.rank())
            if not math.isnan(rho):
                rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


def equal_weight_portfolio(results: list[SymbolComparison], key: str, capital: float = 100_000.0) -> pd.Series:
    """Equal-weight, daily-rebalanced blend of one rule across symbols (average daily return)."""
    rets = pd.concat({r.symbol: r.equity[key].pct_change() for r in results}, axis=1)
    mean = rets.mean(axis=1, skipna=True).fillna(0.0)
    return capital * (1 + mean).cumprod()
