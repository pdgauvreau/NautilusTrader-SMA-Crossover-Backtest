"""Evaluate day-trading rules: Nautilus runs, a fast trade-list evaluator, luck and cost tests.

* :func:`run_rule` is the authoritative result: the rule's per-minute targets
  are traded by :class:`TargetPositionStrategy` in Nautilus, filling at the
  next minute's open quote (bid/ask = open -/+ ``half_spread``) plus
  ``cost_rate`` per side.
* :func:`trade_returns` reproduces those fills from the open prices directly.
  It is verified against Nautilus in the tests, and used where hundreds of runs
  are needed: the random-timing control and the cost stress test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import equity_instrument
from src.intraday.data import bar_end_ns
from src.intraday.data import minute_bar_type
from src.intraday.data import quotes_at
from src.intraday.data import to_nautilus_minute_bars
from src.intraday.rules import FLAT_TAIL
from src.intraday.rules import INTRADAY_RULES
from src.intraday.rules import _sessions
from src.metrics import max_drawdown
from src.metrics import sharpe_ratio
from src.research import random_schedule
from src.strategies.target_position import TargetPositionConfig
from src.strategies.target_position import TargetPositionStrategy

FRACTION = 0.99
DEFAULT_HALF_SPREAD = 0.00005  # 0.5 bp: liquid ETFs/large caps quote about a cent wide
DEFAULT_COST = 0.00005  # 0.5 bp per side: commission + slippage
COST_GRID_BP = (0.0, 0.5, 1.0, 2.0, 5.0)  # all-in cost per side (half-spread + fees), basis points


def _trades(target: np.ndarray) -> list[tuple[int, int]]:
    """(entry decision bar, exit decision bar) pairs; exit is -1 if still long at the end."""
    t = np.asarray(target, dtype=bool)
    entries = np.flatnonzero(t & ~np.r_[False, t[:-1]])
    exits = np.flatnonzero(~t & np.r_[False, t[:-1]])
    pairs, k = [], 0
    for e in entries:
        while k < len(exits) and exits[k] <= e:
            k += 1
        pairs.append((int(e), int(exits[k]) if k < len(exits) else -1))
    return pairs


def trade_returns(df: pd.DataFrame, target: np.ndarray, half_spread: float, cost: float) -> pd.DataFrame:
    """Per-trade results with the same fills as the Nautilus run.

    Entry fills at the next bar's open ask, exit at the next bar's open bid;
    a position still open at the end is closed at the last close.
    """
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    sess = df["session"].to_numpy()
    rows = []
    for e, x in _trades(target):
        if e + 1 >= len(o):
            continue
        entry = o[e + 1] * (1 + half_spread)
        exit_px = o[x + 1] * (1 - half_spread) if 0 <= x < len(o) - 1 else c[-1]
        r = (exit_px * (1 - cost) - entry * (1 + cost)) / entry  # per $ invested, fees both sides
        rows.append({"entry_bar": e + 1, "session": sess[e + 1], "net_return": r, "bars": (x if x >= 0 else len(o)) - e})
    return pd.DataFrame(rows, columns=["entry_bar", "session", "net_return", "bars"])


def daily_returns_from_trades(trades: pd.DataFrame, sessions: np.ndarray) -> pd.Series:
    """Account return per session (trades compounded at ``FRACTION`` of equity)."""
    days = pd.Index(pd.unique(sessions))
    if trades.empty:
        return pd.Series(0.0, index=days)
    growth = (1 + FRACTION * trades["net_return"]).groupby(trades["session"]).prod()
    return (growth.reindex(days).fillna(1.0) - 1).rename("ret")


def _equity(daily: pd.Series, capital: float) -> pd.Series:
    return capital * (1 + daily).cumprod()


@dataclass
class RuleResult:
    key: str
    daily_equity: pd.Series  # end-of-session equity (Nautilus)
    stats: dict
    yearly: dict[int, float]
    target: np.ndarray


def run_rule(df: pd.DataFrame, symbol: str, key: str, capital: float = 100_000.0,
             half_spread: float = DEFAULT_HALF_SPREAD, cost: float = DEFAULT_COST,
             bars: list | None = None) -> RuleResult:
    """Trade one rule through Nautilus on minute bars."""
    instrument = equity_instrument(symbol)
    bar_type = minute_bar_type(symbol)
    bars = bars if bars is not None else to_nautilus_minute_bars(df, instrument, bar_type)
    target = INTRADAY_RULES[key][1](df)
    change = np.flatnonzero(np.r_[target[0], target[1:] != target[:-1]])
    quotes = quotes_at(df, [i + 1 for i in change if i + 1 < len(df)], instrument, half_spread)
    cfg = TargetPositionConfig(bar_type=bar_type, targets=dict(zip(bar_end_ns(df), target.tolist())),
                               position_fraction=FRACTION)
    out = run_backtest(bars, instrument, cfg, starting_capital=capital, cost_rate=cost,
                       strategy_cls=TargetPositionStrategy, quotes=quotes)
    if out.bars_seen != len(bars):
        raise RuntimeError(f"{symbol}/{key}: strategy saw {out.bars_seen} of {len(bars)} bars")
    sess = pd.Series(df["session"].to_numpy(), index=out.equity.index)
    daily = out.equity.groupby(sess).last()
    daily.index = pd.to_datetime(daily.index)
    rets = daily.pct_change()
    rets.iloc[0] = daily.iloc[0] / capital - 1
    trades = trade_returns(df, target, half_spread, cost)
    years = len(daily) / 252
    yearly = {int(y): float((1 + g).prod() - 1) for y, g in rets.groupby(rets.index.year)}
    stats = {
        "total_return": float(daily.iloc[-1] / capital - 1),
        "cagr": float((daily.iloc[-1] / capital) ** (1 / years) - 1) if years > 0 else float("nan"),
        "sharpe_ratio": sharpe_ratio(pd.concat([pd.Series([capital]), daily.reset_index(drop=True)])),
        "max_drawdown": max_drawdown(pd.concat([pd.Series([capital]), daily.reset_index(drop=True)])),
        "trades": len(trades),
        "trades_per_year": len(trades) / years if years > 0 else float("nan"),
        "avg_trade_bps": float(trades["net_return"].mean() * 1e4) if len(trades) else float("nan"),
        "win_rate": float((trades["net_return"] > 0).mean()) if len(trades) else float("nan"),
        "time_in_market": float(np.mean(target)),
    }
    return RuleResult(key, daily, stats, yearly, target)


# ----------------------------------------------------------------------- luck test
def random_control(df: pd.DataFrame, target: np.ndarray, n_samples: int = 200, seed: int = 7,
                   half_spread: float = DEFAULT_HALF_SPREAD, cost: float = DEFAULT_COST) -> pd.DataFrame:
    """Random trades with the rule's per-day structure, placed on random days at random minutes.

    Each sample reassigns the rule's per-session lists of holding lengths to
    randomly chosen sessions, then places those trades at random,
    non-overlapping times within each session (flat by the close). Same number
    of trades, same holding times, same costs: only *when* differs.
    """
    rng = np.random.default_rng(seed)
    sess_rows = _sessions(df)
    by_day: dict[int, list[int]] = {}
    for e, x in _trades(target):
        k = int(np.searchsorted([r[0] for r in sess_rows], e, side="right") - 1)
        if x < 0:
            continue
        by_day.setdefault(k, []).append(x - e)
    day_lists = list(by_day.values())
    sessions = df["session"].to_numpy()
    out = []
    for _ in range(n_samples):
        days = rng.choice(len(sess_rows), size=len(day_lists), replace=False)
        rand = np.zeros(len(df), dtype=bool)
        for holds, k in zip(day_lists, days):
            rows = sess_rows[k]
            usable = len(rows) - FLAT_TAIL + 1  # exit decision must be before the last bar
            holds = [h for h in holds if h < usable - 1]
            if not holds or sum(holds) + len(holds) >= usable:
                continue
            entries, exits = random_schedule(usable, holds, rng)
            for e, x in zip(sorted(entries), sorted(exits)):
                rand[rows[e]:rows[x]] = True
        tr = trade_returns(df, rand, half_spread, cost)
        eq = _equity(daily_returns_from_trades(tr, sessions), 100_000.0)
        out.append({"total_return": float(eq.iloc[-1] / 100_000 - 1),
                    "sharpe_ratio": sharpe_ratio(pd.concat([pd.Series([100_000.0]), eq.reset_index(drop=True)]))})
    return pd.DataFrame(out)


def cost_stress(df: pd.DataFrame, target: np.ndarray, grid_bp=COST_GRID_BP) -> pd.DataFrame:
    """Total return and Sharpe at different all-in costs per side (half-spread + fees)."""
    sessions = df["session"].to_numpy()
    rows = []
    for bp in grid_bp:
        tr = trade_returns(df, target, half_spread=bp / 2 / 1e4, cost=bp / 2 / 1e4)
        eq = _equity(daily_returns_from_trades(tr, sessions), 100_000.0)
        rows.append({"cost_bp_per_side": bp, "total_return": float(eq.iloc[-1] / 100_000 - 1),
                     "sharpe_ratio": sharpe_ratio(pd.concat([pd.Series([100_000.0]), eq.reset_index(drop=True)])),
                     "avg_trade_bps": float(tr["net_return"].mean() * 1e4) if len(tr) else float("nan")})
    return pd.DataFrame(rows)


def breakeven_cost_bp(df: pd.DataFrame, target: np.ndarray) -> float:
    """All-in cost per side (bp) at which the average trade nets zero."""
    tr = trade_returns(df, target, 0.0, 0.0)
    if tr.empty:
        return float("nan")
    return float(tr["net_return"].mean() / 2 * 1e4)  # cost is paid twice per round trip
