"""Trade out-of-sample model probabilities through Nautilus and compare with baselines.

Each comparison uses the same out-of-sample span, the same costs and the same
sizing for every approach: ML strategy, SMA 10/50, and buy & hold.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.backtest import realized_pnls
from src.backtest import run_backtest
from src.data_loader import bar_type_for
from src.data_loader import binance_spot_instrument
from src.data_loader import to_nautilus_bars
from src.metrics import buy_and_hold_equity
from src.metrics import compute_stats
from src.pipeline import run_pipeline
from src.research import ResearchSettings
from src.strategies.ml_signal import MLSignalConfig
from src.strategies.ml_signal import MLSignalStrategy

THRESHOLD_GRID: tuple[tuple[float, float], ...] = ((0.50, 0.50), (0.52, 0.48), (0.55, 0.45), (0.60, 0.40))


@dataclass
class StrategyRun:
    equity: pd.Series
    stats: dict
    num_signals: int


def _oos_slice(df: pd.DataFrame, probs: pd.Series) -> pd.DataFrame:
    """Rows of ``df`` whose bar close falls within the prediction span."""
    close_ts = df.index + pd.Timedelta(df.index[1] - df.index[0])
    mask = (close_ts >= probs.index.min()) & (close_ts <= probs.index.max())
    return df[mask]


def backtest_probabilities(
    df: pd.DataFrame,
    symbol: str,
    probs: pd.Series,
    settings: ResearchSettings = ResearchSettings(),
    enter: float = 0.52,
    exit_: float = 0.48,
) -> StrategyRun:
    """Run :class:`MLSignalStrategy` over the bars covered by ``probs`` (index = close time)."""
    window = _oos_slice(df, probs)
    instrument = binance_spot_instrument(symbol)
    bar_type = bar_type_for(symbol)
    bars = to_nautilus_bars(window, instrument, bar_type)
    cfg = MLSignalConfig(
        bar_type=bar_type,
        probabilities={int(ts.value): float(p) for ts, p in probs.items()},
        enter_threshold=enter,
        exit_threshold=exit_,
        position_fraction=settings.position_fraction,
    )
    out = run_backtest(bars, instrument, cfg, starting_capital=settings.capital,
                       cost_rate=settings.cost_rate, strategy_cls=MLSignalStrategy)
    if out.bars_seen != len(bars):
        raise RuntimeError(f"Strategy saw {out.bars_seen} of {len(bars)} bars")
    index = pd.to_datetime([b.ts_event for b in bars], utc=True)
    equity = out.equity.reindex(index).ffill().fillna(settings.capital)
    stats = compute_stats(equity, realized_pnls(out.positions), settings.capital).as_dict()
    return StrategyRun(equity, stats, len(out.signals))


@dataclass
class Comparison:
    ml: StrategyRun
    sma: StrategyRun
    buy_and_hold: StrategyRun
    thresholds: pd.DataFrame  # threshold sensitivity for the ML strategy


def compare_on_oos(
    df: pd.DataFrame,
    symbol: str,
    probs: pd.Series,
    settings: ResearchSettings = ResearchSettings(),
    enter: float = 0.52,
    exit_: float = 0.48,
) -> Comparison:
    ml = backtest_probabilities(df, symbol, probs, settings, enter, exit_)

    # SMA 10/50 over the same span: warm up on earlier bars, trade from the first OOS bar.
    start = ml.equity.index[0]
    end_open = ml.equity.index[-1] - pd.Timedelta(df.index[1] - df.index[0])
    sma_res = run_pipeline(
        df.loc[:end_open], 10, 50, capital=settings.capital, symbol=symbol,
        position_fraction=settings.position_fraction, cost_rate=settings.cost_rate,
        trade_start=start, enter_if_above_on_start=True,
    )
    sma = StrategyRun(sma_res.equity, sma_res.strategy_stats.as_dict(), len(sma_res.output.signals))

    closes = sma_res.closes
    bnh_eq = buy_and_hold_equity(closes, settings.capital, settings.cost_rate)
    bnh = StrategyRun(
        bnh_eq, compute_stats(bnh_eq, [float(bnh_eq.iloc[-1]) - settings.capital], settings.capital).as_dict(), 1,
    )

    rows = []
    for en, ex in THRESHOLD_GRID:
        run = ml if (en, ex) == (enter, exit_) else backtest_probabilities(df, symbol, probs, settings, en, ex)
        rows.append({"enter": en, "exit": ex, **{k: run.stats[k] for k in
                     ("total_return", "sharpe_ratio", "max_drawdown", "num_trades")},
                     "time_in_market": _time_in_market(run.equity)})
    return Comparison(ml, sma, bnh, pd.DataFrame(rows))


def _time_in_market(equity: pd.Series) -> float:
    """Share of days on which equity changed (a proxy for holding a position)."""
    return float((equity.diff().abs() > 1e-9).mean())
