"""End-to-end backtest pipeline shared by the CLI runner and the Streamlit app.

Takes a validated OHLCV DataFrame plus strategy parameters, runs the Nautilus
backtest, and returns everything needed for reporting: aligned equity curves
for the strategy and buy-and-hold, statistics, and a trades table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from nautilus_trader.model import BarType

from src.backtest import BacktestOutput
from src.backtest import realized_pnls
from src.backtest import run_backtest
from src.data_loader import DEFAULT_BAR_TYPE
from src.data_loader import TAKER_FEE
from src.data_loader import btcusdt_instrument
from src.data_loader import to_nautilus_bars
from src.metrics import PerformanceStats
from src.metrics import buy_and_hold_equity
from src.metrics import compute_stats
from src.strategies.sma_crossover import SmaCrossoverConfig

TRADE_COLUMNS = ["opened", "closed", "entry_px", "exit_px", "quantity", "net_pnl_usdt", "return"]


@dataclass
class PipelineResult:
    closes: pd.Series  # close prices indexed by bar close time (UTC)
    equity: pd.Series  # strategy mark-to-market equity, aligned to ``closes``
    buy_and_hold: pd.Series  # buy-and-hold equity, aligned to ``closes``
    strategy_stats: PerformanceStats
    buy_and_hold_stats: PerformanceStats
    trades: pd.DataFrame  # one row per closed round trip
    output: BacktestOutput  # raw engine output (fills, positions, signals)


def trades_table(out: BacktestOutput) -> pd.DataFrame:
    """Closed round trips as a tidy DataFrame."""
    pos = out.positions
    if pos.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame(
        {
            "opened": pd.to_datetime(pos["ts_opened"], utc=True).dt.date,
            "closed": pd.to_datetime(pos["ts_closed"], utc=True).dt.date,
            "entry_px": pos["avg_px_open"].astype(float).round(2),
            "exit_px": pos["avg_px_close"].astype(float).round(2),
            "quantity": pos["peak_qty"].astype(str).str.split().str[0].astype(float),
            "net_pnl_usdt": [round(p, 2) for p in realized_pnls(pos)],
            "return": pos["realized_return"].astype(float).round(4),
        },
    ).sort_values("opened").reset_index(drop=True)


def run_pipeline(
    df: pd.DataFrame,
    fast: int = 10,
    slow: int = 50,
    trade_size: float = 0.5,
    capital: float = 100_000.0,
) -> PipelineResult:
    """Run the SMA crossover backtest on ``df`` and compute all report data."""
    instrument = btcusdt_instrument()
    bar_type = BarType.from_str(DEFAULT_BAR_TYPE)
    bars = to_nautilus_bars(df, instrument, bar_type)
    config = SmaCrossoverConfig(
        bar_type=bar_type,
        trade_size=trade_size,
        fast_period=fast,
        slow_period=slow,
    )

    out = run_backtest(bars, instrument, config, starting_capital=capital)
    if out.bars_seen != len(bars):
        # Nautilus logs handler exceptions instead of raising; fail loudly here.
        raise RuntimeError(f"Strategy saw {out.bars_seen} of {len(bars)} bars")

    closes = pd.Series(
        [b.close.as_double() for b in bars],
        index=pd.to_datetime([b.ts_event for b in bars], utc=True),
        name="close",
    )
    equity = out.equity.reindex(closes.index).ffill().fillna(capital)
    bnh = buy_and_hold_equity(closes, capital, float(TAKER_FEE))

    strat_stats = compute_stats(equity, realized_pnls(out.positions), capital)
    bnh_stats = compute_stats(bnh, [float(bnh.iloc[-1]) - capital], capital)

    return PipelineResult(
        closes=closes,
        equity=equity,
        buy_and_hold=bnh,
        strategy_stats=strat_stats,
        buy_and_hold_stats=bnh_stats,
        trades=trades_table(out),
        output=out,
    )
