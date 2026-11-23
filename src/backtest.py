"""Thin wrapper around the NautilusTrader v2 low-level ``BacktestEngine``.

Builds a simulated US-equity venue (USD cash account, cost per side as a fee),
runs :class:`~src.strategies.sma_crossover.SmaCrossover` over a list of bars,
and returns plain pandas objects for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from decimal import Decimal

import pandas as pd
from nautilus_trader.backtest import BacktestEngine
from nautilus_trader.backtest import BacktestEngineConfig
from nautilus_trader.common import LoggerConfig
from nautilus_trader.common import LogLevel
from nautilus_trader.model import AccountType
from nautilus_trader.model import Bar
from nautilus_trader.model import Equity
from nautilus_trader.model import Money
from nautilus_trader.model import QuoteTick
from nautilus_trader.model import OmsType
from nautilus_trader.model import TraderId
from nautilus_trader.trading import StrategyConfig

from src.data_loader import DEFAULT_COST_RATE
from src.data_loader import fee_model
from src.strategies.sma_crossover import SmaCrossover
from src.strategies.sma_crossover import SmaCrossoverConfig


@dataclass
class BacktestOutput:
    """Results of a single backtest run, as plain pandas objects."""

    equity: pd.Series  # mark-to-market equity in quote currency, one point per bar
    fills: pd.DataFrame  # order fills report
    positions: pd.DataFrame  # positions report (closed round trips)
    starting_capital: float
    signals: list = field(default_factory=list)
    bars_seen: int = 0
    # A decision made on the final close that would execute at the next open, if any.
    pending: tuple | None = None


def _money_to_float(value: object) -> float:
    """Parse Nautilus report values such as ``"-344.60 USD"``."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).split()[0].replace(",", "").replace("_", ""))


def run_backtest(
    bars: list[Bar],
    instrument: Equity,
    config: SmaCrossoverConfig | StrategyConfig,
    starting_capital: float = 100_000.0,
    log_level: LogLevel = LogLevel.WARNING,
    cost_rate: float | Decimal = DEFAULT_COST_RATE,
    strategy_cls: type = SmaCrossover,
    quotes: list[QuoteTick] | None = None,
) -> BacktestOutput:
    """Run a strategy (SMA crossover by default) over ``bars`` and collect results.

    ``cost_rate`` is charged per side on every fill (commission + slippage).
    Pass ``quotes`` (e.g. :func:`src.data_loader.open_quotes`) for strategies
    that execute at the next open.
    """
    if not bars:
        raise ValueError("No bars supplied")

    engine = BacktestEngine(
        BacktestEngineConfig(
            trader_id=TraderId("BACKTESTER-001"),
            logging=LoggerConfig(stdout_level=log_level),
        ),
    )
    venue = instrument.id.venue
    usd = instrument.quote_currency
    try:
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,  # no margin, no leverage
            starting_balances=[Money(starting_capital, usd)],
            base_currency=usd,
            fee_model=fee_model(cost_rate),
            bar_execution=True,
        )
        engine.add_instrument(instrument)
        engine.add_data(bars)
        if quotes:
            engine.add_data(quotes)
        strategy = strategy_cls(config)
        engine.add_strategy(strategy)
        engine.run()

        fills = engine.generate_order_fills_report()
        positions = engine.generate_positions_report()
        equity = equity_from_fills(engine.generate_fills_report(), bars, starting_capital)
        return BacktestOutput(
            equity=equity,
            fills=fills,
            positions=positions,
            starting_capital=starting_capital,
            signals=list(strategy.signals),
            bars_seen=strategy.bars_seen,
            pending=getattr(strategy, "_awaiting_open", None),
        )
    finally:
        engine.dispose()


def equity_from_fills(fill_events: pd.DataFrame, bars: list[Bar], starting_capital: float) -> pd.Series:
    """Mark-to-market equity at every bar close: cash + shares held x close.

    Rebuilt from individual fill events (price, quantity, commission), so it is
    independent of when the engine happens to snapshot the portfolio.
    """
    index = pd.to_datetime([b.ts_event for b in bars], utc=True)
    closes = pd.Series([b.close.as_double() for b in bars], index=index)
    cash = pd.Series(0.0, index=index)
    shares = pd.Series(0.0, index=index)
    if not fill_events.empty:
        f = fill_events.sort_values("ts_event")
        sign = f["order_side"].astype(str).str.upper().map({"BUY": 1.0, "SELL": -1.0})
        qty = f["last_qty"].map(_money_to_float) * sign
        px = f["last_px"].map(_money_to_float)
        fee = f["commission"].map(_money_to_float)
        ts = pd.to_datetime(f["ts_event"], utc=True)
        flows = pd.DataFrame({"dq": qty.values, "dc": (-qty * px - fee).values})
        # A fill counts toward the first bar close at or after it.
        pos = index.searchsorted(pd.DatetimeIndex(ts), side="left")
        flows = flows[pos < len(index)].assign(i=pos[pos < len(index)])
        by_bar = flows.groupby("i")[["dq", "dc"]].sum()
        shares.iloc[by_bar.index] = by_bar["dq"].values
        cash.iloc[by_bar.index] = by_bar["dc"].values
    held = shares.cumsum()
    equity = starting_capital + cash.cumsum() + held * closes
    return equity.rename("equity")


def realized_pnls(positions: pd.DataFrame) -> list[float]:
    """Realised PnL (net of commissions) of each closed position."""
    if positions.empty or "realized_pnl" not in positions:
        return []
    return [_money_to_float(v) for v in positions["realized_pnl"]]
