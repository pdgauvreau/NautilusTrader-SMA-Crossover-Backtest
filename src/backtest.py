"""Thin wrapper around the NautilusTrader v2 low-level ``BacktestEngine``.

Builds a simulated Binance spot venue (cash account, 0.1% maker/taker fees),
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
from nautilus_trader.model import CurrencyPair
from nautilus_trader.model import Money
from nautilus_trader.model import OmsType
from nautilus_trader.model import TraderId

from src.data_loader import TAKER_FEE
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


def _money_to_float(value: object) -> float:
    """Parse Nautilus report values such as ``"-344.60347000 USDT"``."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).split()[0].replace(",", "").replace("_", ""))


def run_backtest(
    bars: list[Bar],
    instrument: CurrencyPair,
    config: SmaCrossoverConfig,
    starting_capital: float = 100_000.0,
    log_level: LogLevel = LogLevel.WARNING,
    cost_rate: float | Decimal = TAKER_FEE,
) -> BacktestOutput:
    """Run the SMA crossover strategy over ``bars`` and collect results.

    ``cost_rate`` is charged per side on every fill: the Binance taker fee by
    default, or fee + an estimated slippage for more conservative runs.
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
    quote = instrument.quote_currency
    base = instrument.base_currency
    try:
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.CASH,
            # Multi-currency cash account: spot venues require base_currency=None.
            starting_balances=[Money(starting_capital, quote), Money(0, base)],
            base_currency=None,
            fee_model=fee_model(cost_rate),
            # Market orders fill at the close of the bar that generated the signal.
            bar_execution=True,
        )
        engine.add_instrument(instrument)
        engine.add_data(bars)
        strategy = SmaCrossover(config)
        engine.add_strategy(strategy)
        engine.run()

        fills = engine.generate_order_fills_report()
        positions = engine.generate_positions_report()
        equity = _equity_curve(engine, bars, venue, quote.code, base.code)
        return BacktestOutput(
            equity=equity,
            fills=fills,
            positions=positions,
            starting_capital=starting_capital,
            signals=list(strategy.signals),
            bars_seen=strategy.bars_seen,
        )
    finally:
        engine.dispose()


def _equity_curve(
    engine: BacktestEngine,
    bars: list[Bar],
    venue: object,
    quote_code: str,
    base_code: str,
) -> pd.Series:
    """Mark-to-market equity (quote + base * close) from per-bar portfolio snapshots."""
    closes = {bar.ts_event: bar.close.as_double() for bar in bars}
    account = engine.portfolio.account(venue)
    points: list[tuple[pd.Timestamp, float]] = []
    for snap in engine.portfolio.snapshots(account.id):
        px = closes.get(snap.ts_event)
        if px is None:
            continue
        balances = {m.currency.code: m.as_double() for m in snap.total_equity}
        value = balances.get(quote_code, 0.0) + balances.get(base_code, 0.0) * px
        points.append((pd.Timestamp(snap.ts_event, tz="UTC"), value))
    equity = pd.Series(dict(points), name="equity", dtype=float)
    # Snapshots can repeat at a timestamp (e.g. start, or stop-time close); keep the last.
    return equity.groupby(level=0).last().sort_index()


def realized_pnls(positions: pd.DataFrame) -> list[float]:
    """Realised PnL (net of commissions) of each closed position."""
    if positions.empty or "realized_pnl" not in positions:
        return []
    return [_money_to_float(v) for v in positions["realized_pnl"]]
