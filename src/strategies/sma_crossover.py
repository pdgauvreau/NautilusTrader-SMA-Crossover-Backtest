"""Long-only SMA crossover strategy for NautilusTrader v2.

Rule
----
* Go long a fixed quantity when the fast SMA crosses **above** the slow SMA.
* Exit to flat when the fast SMA crosses back **below** the slow SMA.
* Never short. No signals until both SMAs are warmed up.

The crossover decision is factored out into :func:`crossover_signal`, a pure
function, so it can be unit-tested without spinning up an engine.
"""

from __future__ import annotations

from enum import Enum

from nautilus_trader.indicators import SimpleMovingAverage
from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import OrderSide
from nautilus_trader.trading import Strategy
from nautilus_trader.trading import StrategyConfig


class Signal(Enum):
    """Trading signal emitted by the crossover rule."""

    ENTER_LONG = "enter_long"
    EXIT_LONG = "exit_long"


def crossover_signal(prev_diff: float | None, diff: float) -> Signal | None:
    """Return the signal implied by a change in ``fast - slow``.

    A cross *above* happens when the spread moves from ``<= 0`` to ``> 0``;
    a cross *below* when it moves from ``>= 0`` to ``< 0``. With no previous
    value (first warmed-up bar) there is no cross by definition.
    """
    if prev_diff is None:
        return None
    if prev_diff <= 0 < diff:
        return Signal.ENTER_LONG
    if prev_diff >= 0 > diff:
        return Signal.EXIT_LONG
    return None


class SmaCrossoverConfig(StrategyConfig):
    """Parameters for :class:`SmaCrossover`.

    Note: in Nautilus 2.0 ``StrategyConfig`` is a Rust (pyo3) class whose
    ``__new__`` ignores unknown keyword arguments, so custom fields are set
    explicitly in ``__init__`` rather than declared as class annotations.
    """

    def __init__(
        self,
        bar_type: BarType,
        trade_size: float = 0.5,
        fast_period: int = 10,
        slow_period: int = 50,
        close_on_stop: bool = True,
        **kwargs: object,
    ) -> None:
        if fast_period <= 0 or slow_period <= 0:
            raise ValueError("SMA periods must be positive")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        if trade_size <= 0:
            raise ValueError("trade_size must be positive")
        self.bar_type = bar_type
        self.trade_size = trade_size
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.close_on_stop = close_on_stop


class SmaCrossover(Strategy):
    """Long-only fast/slow simple-moving-average crossover."""

    def __init__(self, config: SmaCrossoverConfig) -> None:
        super().__init__(config)
        self.fast = SimpleMovingAverage(config.fast_period)
        self.slow = SimpleMovingAverage(config.slow_period)
        self.instrument = None
        self._prev_diff: float | None = None
        self.bars_seen = 0
        self.signals: list[tuple[int, Signal]] = []

    def on_start(self) -> None:
        bar_type = self.config.bar_type
        self.instrument = self.cache.instrument(bar_type.instrument_id)
        if self.instrument is None:
            raise RuntimeError(f"Instrument {bar_type.instrument_id} not in cache")
        # Indicators registered for bars are updated before on_bar is called.
        self.register_indicator_for_bars(bar_type, self.fast)
        self.register_indicator_for_bars(bar_type, self.slow)
        self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        if not (self.fast.initialized and self.slow.initialized):
            return  # still warming up (needs slow_period bars)

        diff = self.fast.value - self.slow.value
        signal = crossover_signal(self._prev_diff, diff)
        self._prev_diff = diff
        if signal is None:
            return

        instrument_id = bar.bar_type.instrument_id
        if signal is Signal.ENTER_LONG and self.portfolio.is_net_flat(instrument_id):
            order = self.order_factory.market(
                instrument_id=instrument_id,
                order_side=OrderSide.BUY,
                quantity=self.instrument.make_qty(self.config.trade_size),
            )
            self.submit_order(order)
            self.signals.append((bar.ts_event, signal))
        elif signal is Signal.EXIT_LONG and self.portfolio.is_net_long(instrument_id):
            self.close_all_positions(instrument_id)
            self.signals.append((bar.ts_event, signal))

    def on_stop(self) -> None:
        instrument_id = self.config.bar_type.instrument_id
        self.cancel_all_orders(instrument_id)
        if self.config.close_on_stop:
            # Mark the backtest to flat so every trade is realised in the stats.
            self.close_all_positions(instrument_id)
