"""A strategy that trades a pre-computed schedule of long holding periods.

Used for the random-entry baseline: random schedules with the same number of
trades and the same holding lengths as the real strategy, executed through the
same Nautilus engine, fee model and sizing, so the only difference is timing.
"""

from __future__ import annotations

from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import OrderSide
from nautilus_trader.trading import Strategy
from nautilus_trader.trading import StrategyConfig


class ScheduledLongConfig(StrategyConfig):
    def __init__(
        self,
        bar_type: BarType,
        entry_bars: set[int],
        exit_bars: set[int],
        position_fraction: float = 0.99,
        **kwargs: object,
    ) -> None:
        if not 0 < position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        self.bar_type = bar_type
        self.entry_bars = set(entry_bars)  # bar indices (0-based) to enter on
        self.exit_bars = set(exit_bars)  # bar indices to exit on
        self.position_fraction = position_fraction
        self.close_on_stop = True


class ScheduledLong(Strategy):
    """Go long at scheduled bar indices, exit at scheduled bar indices."""

    def __init__(self, config: ScheduledLongConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self.bars_seen = 0
        self.signals: list = []

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.bar_type.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        i = self.bars_seen
        self.bars_seen += 1
        instrument_id = bar.bar_type.instrument_id
        if i in self.config.exit_bars and self.portfolio.is_net_long(instrument_id):
            self.close_all_positions(instrument_id)
        if i in self.config.entry_bars and self.portfolio.is_net_flat(instrument_id):
            account = self.portfolio.account(instrument_id.venue)
            free = account.balance_free(self.instrument.quote_currency).as_double()
            qty = self.instrument.make_qty(self.config.position_fraction * free / bar.close.as_double())
            if qty.as_double() > 0:
                self.submit_order(
                    self.order_factory.market(
                        instrument_id=instrument_id, order_side=OrderSide.BUY, quantity=qty,
                    ),
                )

    def on_stop(self) -> None:
        self.close_all_positions(self.config.bar_type.instrument_id)
