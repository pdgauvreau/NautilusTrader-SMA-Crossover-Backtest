"""Trade any long/flat target signal through Nautilus with identical execution.

Every rule in :mod:`src.strategies.signals` produces "long or cash after this
close". This strategy turns those targets into orders the same way for every
rule: decide on the daily close, execute at the next session's open quote
(buys pay the ask, sells hit the bid), invest a fixed share of cash rounded
down to whole shares, and pay the venue's per-side cost. The only thing that
differs between compared strategies is the signal.
"""

from __future__ import annotations

from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import OrderSide
from nautilus_trader.trading import Strategy
from nautilus_trader.trading import StrategyConfig

from src.strategies.sma_crossover import Signal
from src.strategies.sma_crossover import affordable_qty


class TargetPositionConfig(StrategyConfig):
    def __init__(
        self,
        bar_type: BarType,
        targets: dict[int, bool],
        position_fraction: float = 0.99,
        trade_start_ns: int | None = None,
        close_on_stop: bool = True,
        **kwargs: object,
    ) -> None:
        if not 0 < position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        self.bar_type = bar_type
        self.targets = dict(targets)  # bar close ts (ns) -> want to be long after this close
        self.position_fraction = position_fraction
        self.trade_start_ns = trade_start_ns
        self.close_on_stop = close_on_stop


class TargetPositionStrategy(Strategy):
    def __init__(self, config: TargetPositionConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self.bars_seen = 0
        self.signals: list[tuple[int, Signal]] = []
        self._awaiting_open: tuple[Signal, int] | None = None

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.bar_type.instrument_id)
        self.subscribe_bars(self.config.bar_type)
        self.subscribe_quotes(self.config.bar_type.instrument_id)

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        self._awaiting_open = None
        start = self.config.trade_start_ns
        if start is not None and bar.ts_event < start:
            return
        want_long = self.config.targets.get(bar.ts_event)
        if want_long is None:
            return
        iid = bar.bar_type.instrument_id
        if want_long and self.portfolio.is_net_flat(iid):
            self._awaiting_open = (Signal.ENTER_LONG, bar.ts_event)
        elif not want_long and self.portfolio.is_net_long(iid):
            self._awaiting_open = (Signal.EXIT_LONG, bar.ts_event)

    def on_quote(self, quote) -> None:
        if self._awaiting_open is None:
            return
        signal, decision_ts = self._awaiting_open
        self._awaiting_open = None
        iid = self.config.bar_type.instrument_id
        if signal is Signal.ENTER_LONG and self.portfolio.is_net_flat(iid):
            account = self.portfolio.account(iid.venue)
            free = account.balance_free(self.instrument.quote_currency).as_double()
            qty = affordable_qty(self.instrument, self.config.position_fraction * free, quote.ask_price.as_double())
            if qty.as_double() > 0:
                self.submit_order(
                    self.order_factory.market(instrument_id=iid, order_side=OrderSide.BUY, quantity=qty),
                )
                self.signals.append((decision_ts, signal))
        elif signal is Signal.EXIT_LONG and self.portfolio.is_net_long(iid):
            self.close_all_positions(iid)
            self.signals.append((decision_ts, signal))

    def on_stop(self) -> None:
        if self.config.close_on_stop:
            self.close_all_positions(self.config.bar_type.instrument_id)
