"""Long-only strategy driven by model probabilities.

Each bar looks up ``P(next horizon return > 0)`` for its close timestamp, then:

* enters long (fraction of cash) when ``prob >= enter_threshold`` and flat,
* exits when ``prob <= exit_threshold`` and long.

The gap between the two thresholds is hysteresis, to stop small wobbles around
0.5 from churning fees.

In this backtest the probabilities are pre-computed out-of-sample by
:mod:`src.ml.walk_forward`, so every one comes from a model trained only on
earlier data. A live version would compute features from its own bar history
and call the model in ``on_bar``. The decision logic here would not change.
"""

from __future__ import annotations

from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import OrderSide
from nautilus_trader.trading import Strategy
from nautilus_trader.trading import StrategyConfig

from src.strategies.sma_crossover import Signal


class MLSignalConfig(StrategyConfig):
    def __init__(
        self,
        bar_type: BarType,
        probabilities: dict[int, float],
        enter_threshold: float = 0.52,
        exit_threshold: float = 0.48,
        position_fraction: float = 0.99,
        **kwargs: object,
    ) -> None:
        if not 0 <= exit_threshold <= enter_threshold <= 1:
            raise ValueError("need 0 <= exit_threshold <= enter_threshold <= 1")
        if not 0 < position_fraction <= 1:
            raise ValueError("position_fraction must be in (0, 1]")
        self.bar_type = bar_type
        self.probabilities = dict(probabilities)  # bar close ts (ns) -> probability
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.position_fraction = position_fraction
        self.close_on_stop = True


def ml_decision(prob: float | None, is_long: bool, enter: float, exit_: float) -> Signal | None:
    """Pure decision rule (unit-tested separately from the engine)."""
    if prob is None:
        return None
    if not is_long and prob >= enter:
        return Signal.ENTER_LONG
    if is_long and prob <= exit_:
        return Signal.EXIT_LONG
    return None


class MLSignalStrategy(Strategy):
    def __init__(self, config: MLSignalConfig) -> None:
        super().__init__(config)
        self.instrument = None
        self.bars_seen = 0
        self.signals: list[tuple[int, Signal]] = []

    def on_start(self) -> None:
        self.instrument = self.cache.instrument(self.config.bar_type.instrument_id)
        self.subscribe_bars(self.config.bar_type)

    def on_bar(self, bar: Bar) -> None:
        self.bars_seen += 1
        instrument_id = bar.bar_type.instrument_id
        is_long = self.portfolio.is_net_long(instrument_id)
        signal = ml_decision(
            self.config.probabilities.get(bar.ts_event),
            is_long,
            self.config.enter_threshold,
            self.config.exit_threshold,
        )
        if signal is Signal.ENTER_LONG and self.portfolio.is_net_flat(instrument_id):
            account = self.portfolio.account(instrument_id.venue)
            free = account.balance_free(self.instrument.quote_currency).as_double()
            qty = self.instrument.make_qty(self.config.position_fraction * free / bar.close.as_double())
            if qty.as_double() > 0:
                self.submit_order(
                    self.order_factory.market(
                        instrument_id=instrument_id, order_side=OrderSide.BUY, quantity=qty,
                    ),
                )
                self.signals.append((bar.ts_event, signal))
        elif signal is Signal.EXIT_LONG:
            self.close_all_positions(instrument_id)
            self.signals.append((bar.ts_event, signal))

    def on_stop(self) -> None:
        self.close_all_positions(self.config.bar_type.instrument_id)
