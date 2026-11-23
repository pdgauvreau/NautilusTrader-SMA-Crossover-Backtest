"""The SMA crossover strategy, instrumented for paper trading on live data.

Identical decision logic to the backtested :class:`SmaCrossover`; this subclass
only adds what a live session needs:

* **Warmup from history:** requests the last ``slow_period + 10`` bars on start.
  Nautilus feeds them through the registered SMAs; this class then records the
  last SMA spread so a cross on the very first live bar is detected. Crosses
  *inside* the history are not traded.
* **Live quotes:** subscribes to best bid/ask, so the sandbox fills market
  orders at the real ask/bid instead of the bar close. The gap between the two
  is exactly the spread cost the backtest ignores, and it is journaled.
* **Journal:** every bar, signal, fill and stop is appended to a JSONL file.
"""

from __future__ import annotations

from nautilus_trader.model import Bar
from nautilus_trader.model import QuoteTick

from src.paper.journal import Journal
from src.strategies.sma_crossover import SmaCrossover
from src.strategies.sma_crossover import SmaCrossoverConfig


class PaperSmaCrossoverConfig(SmaCrossoverConfig):
    def __init__(self, journal_path: str, subscribe_quotes: bool = True, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.journal_path = journal_path
        self.subscribe_quotes = subscribe_quotes


class PaperSmaCrossover(SmaCrossover):
    def __init__(self, config: PaperSmaCrossoverConfig) -> None:
        super().__init__(config)
        self.journal = Journal(config.journal_path)
        self.last_quote: QuoteTick | None = None
        self.last_bar_close: float | None = None
        self.warmup_bars = 0

    # ----------------------------------------------------------------- lifecycle
    def on_start(self) -> None:
        super().on_start()  # instrument, indicators, live bar subscription
        bar_type = self.config.bar_type
        if self.config.subscribe_quotes:
            self.subscribe_quotes(bar_type.instrument_id)
        self.request_bars(bar_type, limit=self.config.slow_period + 10)
        self.journal.write(
            "start",
            bar_type=str(bar_type),
            fast=self.config.fast_period,
            slow=self.config.slow_period,
            position_fraction=self.config.position_fraction,
            enter_if_above_on_start=self.config.enter_if_above_on_start,
            equity=self._equity(),
        )

    def on_historical_bars(self, bars) -> None:
        self.warmup_bars += len(bars)
        if self.fast.initialized and self.slow.initialized:
            # Seed the cross detector with the last *historical* spread.
            self._prev_diff = self.fast.value - self.slow.value
        self.journal.write(
            "warmup",
            bars=len(bars),
            first=_bar_time(bars[0]) if bars else None,
            last=_bar_time(bars[-1]) if bars else None,
            fast=self.fast.value if self.fast.initialized else None,
            slow=self.slow.value if self.slow.initialized else None,
            ready=bool(self.fast.initialized and self.slow.initialized),
        )

    def on_stop(self) -> None:
        super().on_stop()
        self.journal.write("stop", equity=self._equity(), position=self._position_qty(),
                           bars_seen=self.bars_seen, signals=len(self.signals))

    # ---------------------------------------------------------------------- data
    def on_quote(self, tick: QuoteTick) -> None:
        self.last_quote = tick

    def on_bar(self, bar: Bar) -> None:
        signals_before = len(self.signals)
        super().on_bar(bar)
        self.last_bar_close = bar.close.as_double()
        new_signal = self.signals[-1][1].value if len(self.signals) > signals_before else None
        ready = self.fast.initialized and self.slow.initialized
        self.journal.write(
            "bar",
            bar_time=_bar_time(bar),
            open=bar.open.as_double(),
            high=bar.high.as_double(),
            low=bar.low.as_double(),
            close=self.last_bar_close,
            fast=self.fast.value if ready else None,
            slow=self.slow.value if ready else None,
            signal=new_signal,
            position=self._position_qty(),
            equity=self._equity(bar.close.as_double()),
        )

    # -------------------------------------------------------------------- events
    def on_order_filled(self, event) -> None:
        px = event.last_px.as_double()
        ref = self.last_bar_close
        side = str(event.order_side).split(".")[-1]
        # Positive slippage = worse than the bar close the backtest would have used.
        slip_bps = None
        if ref:
            slip_bps = (px / ref - 1) * 1e4 * (1 if side == "BUY" else -1)
        q = self.last_quote
        self.journal.write(
            "fill",
            side=side,
            qty=event.last_qty.as_double(),
            price=px,
            bar_close=ref,
            slippage_vs_close_bps=slip_bps,
            bid=q.bid_price.as_double() if q else None,
            ask=q.ask_price.as_double() if q else None,
            commission=str(event.commission),
            liquidity=str(event.liquidity_side).split(".")[-1],
        )

    # ------------------------------------------------------------------- helpers
    def _position_qty(self) -> float:
        return float(self.portfolio.net_position(self.config.bar_type.instrument_id))

    def _equity(self, mark: float | None = None) -> float | None:
        if self.instrument is None:
            return None
        account = self.portfolio.account(self.config.bar_type.instrument_id.venue)
        if account is None:
            return None
        quote = account.balance_total(self.instrument.quote_currency)
        base = account.balance_total(self.instrument.base_currency)
        px = mark if mark is not None else (self.last_bar_close or 0.0)
        return (quote.as_double() if quote else 0.0) + (base.as_double() if base else 0.0) * px


def _bar_time(bar: Bar) -> str:
    import pandas as pd

    return pd.Timestamp(bar.ts_event, unit="ns", tz="UTC").isoformat()
