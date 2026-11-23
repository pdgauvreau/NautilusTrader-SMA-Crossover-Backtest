"""The strategy must not trade until the slow SMA (50 bars) has warmed up."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.strategies.sma_crossover import Signal
from tests.conftest import sine_closes
from tests.conftest import step_closes
from tests.conftest import ts_utc

SLOW = 50


def _fill_times(fills: pd.DataFrame) -> list[pd.Timestamp]:
    return [] if fills.empty else sorted(pd.to_datetime(fills["ts_last"], utc=True))


def test_uptrend_from_bar_one_does_not_buy_during_warmup(make_bars, instrument, config) -> None:
    """Fast > slow from the very start: a naive strategy would buy immediately.

    Once warmed up the spread is already positive, so there is no crossover and
    therefore no trade at all, in particular none inside the warmup window.
    """
    bars = make_bars(300.0 + 1.5 * np.arange(120))
    out = run_backtest(bars, instrument, config)

    assert out.bars_seen == len(bars)
    assert out.signals == []
    assert out.fills.empty


def test_oscillating_market_trades_only_after_warmup(make_bars, instrument, config) -> None:
    """A 20-bar cycle produces fast/slow crosses from the start; none may trade before bar 50."""
    bars = make_bars(sine_closes(200, period=20))
    out = run_backtest(bars, instrument, config)

    assert out.bars_seen == len(bars)
    fills = _fill_times(out.fills)
    assert fills, "expected trades after warmup, otherwise the test is vacuous"

    warm_ts = ts_utc(bars[SLOW - 1].ts_event)  # the 50th bar: first with both SMAs ready
    assert all(ts >= warm_ts for ts in fills)
    assert all(ts_utc(ts) >= warm_ts for ts, _ in out.signals)
    # Stricter: the first warmed-up bar has no previous spread, so it cannot signal either.
    assert min(fills) > warm_ts


@pytest.mark.parametrize("n_bars", [1, 10, SLOW - 1])
def test_fewer_bars_than_warmup_means_no_trades(make_bars, instrument, config, n_bars: int) -> None:
    bars = make_bars(sine_closes(n_bars, period=20))
    out = run_backtest(bars, instrument, config)

    assert out.bars_seen == n_bars
    assert out.signals == []
    assert out.fills.empty
    assert out.equity.iloc[-1] == pytest.approx(out.starting_capital)


def test_crossover_right_after_warmup_trades_on_that_bar(make_bars, instrument, config) -> None:
    """Decline through warmup, then jump on bar index 50: the first bar that can signal.

    Guards against the warmup tests passing vacuously (e.g. a strategy that never trades).
    """
    bars = make_bars(step_closes(decline_bars=SLOW, total_bars=80))
    out = run_backtest(bars, instrument, config)

    assert out.bars_seen == len(bars)
    entry_bar = bars[SLOW]
    assert out.signals[0] == (entry_bar.ts_event, Signal.ENTER_LONG)

    buys = out.fills[out.fills["side"] == "BUY"]
    assert len(buys) == 1
    assert pd.Timestamp(buys["ts_last"].iloc[0]) == ts_utc(entry_bar.ts_event)
    # Filled at the signal bar's close (no look-ahead to a later bar).
    assert float(buys["avg_px"].iloc[0]) == pytest.approx(entry_bar.close.as_double())
    assert float(str(buys["filled_qty"].iloc[0])) == pytest.approx(config.trade_size)
