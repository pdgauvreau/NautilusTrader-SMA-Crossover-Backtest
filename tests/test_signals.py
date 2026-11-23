"""Crossover signal logic: the pure rule, a synthetic series, and config validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategies.sma_crossover import Signal
from src.strategies.sma_crossover import SmaCrossoverConfig
from src.strategies.sma_crossover import crossover_signal

# --- crossover_signal unit tests -------------------------------------------------


@pytest.mark.parametrize("diff", [-5.0, 0.0, 5.0])
def test_no_previous_value_never_signals(diff: float) -> None:
    assert crossover_signal(None, diff) is None


@pytest.mark.parametrize(("prev", "diff"), [(-1.0, 1.0), (-1e-9, 1e-9), (-100.0, 0.01)])
def test_cross_up_enters_long(prev: float, diff: float) -> None:
    assert crossover_signal(prev, diff) is Signal.ENTER_LONG


@pytest.mark.parametrize(("prev", "diff"), [(1.0, -1.0), (1e-9, -1e-9), (100.0, -0.01)])
def test_cross_down_exits_long(prev: float, diff: float) -> None:
    assert crossover_signal(prev, diff) is Signal.EXIT_LONG


def test_leaving_zero_counts_as_a_cross() -> None:
    # A spread that sat exactly on zero and then moves away has crossed.
    assert crossover_signal(0.0, 1.0) is Signal.ENTER_LONG
    assert crossover_signal(0.0, -1.0) is Signal.EXIT_LONG


@pytest.mark.parametrize(("prev", "diff"), [(-1.0, 0.0), (1.0, 0.0), (0.0, 0.0)])
def test_touching_zero_is_not_a_cross(prev: float, diff: float) -> None:
    assert crossover_signal(prev, diff) is None


@pytest.mark.parametrize(("prev", "diff"), [(1.0, 2.0), (2.0, 1.0), (-1.0, -2.0), (-2.0, -1.0)])
def test_no_cross_when_sign_unchanged(prev: float, diff: float) -> None:
    assert crossover_signal(prev, diff) is None


# --- synthetic series with hand-derived crossovers ------------------------------


def _signals_over(closes: pd.Series, fast: int = 10, slow: int = 50) -> list[tuple[int, Signal]]:
    """Run ``crossover_signal`` along pandas SMAs, mirroring the strategy's warmup."""
    diff = closes.rolling(fast).mean() - closes.rolling(slow).mean()
    out: list[tuple[int, Signal]] = []
    prev: float | None = None
    for i, d in enumerate(diff):
        if np.isnan(d):
            continue  # slow SMA not yet warmed up
        sig = crossover_signal(prev, d)
        prev = d
        if sig is not None:
            out.append((i, sig))
    return out


def test_synthetic_series_crosses_at_expected_indices() -> None:
    """Decline (t<60), jump to a high plateau (60<=t<100), drop to a low plateau.

    Hand calculation with fast=10, slow=50:
    * t=49..59: steady decline, the recent (fast) mean is below the slow mean -> diff < 0.
    * t=60: fast ~= 1050 vs slow ~= 986 -> cross up -> ENTER at 60.
    * t=100: fast = 1850 vs slow ~= 1780 -> still above.
    * t=101: fast = 1700 vs slow ~= 1771 -> cross down -> EXIT at 101.
    The series stops at t=130, before the slow SMA converges onto the low plateau.
    """
    t = np.arange(131)
    closes = pd.Series(np.select([t < 60, t < 100], [1000.0 - t, 2000.0], 500.0))

    assert _signals_over(closes) == [(60, Signal.ENTER_LONG), (101, Signal.EXIT_LONG)]


def test_no_signal_on_first_warmed_up_bar() -> None:
    """A series already in an uptrend when warmup ends yields no signal at all."""
    closes = pd.Series(30_000.0 + 100.0 * np.arange(120))
    assert _signals_over(closes) == []


# --- config validation ----------------------------------------------------------


@pytest.mark.parametrize(("fast", "slow"), [(50, 50), (60, 50)])
def test_fast_period_must_be_below_slow(bar_type, fast: int, slow: int) -> None:
    with pytest.raises(ValueError, match="fast_period"):
        SmaCrossoverConfig(bar_type=bar_type, fast_period=fast, slow_period=slow)


@pytest.mark.parametrize(("fast", "slow"), [(0, 50), (-1, 50), (10, 0)])
def test_periods_must_be_positive(bar_type, fast: int, slow: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        SmaCrossoverConfig(bar_type=bar_type, fast_period=fast, slow_period=slow)


@pytest.mark.parametrize("size", [0.0, -0.5])
def test_trade_size_must_be_positive(bar_type, size: float) -> None:
    with pytest.raises(ValueError, match="trade_size"):
        SmaCrossoverConfig(bar_type=bar_type, trade_size=size)


def test_config_defaults(bar_type) -> None:
    cfg = SmaCrossoverConfig(bar_type=bar_type)
    assert (cfg.fast_period, cfg.slow_period, cfg.trade_size, cfg.close_on_stop) == (10, 50, 0.5, True)
    assert cfg.bar_type == bar_type
