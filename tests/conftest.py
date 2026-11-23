"""Shared fixtures and synthetic-data helpers for the test suite.

Synthetic bars are built the same way as real ones: a session-indexed OHLCV
DataFrame (shaped like ``load_bars`` output, weekdays only) is passed through
``to_nautilus_bars``, so every test exercises the real conversion path,
including the "stamp bars at the 16:00 New York close" convention.

Prices are ETF-like (hundreds of dollars) so whole-share sizing behaves as it
would for SPY/QQQ/IWM.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import Equity

from src.data_loader import bar_close_times
from src.data_loader import bar_type_for
from src.data_loader import etf_instrument
from src.data_loader import to_nautilus_bars
from src.strategies.sma_crossover import SmaCrossoverConfig

STARTING_CAPITAL = 100_000.0


def make_ohlcv(closes: Sequence[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Daily OHLCV on consecutive weekdays, indexed by session date (UTC midnight).

    ``open`` equals the previous close; ``high``/``low`` bracket open and close by 0.5%.
    """
    close = np.round(np.asarray(closes, dtype=float), 2)
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.bdate_range(start, periods=len(close), tz="UTC", name="session")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": 1_000_000.0,
        },
        index=index,
    )


def close_times(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Bar close timestamps (what strategies see as ``bar.ts_event``)."""
    return bar_close_times(df)


def sine_closes(n: int, period: int = 20, level: float = 300.0, amplitude: float = 30.0) -> np.ndarray:
    """Oscillating closes that produce frequent fast/slow SMA crossovers."""
    t = np.arange(n)
    return level + amplitude * np.sin(2 * np.pi * t / period)


def step_closes(decline_bars: int, total_bars: int, start: float = 400.0, slope: float = -0.5,
                jump_to: float = 600.0) -> np.ndarray:
    """A steady decline for ``decline_bars`` bars, then a flat plateau at ``jump_to``.

    The decline keeps ``fast - slow`` strictly negative; the jump flips it
    positive on the very first plateau bar (index ``decline_bars``). With the
    defaults and 10/50 SMAs that spread is +8 (a jump to exactly 500 would
    leave it at 0, a mere touch, so the cross would only happen one bar later).
    """
    t = np.arange(total_bars)
    return np.where(t < decline_bars, start + slope * t, jump_to)


def ts_utc(ts_ns: int) -> pd.Timestamp:
    """Nautilus UNIX-nanosecond timestamp -> tz-aware pandas Timestamp."""
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC")


def money(value: object) -> float:
    """Parse a Nautilus money string such as ``"14.22 USD"``."""
    return float(str(value).split()[0])


@pytest.fixture(scope="session")
def instrument() -> Equity:
    return etf_instrument("SPY")


@pytest.fixture(scope="session")
def bar_type() -> BarType:
    return bar_type_for("SPY")


@pytest.fixture(scope="session")
def make_bars(instrument: Equity, bar_type: BarType) -> Callable[[Sequence[float]], list[Bar]]:
    """Factory: closes -> list of Nautilus bars for SPY daily."""

    def _make(closes: Sequence[float], start: str = "2024-01-01") -> list[Bar]:
        return to_nautilus_bars(make_ohlcv(closes, start), instrument, bar_type)

    return _make


@pytest.fixture
def config(bar_type: BarType) -> SmaCrossoverConfig:
    """Default 10/50 crossover config trading a fixed 100 shares, filling at the signal close."""
    return SmaCrossoverConfig(bar_type=bar_type)
