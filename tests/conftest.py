"""Shared fixtures and synthetic-data helpers for the test suite.

Synthetic bars are built the same way as real ones: a UTC-indexed OHLCV
DataFrame (shaped like ``load_klines_csv`` output) is passed through
``to_nautilus_bars``, so every test exercises the real conversion path,
including the "stamp bars at their close" timestamp convention.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import CurrencyPair

from src.data_loader import DEFAULT_BAR_TYPE
from src.data_loader import btcusdt_instrument
from src.data_loader import to_nautilus_bars
from src.strategies.sma_crossover import SmaCrossoverConfig

STARTING_CAPITAL = 100_000.0


def make_ohlcv(closes: Sequence[float], start: str = "2024-01-01") -> pd.DataFrame:
    """Daily OHLCV frame with a UTC open-time index, mimicking ``load_klines_csv``.

    ``open`` equals the previous close, ``high``/``low`` bracket open and close by
    0.5%, and ``close_time`` follows Binance's ``open + 1d - 1ms`` convention.
    """
    close = np.round(np.asarray(closes, dtype=float), 2)
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range(start, periods=len(close), freq="D", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * 1.005,
            "low": np.minimum(open_, close) * 0.995,
            "close": close,
            "volume": 100.0,
            "close_time": index + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1),
        },
        index=index,
    )


def sine_closes(n: int, period: int = 20, level: float = 30_000.0, amplitude: float = 3_000.0) -> np.ndarray:
    """Oscillating closes that produce frequent fast/slow SMA crossovers."""
    t = np.arange(n)
    return level + amplitude * np.sin(2 * np.pi * t / period)


def step_closes(decline_bars: int, total_bars: int, start: float = 40_000.0, slope: float = -50.0,
                jump_to: float = 60_000.0) -> np.ndarray:
    """A steady decline for ``decline_bars`` bars, then a flat plateau at ``jump_to``.

    The decline keeps ``fast - slow`` strictly negative; the jump flips it
    positive on the very first plateau bar (index ``decline_bars``). With the
    defaults and 10/50 SMAs that spread is +800 (a jump to exactly 50_000 would
    leave it at 0, a mere touch, so the cross would only happen one bar later).
    """
    t = np.arange(total_bars)
    return np.where(t < decline_bars, start + slope * t, jump_to)


def ts_utc(ts_ns: int) -> pd.Timestamp:
    """Nautilus UNIX-nanosecond timestamp -> tz-aware pandas Timestamp."""
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC")


def money(value: object) -> float:
    """Parse a Nautilus money string such as ``"14.22746000 USDT"``."""
    return float(str(value).split()[0])


@pytest.fixture(scope="session")
def instrument() -> CurrencyPair:
    return btcusdt_instrument()


@pytest.fixture(scope="session")
def bar_type() -> BarType:
    return BarType.from_str(DEFAULT_BAR_TYPE)


@pytest.fixture(scope="session")
def make_bars(instrument: CurrencyPair, bar_type: BarType) -> Callable[[Sequence[float]], list[Bar]]:
    """Factory: closes -> list of Nautilus bars for BTCUSDT daily."""

    def _make(closes: Sequence[float], start: str = "2024-01-01") -> list[Bar]:
        return to_nautilus_bars(make_ohlcv(closes, start), instrument, bar_type)

    return _make


@pytest.fixture
def config(bar_type: BarType) -> SmaCrossoverConfig:
    """Default 10/50 crossover config trading 0.5 BTC."""
    return SmaCrossoverConfig(bar_type=bar_type)
