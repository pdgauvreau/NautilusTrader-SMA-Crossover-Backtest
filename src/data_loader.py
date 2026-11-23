"""Load Binance spot kline CSVs and convert them to NautilusTrader v2 bars.

Pipeline::

    df = load_klines_csv("data/raw")                 # tidy UTC-indexed DataFrame
    summary = validate_daily(df)                     # integrity checks
    instrument = btcusdt_instrument()
    bars = to_nautilus_bars(df, instrument)          # list[nautilus_trader.model.Bar]

Timestamp convention (look-ahead safety)
----------------------------------------
A daily kline's OHLCV is only fully known when the day *closes*. Every ``Bar``
produced here therefore has ``ts_event == ts_init == open_time + 1 interval``
(i.e. the next day's 00:00 UTC for 1d bars). A strategy receiving the bar for
2025-01-01 sees it at 2025-01-02T00:00Z, so it can never act on a close price
before that close has happened.

Fees
----
In Nautilus v2 ``CurrencyPair`` no longer carries ``maker_fee`` / ``taker_fee``;
commissions are configured on the venue via a fee model. Use
:func:`btcusdt_fee_model` (``MakerTakerFeeModel``, 0.1% / 0.1%) when adding the
venue to the backtest engine.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import CurrencyPair
from nautilus_trader.testkit.providers import TestInstrumentProvider

KLINE_COLUMNS: list[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
FLOAT_COLUMNS: list[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
]

DEFAULT_BAR_TYPE = "BTCUSDT.BINANCE-1-DAY-LAST-EXTERNAL"
MAKER_FEE = Decimal("0.001")
TAKER_FEE = Decimal("0.001")

# Epoch values above this are microseconds; below are milliseconds.
# 1e14 us ~= 1973-03, 1e14 ms ~= year 5138, so the threshold is unambiguous.
_US_THRESHOLD = 10**14


def _epoch_to_utc(values: pd.Series) -> pd.DatetimeIndex | pd.Series:
    """Convert integer epoch timestamps (ms or us, detected per row) to UTC datetimes."""
    ints = values.astype("int64")
    is_us = ints >= _US_THRESHOLD
    as_us = ints.where(is_us, ints * 1_000)  # normalise everything to microseconds
    return pd.to_datetime(as_us, unit="us", utc=True)


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single Binance kline CSV, tolerating an optional header row."""
    df = pd.read_csv(path, header=None, names=KLINE_COLUMNS, dtype=str)
    # Drop a header row if present (first cell not numeric).
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
    return df


def load_klines_csv(
    raw_dir: str | Path,
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load and concatenate monthly Binance kline CSVs from ``raw_dir``.

    Args:
        raw_dir: Directory containing ``{symbol}-{interval}-YYYY-MM.csv`` files.
        symbol: Binance symbol, e.g. ``"BTCUSDT"``.
        interval: Kline interval, e.g. ``"1d"``.
        start: Optional inclusive lower bound on open time (UTC if naive).
        end: Optional inclusive upper bound on open time (UTC if naive).

    Returns:
        DataFrame indexed by ``open_time`` (tz-aware UTC, sorted, unique) with float
        OHLCV/quote-volume columns, int ``count`` and a UTC ``close_time`` column.
        Millisecond (pre-2025) and microsecond (2025+) epochs are both handled.

    Raises:
        FileNotFoundError: If no matching CSV files exist.
    """
    files = sorted(Path(raw_dir).glob(f"{symbol}-{interval}-*.csv"))
    if not files:
        raise FileNotFoundError(f"No {symbol}-{interval}-*.csv files in {raw_dir}")

    raw = pd.concat([_read_one(f) for f in files], ignore_index=True)

    df = pd.DataFrame(
        {
            "open_time": _epoch_to_utc(pd.to_numeric(raw["open_time"])),
            "close_time": _epoch_to_utc(pd.to_numeric(raw["close_time"])),
        }
    )
    for col in FLOAT_COLUMNS:
        df[col] = pd.to_numeric(raw[col]).astype("float64")
    df["count"] = pd.to_numeric(raw["count"]).astype("int64")

    df = (
        df.drop_duplicates(subset="open_time", keep="last")
        .sort_values("open_time")
        .set_index("open_time")
    )

    if start is not None:
        df = df[df.index >= _as_utc(start)]
    if end is not None:
        df = df[df.index <= _as_utc(end)]
    return df[["open", "high", "low", "close", "volume", *FLOAT_COLUMNS[5:], "count", "close_time"]]


def _as_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    """Return ``ts`` as a tz-aware UTC timestamp (naive values are assumed UTC)."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _check(condition: bool, message: str) -> None:
    """Raise ``ValueError`` if a data check fails (unlike ``assert``, survives ``python -O``)."""
    if not condition:
        raise ValueError(message)


def validate_daily(df: pd.DataFrame) -> dict[str, object]:
    """Validate a daily kline DataFrame and return a summary.

    Checks: non-empty, unique timestamps, every step exactly one day, no NaNs in
    OHLCV, ``high >= max(open, close)`` and ``low <= min(open, close)``.

    Args:
        df: Output of :func:`load_klines_csv` for a ``1d`` interval.

    Returns:
        Summary dict with ``rows``, ``first``, ``last``, ``first_close``, ``last_close``.

    Raises:
        ValueError: If any check fails, with a message naming the offending rows.
    """
    _check(len(df) > 0, "DataFrame is empty")
    idx = df.index
    _check(idx.is_unique, f"Duplicate timestamps: {idx[idx.duplicated()].tolist()}")
    _check(idx.is_monotonic_increasing, "Index is not sorted")

    steps = idx.to_series().diff().dropna()
    bad_steps = steps[steps != pd.Timedelta(days=1)]
    _check(bad_steps.empty, f"Gaps/irregular steps at: {bad_steps.index.tolist()}")

    ohlcv = df[["open", "high", "low", "close", "volume"]]
    _check(not ohlcv.isna().any().any(), f"NaNs in rows: {ohlcv[ohlcv.isna().any(axis=1)].index.tolist()}")

    bad_high = df[df["high"] < df[["open", "close"]].max(axis=1)]
    _check(bad_high.empty, f"high < max(open, close) at: {bad_high.index.tolist()}")
    bad_low = df[df["low"] > df[["open", "close"]].min(axis=1)]
    _check(bad_low.empty, f"low > min(open, close) at: {bad_low.index.tolist()}")

    return {
        "rows": len(df),
        "first": idx[0].date().isoformat(),
        "last": idx[-1].date().isoformat(),
        "first_close": float(df["close"].iloc[0]),
        "last_close": float(df["close"].iloc[-1]),
    }


def btcusdt_instrument() -> CurrencyPair:
    """Return the Binance BTCUSDT spot ``CurrencyPair`` (price precision 2, size precision 6).

    Note: v2 instruments have no fee fields; pair this with :func:`btcusdt_fee_model`.
    """
    return TestInstrumentProvider.btcusdt_binance()


def btcusdt_fee_model() -> MakerTakerFeeModel:
    """Return a maker 0.1% / taker 0.1% fee model for the simulated Binance venue."""
    return MakerTakerFeeModel(maker_rate=MAKER_FEE, taker_rate=TAKER_FEE)


def to_nautilus_bars(
    df: pd.DataFrame,
    instrument: CurrencyPair,
    bar_type: BarType | str | None = None,
) -> list[Bar]:
    """Convert a kline DataFrame into Nautilus ``Bar`` objects.

    Bars are constructed directly: the v2 ``BarDataWrangler`` only accepts Arrow
    record-batch bytes (it requires ``pyarrow``), so it is no simpler than this.

    Timestamps: ``ts_event`` and ``ts_init`` are set to the bar's *close* time
    (``open_time + bar duration``), not its open, so a backtest never sees a bar
    before it has completed. See the module docstring.

    Args:
        df: Output of :func:`load_klines_csv` (UTC ``open_time`` index).
        instrument: Instrument whose price/size precision is applied.
        bar_type: Bar type or its string form; defaults to
            ``BTCUSDT.BINANCE-1-DAY-LAST-EXTERNAL``.

    Returns:
        Bars in chronological order, prices rounded to ``instrument.price_precision``
        and volumes to ``instrument.size_precision``.
    """
    bt = BarType.from_str(bar_type or DEFAULT_BAR_TYPE) if not isinstance(bar_type, BarType) else bar_type
    if bt.instrument_id != instrument.id:
        raise ValueError(f"Bar type {bt} does not match instrument {instrument.id}")

    duration = _bar_duration(df)
    close_ns = (df.index + duration).as_unit("ns").asi8

    make_px, make_qty = instrument.make_price, instrument.make_qty
    return [
        Bar(
            bt,
            make_px(o),
            make_px(h),
            make_px(lo),
            make_px(c),
            make_qty(v),
            int(ts),
            int(ts),
        )
        for o, h, lo, c, v, ts in zip(
            df["open"], df["high"], df["low"], df["close"], df["volume"], close_ns, strict=True
        )
    ]


def _bar_duration(df: pd.DataFrame) -> pd.Timedelta:
    """Infer the bar duration from ``close_time`` (Binance close is end-of-bar minus 1 unit)."""
    if len(df) == 0:
        return pd.Timedelta(days=1)
    span = df["close_time"].iloc[0] - df.index[0]
    # Binance close_time = open_time + duration - 1ms (or 1us); round up to whole seconds.
    return span.ceil("s")


def _main() -> None:
    """Verify the downloaded 2025 BTCUSDT daily data end to end."""
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    df = load_klines_csv(raw_dir, start="2025-01-01", end="2025-12-31")
    summary = validate_daily(df)
    print("DataFrame summary:", summary)
    assert summary["rows"] == 365, summary
    assert summary["first"] == "2025-01-01" and summary["last"] == "2025-12-31", summary

    instrument = btcusdt_instrument()
    bars = to_nautilus_bars(df, instrument)
    assert len(bars) == len(df)
    first, last = bars[0], bars[-1]
    print(f"Instrument: {instrument}")
    print(f"Fee model: {btcusdt_fee_model()} (maker={MAKER_FEE}, taker={TAKER_FEE})")
    print(f"Bars: {len(bars)}  type={first.bar_type}")
    print(f"First bar: {first}  ts_event={pd.Timestamp(first.ts_event, unit='ns', tz='UTC')}")
    print(f"Last bar:  {last}  ts_event={pd.Timestamp(last.ts_event, unit='ns', tz='UTC')}")
    assert pd.Timestamp(first.ts_event, unit="ns", tz="UTC") == pd.Timestamp("2025-01-02", tz="UTC")
    assert float(first.close) == round(summary["first_close"], 2)
    assert float(last.close) == round(summary["last_close"], 2)
    print("OK: all checks passed")


if __name__ == "__main__":
    _main()
