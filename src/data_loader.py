"""Load daily US stock/ETF bars (from Alpaca CSVs) and convert them to NautilusTrader v2 objects.

Pipeline::

    df = load_bars("data/raw", "SPY")              # one row per trading session
    summary = validate_daily(df)                   # integrity checks
    instrument = equity_instrument("SPY")
    bars = to_nautilus_bars(df, instrument)        # list[Bar], stamped at the 16:00 ET close
    quotes = open_quotes(df, instrument)           # QuoteTicks at the 09:30 ET open

Index convention
----------------
Rows are indexed by the **session date** (UTC midnight of that date). Real event
times come from :func:`bar_close_times` (16:00 America/New_York) and
:func:`bar_open_times` (09:30 America/New_York), so daylight-saving changes are
handled correctly. Early-close days (e.g. 13:00) are treated as 16:00; that
only shifts timestamps, not prices.

Look-ahead safety and fills
---------------------------
A daily bar is only known at the close, so each ``Bar`` has ``ts_event`` at
16:00 ET. A strategy that decides on that bar can realistically trade at the
**next session's open**. :func:`open_quotes` produces a quote at each open
(bid/ask = open -/+ half the assumed spread) so the simulated venue can fill
there. See ``execute_at_open`` in :mod:`src.strategies.sma_crossover`.

Prices are split- and dividend-adjusted (Alpaca ``adjustment=all``), so
returns include dividends.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pandas as pd
from nautilus_trader.execution import MakerTakerFeeModel
from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import Currency
from nautilus_trader.model import Equity
from nautilus_trader.model import InstrumentId
from nautilus_trader.model import Price
from nautilus_trader.model import Quantity
from nautilus_trader.model import QuoteTick
from nautilus_trader.model import Symbol

# Listing venue for each supported symbol (matches Interactive Brokers' symbology in Nautilus).
# To add a US stock or ETF: add "TICKER": "VENUE" here, then run scripts/download_data.py.
ETFS: dict[str, str] = {
    "SPY": "ARCA",
    "QQQ": "NASDAQ",
    "IWM": "ARCA",
}
STOCKS: dict[str, str] = {
    "NVDA": "NASDAQ",  # NVIDIA
    "SOFI": "NASDAQ",  # SoFi Technologies (listed June 2021, so shorter history)
}
SYMBOL_VENUES: dict[str, str] = {**ETFS, **STOCKS}
DEFAULT_SYMBOLS: tuple[str, ...] = tuple(SYMBOL_VENUES)
DEFAULT_SYMBOL = "SPY"

# Execution cost per side, as a fraction of notional.
# Commission: IBKR-style ~$0.005/share is well under 0.01% for these ETFs; 0.01% is conservative.
COMMISSION_RATE = 0.0001
# Default extra slippage on top of the modelled open spread, for research runs.
DEFAULT_SLIPPAGE = 0.0002
DEFAULT_COST_RATE = COMMISSION_RATE + DEFAULT_SLIPPAGE
# Half the bid/ask spread assumed at the open, applied to the synthetic open quotes.
DEFAULT_HALF_SPREAD = 0.0001  # 1 bp each side; SPY/QQQ/IWM typically trade tighter

NEW_YORK = "America/New_York"
CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
PRICE_PRECISION = 4  # adjusted historical prices are fractional (e.g. NVDA ~$0.79 in 2016)


def _check_symbol(symbol: str) -> str:
    if symbol not in SYMBOL_VENUES:
        raise ValueError(f"Unsupported symbol {symbol!r}; add it to SYMBOL_VENUES ({', '.join(SYMBOL_VENUES)})")
    return symbol


def csv_path(raw_dir: str | Path, symbol: str) -> Path:
    return Path(raw_dir) / f"{_check_symbol(symbol)}_1d.csv"


def load_bars(
    raw_dir: str | Path,
    symbol: str = DEFAULT_SYMBOL,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load ``<raw_dir>/<SYMBOL>_1d.csv`` written by ``scripts/download_data.py``.

    Returns a DataFrame indexed by session date (UTC midnight), with float
    ``open, high, low, close, volume`` columns, sorted and de-duplicated.
    ``start``/``end`` are inclusive session dates.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    path = csv_path(raw_dir, symbol)
    if not path.exists():
        raise FileNotFoundError(f"No data for {symbol} at {path}; run scripts/download_data.py")
    raw = pd.read_csv(path)
    missing = set(CSV_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    df = raw[CSV_COLUMNS].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    for col in CSV_COLUMNS[1:]:
        df[col] = pd.to_numeric(df[col]).astype("float64")
    df = df.drop_duplicates("date", keep="last").sort_values("date").set_index("date")
    df.index.name = "session"
    if start is not None:
        df = df[df.index >= _as_utc(start)]
    if end is not None:
        df = df[df.index <= _as_utc(end)]
    return df


def _as_utc(ts: str | pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _check(condition: bool, message: str) -> None:
    """Raise ``ValueError`` if a data check fails (unlike ``assert``, survives ``python -O``)."""
    if not condition:
        raise ValueError(message)


def validate_daily(df: pd.DataFrame, max_gap_days: int = 5) -> dict[str, object]:
    """Integrity checks for daily trading-session bars.

    Weekends and holidays are expected, so the gap check allows up to
    ``max_gap_days`` calendar days between sessions (a long weekend plus a
    holiday is 4). Larger gaps mean missing data.

    Raises:
        ValueError: If any check fails, with a message naming the offending rows.
    """
    _check(len(df) > 0, "DataFrame is empty")
    idx = df.index
    _check(idx.is_unique, f"Duplicate sessions: {idx[idx.duplicated()].tolist()}")
    _check(idx.is_monotonic_increasing, "Index is not sorted")
    gaps = idx.to_series().diff().dt.days
    big = gaps[gaps > max_gap_days]
    _check(big.empty, f"Gaps longer than {max_gap_days} days before: {big.index.tolist()}")
    weekend = idx[idx.day_of_week >= 5]
    _check(len(weekend) == 0, f"Weekend sessions: {weekend.tolist()}")
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    _check(not ohlcv.isna().any().any(), f"NaNs in rows: {ohlcv[ohlcv.isna().any(axis=1)].index.tolist()}")
    _check((ohlcv[["open", "high", "low", "close"]] > 0).all().all(), "Non-positive prices")
    tol = 1e-9
    bad_high = df[df["high"] < df[["open", "close"]].max(axis=1) - tol]
    _check(bad_high.empty, f"high < max(open, close) at: {bad_high.index.tolist()}")
    bad_low = df[df["low"] > df[["open", "close"]].min(axis=1) + tol]
    _check(bad_low.empty, f"low > min(open, close) at: {bad_low.index.tolist()}")
    return {
        "rows": len(df),
        "first": idx[0].date().isoformat(),
        "last": idx[-1].date().isoformat(),
        "first_close": float(df["close"].iloc[0]),
        "last_close": float(df["close"].iloc[-1]),
        "max_gap_days": int(gaps.max()) if len(gaps) > 1 else 0,
    }


# ------------------------------------------------------------------ timestamps
def _session_time(index: pd.DatetimeIndex, hhmm: str) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index).tz_localize(None).normalize() if index.tz is not None \
        else pd.DatetimeIndex(index).normalize()
    local = (dates + pd.Timedelta(hhmm + ":00")).tz_localize(NEW_YORK)
    return local.tz_convert("UTC")


def bar_close_times(df_or_index: pd.DataFrame | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """16:00 America/New_York on each session date, in UTC."""
    index = df_or_index.index if isinstance(df_or_index, pd.DataFrame) else df_or_index
    return pd.DatetimeIndex(_session_time(index, "16:00"), name="close_time")


def bar_open_times(df_or_index: pd.DataFrame | pd.DatetimeIndex) -> pd.DatetimeIndex:
    """09:30 America/New_York on each session date, in UTC."""
    index = df_or_index.index if isinstance(df_or_index, pd.DataFrame) else df_or_index
    return pd.DatetimeIndex(_session_time(index, "09:30"), name="open_time")


# ----------------------------------------------------------------- instruments
def equity_instrument(symbol: str = DEFAULT_SYMBOL) -> Equity:
    """Nautilus ``Equity`` for a US stock or ETF: USD, whole shares, 4-decimal prices (adjusted data)."""
    venue = SYMBOL_VENUES[_check_symbol(symbol)]
    return Equity(
        instrument_id=InstrumentId.from_str(f"{symbol}.{venue}"),
        raw_symbol=Symbol(symbol),
        currency=Currency.from_str("USD"),
        price_precision=PRICE_PRECISION,
        price_increment=Price(10**-PRICE_PRECISION, PRICE_PRECISION),
        lot_size=Quantity(1, 0),
        ts_event=0,
        ts_init=0,
    )


def bar_type_for(symbol: str = DEFAULT_SYMBOL, interval: str = "1d") -> BarType:
    """Nautilus ``BarType`` for a symbol, e.g. ``SPY.ARCA-1-DAY-LAST-EXTERNAL``."""
    units = {"m": "MINUTE", "h": "HOUR", "d": "DAY"}
    step, unit = int(interval[:-1]), units[interval[-1]]
    venue = SYMBOL_VENUES[_check_symbol(symbol)]
    return BarType.from_str(f"{symbol}.{venue}-{step}-{unit}-LAST-EXTERNAL")


def fee_model(cost_rate: float | Decimal = DEFAULT_COST_RATE) -> MakerTakerFeeModel:
    """Fee model charging ``cost_rate`` of notional per side (commission + slippage)."""
    rate = Decimal(str(cost_rate))
    return MakerTakerFeeModel(maker_rate=rate, taker_rate=rate)


# ------------------------------------------------------------------ conversion
def to_nautilus_bars(df: pd.DataFrame, instrument: Equity, bar_type: BarType | None = None) -> list[Bar]:
    """Daily bars stamped at the session close (16:00 ET)."""
    bar_type = bar_type or bar_type_for(str(instrument.raw_symbol))
    if bar_type.instrument_id != instrument.id:
        raise ValueError(f"bar_type {bar_type} does not match instrument {instrument.id}")
    ts = [int(t.value) for t in bar_close_times(df)]  # Timestamp.value is always nanoseconds
    bars = []
    for t, (o, h, lo, c, v) in zip(ts, df[["open", "high", "low", "close", "volume"]].itertuples(index=False)):
        bars.append(
            Bar(
                bar_type,
                instrument.make_price(o),
                instrument.make_price(h),
                instrument.make_price(lo),
                instrument.make_price(c),
                Quantity(max(v, 0.0), 0),
                t,
                t,
            ),
        )
    return bars


def open_quotes(
    df: pd.DataFrame,
    instrument: Equity,
    half_spread: float = DEFAULT_HALF_SPREAD,
    size: float = 1_000_000,
) -> list[QuoteTick]:
    """One quote per session at the 09:30 ET open: bid/ask = open * (1 -/+ half_spread).

    Deep size so a market order fills entirely at the quote (these ETFs are
    very liquid; the spread term is the modelled cost).
    """
    quotes = []
    for t, o in zip(bar_open_times(df), df["open"]):
        bid = instrument.make_price(o * (1 - half_spread))
        ask = instrument.make_price(o * (1 + half_spread))
        if ask.as_double() <= bid.as_double():
            ask = instrument.make_price(bid.as_double() + 0.01)
        ts = int(t.value)
        quotes.append(QuoteTick(instrument.id, bid, ask, Quantity(size, 0), Quantity(size, 0), ts, ts))
    return quotes


def _main() -> None:
    """Verify every downloaded file (``python -m src.data_loader``)."""
    raw = Path(__file__).resolve().parents[1] / "data" / "raw"
    for symbol in DEFAULT_SYMBOLS:
        try:
            df = load_bars(raw, symbol)
        except FileNotFoundError as exc:
            print(f"{symbol}: {exc}")
            continue
        summary = validate_daily(df)
        bars = to_nautilus_bars(df, equity_instrument(symbol))
        print(f"{symbol}: {summary} -> {len(bars)} bars, first close stamped "
              f"{pd.Timestamp(bars[0].ts_event, unit='ns', tz='UTC').tz_convert(NEW_YORK)}")


if __name__ == "__main__":
    _main()
