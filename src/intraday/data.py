"""Regular-hours 1-minute bars for day-trading research.

Rows are indexed by each bar's **start** time (UTC) and carry a ``session``
column (the New York trading date) and ``minute`` (minutes since midnight New
York, so 09:30 = 570). A bar is only complete at its end, so Nautilus bars are
stamped at ``start + 1 minute``, and a decision made on a bar's close can fill
no earlier than the next bar's open.

NYSE closes at 13:00 on a few days a year. Alpaca still reports trades after
13:00 on those days (extended hours), so they are removed here:

* the day after US Thanksgiving,
* July 3 and December 24 when they fall on a trading day.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from nautilus_trader.model import Bar
from nautilus_trader.model import BarType
from nautilus_trader.model import Equity
from nautilus_trader.model import Quantity
from nautilus_trader.model import QuoteTick

from src.data_loader import SYMBOL_VENUES
from src.data_loader import _check_symbol

NEW_YORK = "America/New_York"
OPEN_MINUTE = 9 * 60 + 30
CLOSE_MINUTE = 16 * 60
EARLY_CLOSE_MINUTE = 13 * 60
INTRADAY_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "intraday"


def is_early_close(d: date) -> bool:
    """NYSE 13:00 early-close days (only applied to days that actually traded)."""
    if d.month == 7 and d.day == 3:
        return True
    if d.month == 12 and d.day == 24:
        return True
    if d.month == 11 and d.weekday() == 4:  # Friday after the 4th Thursday of November
        thanksgiving = d.day - 1
        return 22 <= thanksgiving <= 28
    return False


def minute_csv_path(raw_dir: str | Path, symbol: str) -> Path:
    return Path(raw_dir) / f"{_check_symbol(symbol)}_1m.csv.gz"


def load_minute_bars(raw_dir: str | Path = INTRADAY_DIR, symbol: str = "SPY",
                     start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Load ``<SYMBOL>_1m.csv.gz`` (from ``scripts/download_intraday.py``), regular hours only."""
    path = minute_csv_path(raw_dir, symbol)
    if not path.exists():
        raise FileNotFoundError(f"No minute data for {symbol} at {path}; run scripts/download_intraday.py")
    raw = pd.read_csv(path)
    t = pd.to_datetime(raw["time"], utc=True)
    ny = t.dt.tz_convert(NEW_YORK)
    df = pd.DataFrame({c: pd.to_numeric(raw[c]).astype("float64") for c in ("open", "high", "low", "close", "volume")})
    df.index = pd.DatetimeIndex(t, name="start")
    df["session"] = ny.dt.date.to_numpy()
    df["minute"] = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    early = df["session"].map(is_early_close)
    close_minute = early.map({True: EARLY_CLOSE_MINUTE, False: CLOSE_MINUTE})
    df = df[(df["minute"] >= OPEN_MINUTE) & (df["minute"] < close_minute)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if start:
        df = df[df["session"] >= date.fromisoformat(start)]
    if end:
        df = df[df["session"] <= date.fromisoformat(end)]
    return df


def validate_minutes(df: pd.DataFrame) -> dict[str, object]:
    """Basic integrity checks; raises ``ValueError`` on failure."""
    if df.empty:
        raise ValueError("no minute bars")
    if not df.index.is_unique or not df.index.is_monotonic_increasing:
        raise ValueError("minute index must be unique and sorted")
    per_day = df.groupby("session").size()
    if (per_day > 390).any():
        raise ValueError(f"more than 390 regular-hours bars on {per_day[per_day > 390].index.tolist()}")
    bad = df[(df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) | (df["low"] > df[["open", "close"]].min(axis=1) + 1e-9)]
    if not bad.empty:
        raise ValueError(f"OHLC inconsistent at {bad.index[:5].tolist()}")
    return {"bars": len(df), "sessions": len(per_day), "first": str(per_day.index[0]),
            "last": str(per_day.index[-1]), "median_bars_per_session": int(per_day.median()),
            "min_bars_per_session": int(per_day.min())}


def minute_bar_type(symbol: str) -> BarType:
    return BarType.from_str(f"{symbol}.{SYMBOL_VENUES[_check_symbol(symbol)]}-1-MINUTE-LAST-EXTERNAL")


def to_nautilus_minute_bars(df: pd.DataFrame, instrument: Equity, bar_type: BarType) -> list[Bar]:
    """Minute bars stamped at their end (start + 60 s)."""
    ts = (df.index.asi8 // 1 if df.index.unit == "ns" else df.index.as_unit("ns").asi8) + 60_000_000_000
    mp, o, h, lo, c, v = (instrument.make_price, df["open"].to_numpy(), df["high"].to_numpy(),
                          df["low"].to_numpy(), df["close"].to_numpy(), df["volume"].to_numpy())
    return [Bar(bar_type, mp(o[i]), mp(h[i]), mp(lo[i]), mp(c[i]), Quantity(max(v[i], 0.0), 0), int(ts[i]), int(ts[i]))
            for i in range(len(df))]


def bar_end_ns(df: pd.DataFrame) -> list[int]:
    """Each bar's end timestamp in ns (what strategies see as ``bar.ts_event``)."""
    idx = df.index if df.index.unit == "ns" else df.index.as_unit("ns")
    return (idx.asi8 + 60_000_000_000).tolist()


def quotes_at(df: pd.DataFrame, positions: list[int], instrument: Equity, half_spread: float) -> list[QuoteTick]:
    """Quotes at the *open* of the given bar positions (1 s after the bar starts).

    The 1-second offset puts each quote strictly after the previous bar's
    close event, so a decision on that close is executed against this quote.
    """
    idx = df.index if df.index.unit == "ns" else df.index.as_unit("ns")
    starts, opens = idx.asi8, df["open"].to_numpy()
    out = []
    for i in sorted(set(positions)):
        o = float(opens[i])
        bid, ask = instrument.make_price(o * (1 - half_spread)), instrument.make_price(o * (1 + half_spread))
        if ask.as_double() <= bid.as_double():
            ask = instrument.make_price(bid.as_double() + 10 ** -instrument.price_precision)
        ts = int(starts[i]) + 1_000_000_000
        out.append(QuoteTick(instrument.id, bid, ask, Quantity(10_000_000, 0), Quantity(10_000_000, 0), ts, ts))
    return out
