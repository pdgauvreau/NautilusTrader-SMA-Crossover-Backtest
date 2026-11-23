"""Alpaca Market Data client: daily ETF bars, read-only.

Credentials come from environment variables or a git-ignored ``.env`` file at
the repo root (``ALPACA_API_KEY_ID`` / ``ALPACA_API_SECRET_KEY``). They are
sent only to Alpaca's market-data endpoint and never printed or logged. Only
market data is requested; nothing here can place orders.
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "data" / "raw"
DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"
KEY_VARS = ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY")
SYMBOL_PATTERN = re.compile(r"^[A-Z]{1,5}$")
MAX_GAP_DAYS = 5  # longer than any weekend + holiday; matches validate_daily


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines; existing environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def credentials() -> dict[str, str]:
    load_dotenv()
    missing = [k for k in KEY_VARS if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"Missing {', '.join(missing)}. Create a free Alpaca account, generate API keys "
            f"(paper keys are fine), and put them in {ROOT / '.env'} as:\n"
            "  ALPACA_API_KEY_ID=...\n  ALPACA_API_SECRET_KEY=...",
        )
    return {"APCA-API-KEY-ID": os.environ[KEY_VARS[0]], "APCA-API-SECRET-KEY": os.environ[KEY_VARS[1]]}


def _get_all_bars(session: requests.Session, params: dict, symbol: str, feed: str) -> list[dict]:
    """Follow Alpaca's pagination and return every bar for ``symbol``."""
    rows: list[dict] = []
    while True:
        for attempt in range(5):
            resp = session.get(DATA_URL, params=params, timeout=60)
            if resp.status_code != 429:
                break
            time.sleep(2 ** attempt)  # rate limited: back off
        if resp.status_code in (401, 403):
            raise SystemExit(f"Alpaca rejected the request ({resp.status_code}): check your API keys"
                             + (" or try --feed iex" if feed == "sip" else ""))
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(payload.get("bars", {}).get(symbol, []))
        token = payload.get("next_page_token")
        if not token:
            return rows
        params["page_token"] = token


_RENAME = {"t": "time", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "vw": "vwap",
           "n": "trade_count"}


def fetch_bars(
    session: requests.Session,
    symbol: str,
    start: str,
    end: str | None,
    feed: str,
) -> pd.DataFrame:
    """All daily bars for ``symbol`` between ``start`` and ``end`` (inclusive), following pagination."""
    params = {"symbols": symbol, "timeframe": "1Day", "start": start, "adjustment": "all",
              "feed": feed, "limit": 10000, "sort": "asc"}
    if end:
        params["end"] = end
    rows = _get_all_bars(session, params, symbol, feed)
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "vwap", "trade_count"])
    df = pd.DataFrame(rows).rename(columns=_RENAME)
    # Alpaca stamps daily bars at midnight New York time; keep the session date.
    df["date"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    df = drop_unfinished_session(df)
    return df[["date", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]


def fetch_minute_bars(
    session: requests.Session,
    symbol: str,
    start: str,
    end: str | None,
    feed: str = "sip",
) -> pd.DataFrame:
    """Split/dividend-adjusted 1-minute bars during regular hours (09:30-15:59 New York).

    ``time`` is each bar's start (UTC). Extended-hours bars are dropped, and so
    is today's session until it has closed.
    """
    params = {"symbols": symbol, "timeframe": "1Min", "start": start, "adjustment": "all",
              "feed": feed, "limit": 10000, "sort": "asc"}
    if end:
        params["end"] = end
    rows = _get_all_bars(session, params, symbol, feed)
    cols = ["time", "open", "high", "low", "close", "volume", "vwap", "trade_count"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).rename(columns=_RENAME)
    t = pd.to_datetime(df["time"], utc=True)
    ny = t.dt.tz_convert("America/New_York")
    minutes = ny.dt.hour * 60 + ny.dt.minute
    df = df[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)].copy()
    df["time"] = t[df.index]
    df["date"] = ny[df.index].dt.date.astype(str)
    df = drop_unfinished_session(df)
    return df[cols].reset_index(drop=True)


def update_minute_csv(session: requests.Session, symbol: str, out_dir: Path, start: str,
                      end: str | None = None, feed: str = "sip") -> pd.DataFrame:
    """Download minute bars and write ``<out_dir>/<SYMBOL>_1m.csv.gz`` atomically."""
    df = fetch_minute_bars(session, symbol, start, end, feed)
    if df.empty:
        return df
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_1m.csv.gz"
    tmp = out_dir / f"{symbol}_1m.part.csv.gz"
    df.to_csv(tmp, index=False, compression="gzip")
    tmp.replace(path)
    return df


def drop_unfinished_session(df: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Remove today's bar unless the session is over (16:20 New York, after the closing auction).

    Pulling data during market hours returns a still-forming daily bar; a
    strategy must never decide on it.
    """
    now_ny = (now or pd.Timestamp.now(tz="UTC")).tz_convert("America/New_York")
    today = now_ny.date().isoformat()
    finished = now_ny.hour * 60 + now_ny.minute >= 16 * 60 + 20
    if finished:
        return df[df["date"] <= today]
    return df[df["date"] < today]


def update_csv(session: requests.Session, symbol: str, raw_dir: Path, start: str, end: str | None = None,
               feed: str = "sip") -> pd.DataFrame:
    """Download ``symbol`` from ``start`` and (re)write ``<raw_dir>/<SYMBOL>_1d.csv`` atomically.

    The full history is re-fetched each time because adjusted prices change
    retroactively after every dividend or split. If trading was halted for a
    long stretch (e.g. a multi-year suspension), only the data after
    the last such gap is kept: earlier prices describe a different situation.
    """
    df = fetch_bars(session, symbol, start, end, feed)
    if df.empty:
        return df
    df, dropped = trim_before_last_gap(df)
    if dropped:
        print(f"  {symbol}: trading gap before {df['date'].iloc[0]}; dropped {dropped} earlier sessions")
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{symbol}_1d.csv"
    tmp = path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return df


def trim_before_last_gap(df: pd.DataFrame, max_gap_days: int = MAX_GAP_DAYS) -> tuple[pd.DataFrame, int]:
    """Keep only rows after the last gap longer than ``max_gap_days`` calendar days."""
    dates = pd.to_datetime(df["date"])
    gaps = dates.diff().dt.days
    big = gaps[gaps > max_gap_days]
    if big.empty:
        return df, 0
    first = big.index[-1]
    pos = df.index.get_loc(first)
    return df.iloc[pos:].reset_index(drop=True), pos


def open_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(credentials())
    session.headers["User-Agent"] = "etf-research-downloader/1.0"
    return session
