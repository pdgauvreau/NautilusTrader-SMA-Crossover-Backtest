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
    rows: list[dict] = []
    while True:
        for attempt in range(4):
            resp = session.get(DATA_URL, params=params, timeout=30)
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
            break
        params["page_token"] = token
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "vwap", "trade_count"])
    df = pd.DataFrame(rows).rename(columns={"t": "time", "o": "open", "h": "high", "l": "low", "c": "close",
                                            "v": "volume", "vw": "vwap", "n": "trade_count"})
    # Alpaca stamps daily bars at midnight New York time; keep the session date.
    df["date"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York").dt.date.astype(str)
    df = drop_unfinished_session(df)
    return df[["date", "open", "high", "low", "close", "volume", "vwap", "trade_count"]]


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
    retroactively after every dividend or split.
    """
    df = fetch_bars(session, symbol, start, end, feed)
    if df.empty:
        return df
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{symbol}_1d.csv"
    tmp = path.with_suffix(".csv.part")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return df


def open_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(credentials())
    session.headers["User-Agent"] = "etf-research-downloader/1.0"
    return session
