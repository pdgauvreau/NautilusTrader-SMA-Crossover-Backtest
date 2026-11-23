"""Download daily ETF bars from Alpaca's Market Data API into data/raw/<SYMBOL>_1d.csv.

Needs a free Alpaca account. Put your keys in a ``.env`` file at the repo root
(git-ignored; see ``.env.example``) or in environment variables. They are never
printed::

    ALPACA_API_KEY_ID=...
    ALPACA_API_SECRET_KEY=...

Paper-trading keys work for market data too. Only market data is read; this
script cannot place orders.

Prices are split- and dividend-adjusted (``adjustment=all``) so returns include
dividends. ``--feed sip`` (consolidated tape) is the default; use ``--feed iex``
if your plan rejects SIP.

Usage:
    python scripts/download_data.py                       # SPY QQQ IWM, 2016-01-01 to latest
    python scripts/download_data.py --symbols SPY --start 2020-01-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alpaca_data import SYMBOL_PATTERN  # noqa: E402
from src.alpaca_data import open_session  # noqa: E402
from src.alpaca_data import update_csv  # noqa: E402
from src.data_loader import DEFAULT_SYMBOLS  # noqa: E402

DEFAULT_OUT_DIR = ROOT / "data" / "raw"
DEFAULT_START = "2016-01-01"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--start", default=DEFAULT_START, help="first session date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="last session date (default: latest available)")
    p.add_argument("--feed", choices=("sip", "iex"), default="sip")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args(argv)
    args.symbols = [s.upper() for s in args.symbols]
    bad = [s for s in args.symbols if not SYMBOL_PATTERN.match(s)]
    if bad:
        p.error(f"invalid symbol(s): {bad}")
    for value in (args.start, args.end):
        if value is not None:
            try:
                date.fromisoformat(value)
            except ValueError:
                p.error(f"dates must be YYYY-MM-DD, got {value!r}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failed = []
    with open_session() as session:
        for symbol in args.symbols:
            df = update_csv(session, symbol, args.out_dir, args.start, args.end, args.feed)
            if df.empty:
                print(f"{symbol}: no bars returned", file=sys.stderr)
                failed.append(symbol)
                continue
            print(f"{symbol}: {len(df)} sessions {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
