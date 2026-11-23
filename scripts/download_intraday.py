"""Download regular-hours 1-minute bars from Alpaca into data/raw/intraday/<SYMBOL>_1m.csv.gz.

Uses the same Alpaca keys as scripts/download_data.py (from .env). Market data only.

Usage:
    python scripts/download_intraday.py                         # SPY QQQ IWM NVDA, last ~3 years
    python scripts/download_intraday.py --symbols MSFT --start 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alpaca_data import SYMBOL_PATTERN  # noqa: E402
from src.alpaca_data import open_session  # noqa: E402
from src.alpaca_data import update_minute_csv  # noqa: E402
from src.intraday.data import INTRADAY_DIR  # noqa: E402

DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "NVDA"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    p.add_argument("--start", default="2023-11-01")
    p.add_argument("--end", default=None)
    p.add_argument("--feed", choices=("sip", "iex"), default="sip")
    p.add_argument("--out-dir", type=Path, default=INTRADAY_DIR)
    args = p.parse_args(argv)
    args.symbols = [s.upper() for s in args.symbols]
    if bad := [s for s in args.symbols if not SYMBOL_PATTERN.match(s)]:
        p.error(f"invalid symbol(s): {bad}")
    for v in (args.start, args.end):
        if v is not None:
            try:
                date.fromisoformat(v)
            except ValueError:
                p.error(f"dates must be YYYY-MM-DD, got {v!r}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    failed = []
    with open_session() as session:
        for s in args.symbols:
            t = time.time()
            df = update_minute_csv(session, s, args.out_dir, args.start, args.end, args.feed)
            if df.empty:
                failed.append(s)
                print(f"{s}: no bars returned", file=sys.stderr)
                continue
            print(f"{s}: {len(df):,} regular-hours minute bars, {df['time'].iloc[0]} -> {df['time'].iloc[-1]} "
                  f"({time.time() - t:.0f}s)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
