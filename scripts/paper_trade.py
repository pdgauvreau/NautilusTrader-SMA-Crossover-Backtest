"""Paper-trade the SMA crossover on live public Binance data with simulated fills.

No API keys, no exchange orders: execution is Nautilus's in-process sandbox.
Every bar, signal and fill is appended to a JSONL journal under paper/.

Examples:
    # The real strategy: daily bars, 10/50 SMAs. Leave it running; one bar per day at 00:00 UTC.
    python scripts/paper_trade.py

    # Smoke test: 1-minute bars, short SMAs, stop after 10 minutes.
    python scripts/paper_trade.py --interval 1m --fast 3 --slow 8 --duration 600

Stop with Ctrl+C; open positions are closed (on paper) and the session summary
is journaled. Inspect a journal with: python scripts/paper_report.py <journal>
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from datetime import datetime
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_loader import bar_type_for  # noqa: E402
from src.paper.node import REGIONS  # noqa: E402
from src.paper.node import PaperSettings  # noqa: E402
from src.paper.node import build_paper_node  # noqa: E402
from src.paper.strategy import PaperSmaCrossover  # noqa: E402
from src.paper.strategy import PaperSmaCrossoverConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", default="1d", help="kline interval: 1m, 5m, 1h, 4h, 1d ...")
    p.add_argument("--fast", type=int, default=10)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--fraction", type=float, default=0.99, help="share of cash per entry")
    p.add_argument("--capital", type=float, default=10_000.0, help="simulated starting USDT")
    p.add_argument("--fee", type=float, default=0.001, help="simulated fee per side")
    p.add_argument("--region", choices=REGIONS, default="us",
                   help="us = Binance.US public data (api.binance.com is geo-restricted here)")
    p.add_argument("--bar-fills", action="store_true",
                   help="fill at the bar close like the backtest, instead of the live bid/ask")
    p.add_argument("--no-join-trend", action="store_true",
                   help="wait for a fresh cross instead of entering if fast > slow at start")
    p.add_argument("--duration", type=float, default=None, help="stop after N seconds (default: run until Ctrl+C)")
    p.add_argument("--journal", type=Path, default=None, help="default: paper/<symbol>_<interval>_<fast>-<slow>.jsonl")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    journal = args.journal or ROOT / "paper" / f"{args.symbol}_{args.interval}_{args.fast}-{args.slow}.jsonl"
    settings = PaperSettings(symbol=args.symbol, region=args.region, starting_usdt=args.capital, fee_rate=args.fee,
                             quote_fills=not args.bar_fills)
    config = PaperSmaCrossoverConfig(
        journal_path=str(journal),
        bar_type=bar_type_for(args.symbol, args.interval),
        fast_period=args.fast,
        slow_period=args.slow,
        position_fraction=args.fraction,
        enter_if_above_on_start=not args.no_join_trend,
        subscribe_quotes=not args.bar_fills,
    )

    node = build_paper_node(settings)
    strategy = PaperSmaCrossover(config)
    node.add_strategy(strategy)
    handle = node.handle()

    def stop(*_: object) -> None:
        if not handle.is_stopping:
            print("\nStopping paper session...", flush=True)
            handle.stop()

    signal.signal(signal.SIGINT, stop)
    if args.duration:
        def timer() -> None:
            time.sleep(args.duration)
            stop()

        threading.Thread(target=timer, daemon=True).start()

    print(
        f"Paper trading {config.bar_type} | SMA {args.fast}/{args.slow} | {args.capital:,.0f} simulated USDT | "
        f"Binance{'.US' if args.region == 'us' else ''} public data | sandbox fills\n"
        f"Journal: {journal}\nStarted {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC. Ctrl+C to stop.",
        flush=True,
    )
    try:
        node.run()
    finally:
        node.dispose()
    print(f"Session ended: {strategy.bars_seen} live bars, {len(strategy.signals)} signals. Journal: {journal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
