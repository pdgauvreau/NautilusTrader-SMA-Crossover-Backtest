"""Daily forward test of the SMA 10/50 ETF strategy (paper trading, no broker, no orders).

Run once per trading day after the close (after ~16:20 New York time):

    python scripts/forward_test.py                 # refresh Alpaca data, update the forward test
    python scripts/forward_test.py --no-download   # reuse data/raw as is

The first run fixes the start session (the latest completed session) in
``paper/forward_state.json``. Every later run replays the strategy from that
session with fresh data and reports, per ETF: trend, position, equity since
the start, and the action due at the next open. Each run is appended to
``paper/forward_journal.jsonl`` and summarised in ``paper/forward_status.md``.

    python scripts/forward_test.py --reset         # start a new forward test from the latest session
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from datetime import timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

from src.data_loader import DEFAULT_COST_RATE  # noqa: E402
from src.data_loader import DEFAULT_SYMBOLS  # noqa: E402
from src.data_loader import load_bars  # noqa: E402
from src.data_loader import validate_daily  # noqa: E402
from src.paper.forward import ForwardState  # noqa: E402
from src.paper.forward import forward_status  # noqa: E402
from src.paper.journal import Journal  # noqa: E402

PAPER_DIR = ROOT / "paper"
HISTORY_START = "2016-01-01"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--fast", type=int, default=10)
    p.add_argument("--slow", type=int, default=50)
    p.add_argument("--capital", type=float, default=10_000.0, help="simulated USD per ETF")
    p.add_argument("--cost", type=float, default=DEFAULT_COST_RATE, help="cost per side (fraction)")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    p.add_argument("--no-download", action="store_true", help="skip refreshing data from Alpaca")
    p.add_argument("--reset", action="store_true", help="start a new forward test from the latest session")
    return p.parse_args(argv)


def refresh_data(symbols: list[str], raw_dir: Path) -> None:
    from src.alpaca_data import open_session
    from src.alpaca_data import update_csv

    with open_session() as session:
        for s in symbols:
            df = update_csv(session, s, raw_dir, HISTORY_START)
            print(f"  {s}: {len(df)} sessions, latest {df['date'].iloc[-1] if len(df) else 'n/a'}")


def status_table(rows) -> str:
    head = ("| ETF | Last session | Close | SMA fast | SMA slow | Trend | Shares | Equity | "
            "Since start | Trades | Next open |")
    lines = [head, "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r.symbol} | {r.last_session} | {r.close:,.2f} | {r.sma_fast:,.2f} | {r.sma_slow:,.2f} | "
            f"{r.trend} | {r.shares:,.0f} | {r.equity:,.2f} | {r.return_since_start * 100:+.2f}% | "
            f"{r.trades_since_start} | **{r.next_open_action}** |",
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    state_path = args.paper_dir / "forward_state.json"

    if not args.no_download:
        print("Refreshing data from Alpaca...")
        refresh_data(args.symbols, args.raw_dir)

    frames = {}
    for s in args.symbols:
        df = load_bars(args.raw_dir, s)
        validate_daily(df)
        frames[s] = df
    latest = min(df.index[-1] for df in frames.values()).strftime("%Y-%m-%d")

    if args.reset or not state_path.exists():
        state = ForwardState.new(latest, args.symbols, fast=args.fast, slow=args.slow,
                                 capital_per_symbol=args.capital, cost_rate=args.cost)
        state.save(state_path)
        print(f"Started a new forward test from the {latest} close ({state_path}).")
    else:
        state = ForwardState.load(state_path)

    rows = [forward_status(frames[s], s, state) for s in state.symbols]
    total = sum(r.equity for r in rows)
    capital = state.capital_per_symbol * len(rows)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Journal(args.paper_dir / "forward_journal.jsonl").write(
        "forward_run", start_session=state.start_session, total_equity=total,
        statuses=[{k: v for k, v in asdict(r).items() if k != "fills"} for r in rows],
    )
    table = status_table(rows)
    md = "\n".join([
        "# Forward test status",
        "",
        f"Updated {now}. Started at the {state.start_session} close with "
        f"{state.capital_per_symbol:,.0f} simulated USD per ETF. Strategy: SMA {state.fast}/{state.slow}, "
        f"long only, decide at the close, fill at the next open, {state.cost_rate * 100:.2f}% cost per side.",
        "",
        table,
        "",
        f"**Total equity:** {total:,.2f} USD ({(total / capital - 1) * 100:+.2f}% since start)",
        "",
        "## Fills since start",
        "",
        *(f"- {r.symbol}: {f['time']} {f['side']} {f['qty']:.0f} @ {f['avg_px']:.2f}"
          for r in rows for f in r.fills),
        "" if any(r.fills for r in rows) else "_None yet._",
    ])
    (args.paper_dir / "forward_status.md").write_text(md, encoding="utf-8")
    print(table)
    print(f"\nTotal equity {total:,.2f} USD ({(total / capital - 1) * 100:+.2f}% since {state.start_session}).")
    print(json.dumps({r.symbol: r.next_open_action for r in rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
