"""Daily forward test of eight trading rules on US ETFs and stocks (paper, no broker, no orders).

Run once per trading day after the close (after ~16:20 New York time):

    python scripts/forward_test.py                 # refresh Alpaca data, update the forward test
    python scripts/forward_test.py --no-download   # reuse data/raw as is

The first run fixes the start session (the latest completed session) in
``paper/forward_state.json``. Every later run replays every rule in
``src/strategies/signals.py`` (plus the detailed SMA 10/50 strategy) from that
session with fresh data, and reports a leaderboard of the rules, each rule's
position per symbol, and the simulated orders due at the next open. Each run is appended to
``paper/forward_journal.jsonl`` and summarised in ``paper/forward_status.md``.

    python scripts/forward_test.py --reset         # start a new forward test from the latest session

Symbols added later (e.g. a new stock in ``SYMBOL_VENUES``) join the running
forward test with their own start date; existing symbols keep theirs.
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
from src.paper.forward import forward_rules  # noqa: E402
from src.paper.forward import forward_status  # noqa: E402
from src.paper.forward import rule_leaderboard  # noqa: E402
from src.strategies.signals import STRATEGIES  # noqa: E402
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
    head = ("| Symbol | Last session | Close | SMA fast | SMA slow | Trend | Shares | Equity | "
            "Since start | Trades | Next open |")
    lines = [head, "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(
            f"| {r.symbol} | {r.last_session} | {r.close:,.2f} | {r.sma_fast:,.2f} | {r.sma_slow:,.2f} | "
            f"{r.trend} | {r.shares:,.0f} | {r.equity:,.2f} | {r.return_since_start * 100:+.2f}% | "
            f"{r.trades_since_start} | **{r.next_open_action}** |",
        )
    return "\n".join(lines)


def leaderboard_table(board) -> str:
    lines = ["| Rule | Equity | Since start | Holding | Trades | Buys at next open | Sells at next open |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, r in board.iterrows():
        lines.append(
            f"| {STRATEGIES[key][0]} | {r['equity']:,.2f} | {r['return_since_start'] * 100:+.2f}% | "
            f"{int(r['holding'])}/{int(r['symbols'])} | {int(r['trades'])} | {int(r['buys_next_open'])} | "
            f"{int(r['sells_next_open'])} |",
        )
    return "\n".join(lines)


def action_matrix(rule_rows, symbols) -> str:
    """Symbol x rule: what each rule does at the next open (or whether it is holding)."""
    keys = list(STRATEGIES)
    cell = {}
    for r in rule_rows:
        if r.next_open_action != "HOLD":
            cell[(r.symbol, r.rule)] = f"**{r.next_open_action}**"
        else:
            cell[(r.symbol, r.rule)] = "long" if r.long_now else "cash"
    short = {"buy_and_hold": "B&H", "sma_10_50": "10/50", "golden_cross": "50/200", "trend_200": "200d",
             "momentum_12_1": "Mom", "donchian_55_20": "Donch", "macd": "MACD", "rsi2_reversion": "RSI2"}
    lines = ["| Symbol | " + " | ".join(short.get(k, k) for k in keys) + " |",
             "|---|" + "|".join([":-:"] * len(keys)) + "|"]
    for sym in symbols:
        lines.append(f"| {sym} | " + " | ".join(cell.get((sym, k), "") for k in keys) + " |")
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
        added = [s for s in args.symbols if s not in state.symbols]
        if added:
            last = {s: frames[s].index[-1].strftime("%Y-%m-%d") for s in added}
            state = state.with_symbols(args.symbols, last)
            state.save(state_path)
            print(f"Added {', '.join(added)} to the forward test, starting at their latest close.")

    rows = [forward_status(frames[s], s, state) for s in state.symbols]
    total = sum(r.equity for r in rows)
    capital = state.capital_per_symbol * len(rows)

    rule_rows = [r for s in state.symbols for r in forward_rules(frames[s], s, state)]
    board = rule_leaderboard(rule_rows, state.capital_per_symbol)
    (args.paper_dir / "forward_rules.json").write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start_session": state.start_session,
        "capital_per_symbol": state.capital_per_symbol,
        "rule_names": {k: v[0] for k, v in STRATEGIES.items()},
        "leaderboard": board.reset_index().to_dict(orient="records"),
        "statuses": [asdict(r) for r in rule_rows],
    }, indent=2), encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    Journal(args.paper_dir / "forward_journal.jsonl").write(
        "forward_run", start_session=state.start_session, total_equity=total,
        statuses=[{k: v for k, v in asdict(r).items() if k != "fills"} for r in rows],
        rules={k: {"equity": float(v["equity"]), "holding": int(v["holding"])} for k, v in board.iterrows()},
        rule_actions={f"{r.symbol}:{r.rule}": r.next_open_action for r in rule_rows
                      if r.next_open_action != "HOLD"},
    )
    table = status_table(rows)
    md = "\n".join([
        "# Forward test status",
        "",
        f"Updated {now}. Started at the {state.start_session} close. Every rule gets "
        f"{state.capital_per_symbol:,.0f} simulated USD per symbol, long or cash only, decides at the close and "
        f"fills at the next open, {state.cost_rate * 100:.2f}% cost per side. Nothing here is a real order.",
        "",
        "## Rule leaderboard (all symbols)",
        "",
        leaderboard_table(board),
        "",
        "## What each rule does at the next open",
        "",
        "**BUY**/**SELL** = order at the next open; long/cash = current simulated position.",
        "",
        action_matrix(rule_rows, state.symbols),
        "",
        f"## SMA {state.fast}/{state.slow} detail",
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
    print(leaderboard_table(board))
    acts = [f"{r.symbol}:{STRATEGIES[r.rule][0]} {r.next_open_action}" for r in rule_rows if r.next_open_action != "HOLD"]
    print(f"\n{len(acts)} simulated orders at the next open across {len(STRATEGIES)} rules. "
          f"Details: {args.paper_dir / 'forward_status.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
