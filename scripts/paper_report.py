"""Summarise a paper-trading journal and check it against the backtest's assumptions.

Usage:
    python scripts/paper_report.py paper/BTCUSDT_1d_10-50.jsonl

Reports sessions, bars, signals and fills; simulated equity; and the key
paper-vs-backtest gap: how far fills landed from the bar close the backtest
fills at, compared with the 5 bps slippage the research assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.paper.journal import read_journal  # noqa: E402

BACKTEST_SLIPPAGE_BPS = 5.0


def summarise(records: list[dict]) -> dict:
    df = pd.DataFrame(records)
    if df.empty:
        return {"records": 0}
    by = {e: df[df["event"] == e] for e in ("start", "warmup", "bar", "fill", "stop")}
    fills = by["fill"]
    bars = by["bar"]
    out: dict = {
        "records": len(df),
        "sessions": len(by["start"]),
        "first_record": df["ts"].iloc[0],
        "last_record": df["ts"].iloc[-1],
        "live_bars": len(bars),
        "signals": int(bars["signal"].notna().sum()) if "signal" in bars else 0,
        "fills": len(fills),
    }
    if not bars.empty:
        out["last_bar_time"] = bars["bar_time"].iloc[-1]
        out["last_close"] = bars["close"].iloc[-1]
        out["last_fast"] = bars["fast"].iloc[-1]
        out["last_slow"] = bars["slow"].iloc[-1]
        out["position"] = bars["position"].iloc[-1]
    stop = by["stop"]
    if not stop.empty and (bars.empty or stop["ts"].iloc[-1] > bars["ts"].iloc[-1]):
        # Bar records are written before that bar's fills arrive; the stop record is final.
        out["position"] = stop["position"].iloc[-1]
    equity = pd.concat([by["start"].get("equity", pd.Series(dtype=float)),
                        bars.get("equity", pd.Series(dtype=float)),
                        by["stop"].get("equity", pd.Series(dtype=float))]).dropna()
    if not equity.empty:
        out["equity_start"] = float(equity.iloc[0])
        out["equity_last"] = float(equity.iloc[-1])
    if not fills.empty:
        slip = fills["slippage_vs_close_bps"].dropna()
        w = fills.loc[slip.index, "qty"]
        out["avg_slippage_bps"] = float((slip * w).sum() / w.sum()) if w.sum() else None
        out["max_slippage_bps"] = float(slip.max())
        spread = (fills["ask"] - fills["bid"]) / ((fills["ask"] + fills["bid"]) / 2) * 1e4
        out["avg_spread_bps"] = float(spread.dropna().mean()) if spread.notna().any() else None
        out["commissions_usdt"] = float(fills["commission"].astype(str).str.split().str[0].astype(float).sum())
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("journal", type=Path)
    args = p.parse_args(argv)
    if not args.journal.exists():
        print(f"No journal at {args.journal}", file=sys.stderr)
        return 1
    s = summarise(read_journal(args.journal))
    for k, v in s.items():
        print(f"{k:>20}: {v:,.4f}" if isinstance(v, float) else f"{k:>20}: {v}")
    if s.get("avg_slippage_bps") is not None:
        verdict = "within" if s["avg_slippage_bps"] <= BACKTEST_SLIPPAGE_BPS else "WORSE than"
        print(f"\nFills averaged {s['avg_slippage_bps']:.2f} bps from the bar close: {verdict} the "
              f"{BACKTEST_SLIPPAGE_BPS:.0f} bps the backtests assumed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
