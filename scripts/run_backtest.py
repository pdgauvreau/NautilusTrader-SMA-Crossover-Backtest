"""Run the 10/50 SMA crossover backtest on BTCUSDT daily bars and write a report.

Usage:
    python scripts/download_data.py          # once, fetches data/raw/
    python scripts/run_backtest.py           # writes reports/

Outputs (in ``--report-dir``):
    summary.md         strategy vs buy-and-hold statistics table
    metrics.json       the same numbers, machine-readable
    trades.csv         one row per closed round trip
    equity_curve.png   equity + drawdown chart
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from nautilus_trader.model import BarType  # noqa: E402

from src.backtest import BacktestOutput  # noqa: E402
from src.backtest import realized_pnls  # noqa: E402
from src.backtest import run_backtest  # noqa: E402
from src.data_loader import DEFAULT_BAR_TYPE  # noqa: E402
from src.data_loader import TAKER_FEE  # noqa: E402
from src.data_loader import btcusdt_instrument  # noqa: E402
from src.data_loader import load_klines_csv  # noqa: E402
from src.data_loader import to_nautilus_bars  # noqa: E402
from src.data_loader import validate_daily  # noqa: E402
from src.metrics import PerformanceStats  # noqa: E402
from src.metrics import buy_and_hold_equity  # noqa: E402
from src.metrics import compute_stats  # noqa: E402
from src.strategies.sma_crossover import Signal  # noqa: E402
from src.strategies.sma_crossover import SmaCrossoverConfig  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--fast", type=int, default=10, help="fast SMA period (bars)")
    p.add_argument("--slow", type=int, default=50, help="slow SMA period (bars)")
    p.add_argument("--trade-size", type=float, default=0.5, help="fixed BTC quantity per entry")
    p.add_argument("--capital", type=float, default=100_000.0, help="starting USDT balance")
    return p.parse_args(argv)


def _fmt_pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:+.2f}%"


def _fmt_num(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.2f}"


def summary_table(strategy: PerformanceStats, bnh: PerformanceStats) -> str:
    rows = [
        ("Total return", _fmt_pct(strategy.total_return), _fmt_pct(bnh.total_return)),
        ("Final equity (USDT)", f"{strategy.final_equity:,.2f}", f"{bnh.final_equity:,.2f}"),
        ("Number of trades", str(strategy.num_trades), str(bnh.num_trades)),
        ("Win rate", _fmt_pct(strategy.win_rate).lstrip("+"), _fmt_pct(bnh.win_rate).lstrip("+")),
        ("Max drawdown", _fmt_pct(strategy.max_drawdown), _fmt_pct(bnh.max_drawdown)),
        ("Sharpe ratio (daily, ann. 365)", _fmt_num(strategy.sharpe_ratio), _fmt_num(bnh.sharpe_ratio)),
    ]
    lines = ["| Metric | SMA 10/50 crossover | Buy & hold BTC |", "|---|---:|---:|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def trades_table(out: BacktestOutput) -> pd.DataFrame:
    pos = out.positions
    if pos.empty:
        return pd.DataFrame(columns=["opened", "closed", "entry_px", "exit_px", "quantity", "net_pnl_usdt", "return"])
    return pd.DataFrame(
        {
            "opened": pd.to_datetime(pos["ts_opened"], utc=True).dt.date,
            "closed": pd.to_datetime(pos["ts_closed"], utc=True).dt.date,
            "entry_px": pos["avg_px_open"].astype(float).round(2),
            "exit_px": pos["avg_px_close"].astype(float).round(2),
            "quantity": pos["peak_qty"].astype(str).str.split().str[0].astype(float),
            "net_pnl_usdt": [round(p, 2) for p in realized_pnls(pos)],
            "return": pos["realized_return"].astype(float).round(4),
        },
    ).sort_values("opened").reset_index(drop=True)


def _markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub markdown table (no tabulate dependency)."""
    def cell(v: object) -> str:
        return f"{v:,.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_equity(
    equity: pd.Series,
    bnh: pd.Series,
    closes: pd.Series,
    out: BacktestOutput,
    fast: int,
    slow: int,
    path: Path,
) -> None:
    fig, (ax_eq, ax_px, ax_dd) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.2]},
    )
    ax_eq.plot(equity.index, equity.values, label=f"SMA {fast}/{slow} crossover", color="#1f6feb", lw=1.8)
    ax_eq.plot(bnh.index, bnh.values, label="Buy & hold BTC", color="#8b949e", lw=1.4, ls="--")
    ax_eq.axhline(out.starting_capital, color="black", lw=0.6, alpha=0.4)
    ax_eq.set_ylabel("Equity (USDT)")
    ax_eq.set_title("BTCUSDT daily — SMA crossover vs buy & hold (Binance spot fees incl.)")
    ax_eq.legend(loc="upper left", frameon=False)
    ax_eq.grid(alpha=0.25)

    ax_px.plot(closes.index, closes.values, color="black", lw=1.0, label="BTCUSDT close")
    ax_px.plot(closes.index, closes.rolling(fast).mean(), color="#d29922", lw=1.0, label=f"SMA {fast}")
    ax_px.plot(closes.index, closes.rolling(slow).mean(), color="#8250df", lw=1.0, label=f"SMA {slow}")
    for ts_ns, sig in out.signals:
        ts = pd.Timestamp(ts_ns, tz="UTC")
        px = closes.asof(ts)
        if sig is Signal.ENTER_LONG:
            ax_px.scatter(ts, px, marker="^", color="#1a7f37", s=70, zorder=5)
        else:
            ax_px.scatter(ts, px, marker="v", color="#cf222e", s=70, zorder=5)
    ax_px.set_ylabel("Price (USDT)")
    ax_px.legend(loc="upper left", frameon=False, ncol=3, fontsize=8)
    ax_px.grid(alpha=0.25)

    dd = equity / equity.cummax() - 1.0
    dd_bnh = bnh / bnh.cummax() - 1.0
    ax_dd.fill_between(dd.index, dd.values * 100, 0, color="#1f6feb", alpha=0.35, label="Strategy")
    ax_dd.plot(dd_bnh.index, dd_bnh.values * 100, color="#8b949e", lw=1.0, ls="--", label="Buy & hold")
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.legend(loc="lower left", frameon=False, fontsize=8)
    ax_dd.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    df = load_klines_csv(args.raw_dir, start=args.start, end=args.end)
    if df.empty:
        print(f"No data in {args.raw_dir}. Run scripts/download_data.py first.", file=sys.stderr)
        return 1
    data_summary = validate_daily(df)

    instrument = btcusdt_instrument()
    bar_type = BarType.from_str(DEFAULT_BAR_TYPE)
    bars = to_nautilus_bars(df, instrument, bar_type)
    config = SmaCrossoverConfig(
        bar_type=bar_type,
        trade_size=args.trade_size,
        fast_period=args.fast,
        slow_period=args.slow,
    )

    out = run_backtest(bars, instrument, config, starting_capital=args.capital)
    if out.bars_seen != len(bars):
        # Nautilus logs handler exceptions instead of raising; fail loudly here.
        raise RuntimeError(f"Strategy saw {out.bars_seen} of {len(bars)} bars")

    # Bar-close-stamped closes, aligned with the equity curve's timestamps.
    closes = pd.Series(
        [b.close.as_double() for b in bars],
        index=pd.to_datetime([b.ts_event for b in bars], utc=True),
        name="close",
    )
    equity = out.equity.reindex(closes.index).ffill().fillna(args.capital)
    bnh = buy_and_hold_equity(closes, args.capital, float(TAKER_FEE))

    pnls = realized_pnls(out.positions)
    strat_stats = compute_stats(equity, pnls, args.capital)
    bnh_pnl = float(bnh.iloc[-1]) - args.capital
    bnh_stats = compute_stats(bnh, [bnh_pnl], args.capital)

    trades = trades_table(out)
    trades.to_csv(args.report_dir / "trades.csv", index=False)
    plot_equity(equity, bnh, closes, out, args.fast, args.slow, args.report_dir / "equity_curve.png")

    table = summary_table(strat_stats, bnh_stats)
    period = f"{data_summary['first']} to {data_summary['last']}"
    md = "\n".join(
        [
            "# Backtest summary",
            "",
            f"- Period: {period} ({data_summary['rows']} daily bars, BTCUSDT Binance spot)",
            f"- Strategy: long {args.trade_size} BTC when SMA{args.fast} crosses above SMA{args.slow}; flat on cross below",
            f"- Starting capital: {args.capital:,.0f} USDT; fees: {float(TAKER_FEE) * 100:.1f}% taker per side",
            "- Buy & hold: all capital into BTC at the first close, sold at the last close (fees on both sides)",
            "- Fills: market orders at the close of the signal bar; open position closed at the final bar",
            "",
            table,
            "",
            "## Trades",
            "",
            _markdown(trades) if not trades.empty else "_No trades._",
            "",
        ],
    )
    (args.report_dir / "summary.md").write_text(md, encoding="utf-8")
    (args.report_dir / "metrics.json").write_text(
        json.dumps(
            {
                "period": period,
                "parameters": {
                    "fast": args.fast,
                    "slow": args.slow,
                    "trade_size_btc": args.trade_size,
                    "starting_capital_usdt": args.capital,
                    "taker_fee": float(TAKER_FEE),
                },
                "strategy": strat_stats.as_dict(),
                "buy_and_hold": bnh_stats.as_dict(),
            },
            indent=2,
            default=lambda x: None,
        ).replace("NaN", "null"),
        encoding="utf-8",
    )

    print(f"Period: {period}")
    print(table)
    print(f"\nReports written to {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
