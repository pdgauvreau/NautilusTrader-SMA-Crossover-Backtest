"""Run the SMA crossover backtest on one or more ETFs and write a report.

Usage:
    python scripts/download_data.py                 # once: fetches data/raw/ from Alpaca
    python scripts/run_backtest.py                  # SPY, QQQ, IWM; writes reports/

Outputs (in ``--report-dir``), per ETF:
    <SYMBOL>_summary.md         strategy vs buy-and-hold statistics + trades
    <SYMBOL>_metrics.json       the same numbers, machine-readable
    <SYMBOL>_trades.csv         one row per closed round trip
    <SYMBOL>_equity_curve.png   equity, price with SMAs and signals, drawdown
plus summary.md comparing all ETFs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data_loader import DEFAULT_COST_RATE  # noqa: E402
from src.data_loader import DEFAULT_HALF_SPREAD  # noqa: E402
from src.data_loader import DEFAULT_SYMBOLS  # noqa: E402
from src.data_loader import load_bars  # noqa: E402
from src.data_loader import validate_daily  # noqa: E402
from src.metrics import PerformanceStats  # noqa: E402
from src.pipeline import PipelineResult  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.strategies.sma_crossover import Signal  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports")
    p.add_argument("--start", default=None, help="first session YYYY-MM-DD (default: all data)")
    p.add_argument("--end", default=None, help="last session YYYY-MM-DD (default: all data)")
    p.add_argument("--fast", type=int, default=10, help="fast SMA period (sessions)")
    p.add_argument("--slow", type=int, default=50, help="slow SMA period (sessions)")
    p.add_argument("--fraction", type=float, default=0.99, help="share of cash invested per entry")
    p.add_argument("--capital", type=float, default=100_000.0, help="starting USD")
    p.add_argument("--cost", type=float, default=DEFAULT_COST_RATE, help="cost per side (fraction of notional)")
    return p.parse_args(argv)


def _fmt_pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x * 100:+.2f}%"


def _fmt_num(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.2f}"


def summary_table(strategy: PerformanceStats, bnh: PerformanceStats, symbol: str = "") -> str:
    rows = [
        ("Total return", _fmt_pct(strategy.total_return), _fmt_pct(bnh.total_return)),
        ("Final equity (USD)", f"{strategy.final_equity:,.2f}", f"{bnh.final_equity:,.2f}"),
        ("Number of trades", str(strategy.num_trades), str(bnh.num_trades)),
        ("Win rate", _fmt_pct(strategy.win_rate).lstrip("+"), _fmt_pct(bnh.win_rate).lstrip("+")),
        ("Max drawdown", _fmt_pct(strategy.max_drawdown), _fmt_pct(bnh.max_drawdown)),
        ("Sharpe ratio (daily, ann. 252)", _fmt_num(strategy.sharpe_ratio), _fmt_num(bnh.sharpe_ratio)),
    ]
    lines = [f"| Metric | SMA crossover | Buy & hold {symbol} |", "|---|---:|---:|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    return "\n".join(lines)


def _markdown(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub markdown table (no tabulate dependency)."""
    def cell(v: object) -> str:
        return f"{v:,.4f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_equity(res: PipelineResult, symbol: str, fast: int, slow: int, path: Path) -> None:
    equity, bnh, closes, out = res.equity, res.buy_and_hold, res.closes, res.output
    fig, (ax_eq, ax_px, ax_dd) = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.2]},
    )
    ax_eq.plot(equity.index, equity.values, label=f"SMA {fast}/{slow} crossover", color="#1f6feb", lw=1.8)
    ax_eq.plot(bnh.index, bnh.values, label=f"Buy & hold {symbol}", color="#8b949e", lw=1.4, ls="--")
    ax_eq.axhline(out.starting_capital, color="black", lw=0.6, alpha=0.4)
    ax_eq.set_ylabel("Equity (USD)")
    ax_eq.set_title(f"{symbol} daily: SMA {fast}/{slow} crossover vs buy & hold (next-open fills, costs incl.)")
    ax_eq.legend(loc="upper left", frameon=False)
    ax_eq.grid(alpha=0.25)

    ax_px.plot(closes.index, closes.values, color="black", lw=1.0, label=f"{symbol} close")
    ax_px.plot(closes.index, closes.rolling(fast).mean(), color="#d29922", lw=1.0, label=f"SMA {fast}")
    ax_px.plot(closes.index, closes.rolling(slow).mean(), color="#8250df", lw=1.0, label=f"SMA {slow}")
    for ts_ns, sig in out.signals:
        ts = pd.Timestamp(ts_ns, unit="ns", tz="UTC")
        marker, color = ("^", "#1a7f37") if sig is Signal.ENTER_LONG else ("v", "#cf222e")
        ax_px.scatter(ts, closes.asof(ts), marker=marker, color=color, s=50, zorder=5)
    ax_px.set_ylabel("Price (USD, adjusted)")
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


def run_symbol(symbol: str, args: argparse.Namespace) -> dict | None:
    try:
        df = load_bars(args.raw_dir, symbol, start=args.start, end=args.end)
    except FileNotFoundError:
        print(f"{symbol}: no data in {args.raw_dir}. Run scripts/download_data.py first.", file=sys.stderr)
        return None
    if df.empty:
        print(f"{symbol}: no sessions in the selected range.", file=sys.stderr)
        return None
    data = validate_daily(df)

    res = run_pipeline(df, args.fast, args.slow, capital=args.capital, symbol=symbol,
                       position_fraction=args.fraction, cost_rate=args.cost)
    out_dir = args.report_dir
    res.trades.to_csv(out_dir / f"{symbol}_trades.csv", index=False)
    plot_equity(res, symbol, args.fast, args.slow, out_dir / f"{symbol}_equity_curve.png")

    period = f"{data['first']} to {data['last']}"
    table = summary_table(res.strategy_stats, res.buy_and_hold_stats, symbol)
    md = "\n".join([
        f"# {symbol} backtest summary",
        "",
        f"- Period: {period} ({data['rows']} trading sessions, split/dividend-adjusted)",
        f"- Strategy: invest {args.fraction:.0%} of cash in {symbol} when SMA{args.fast} crosses above "
        f"SMA{args.slow}; sell to cash on the cross below",
        f"- Execution: decided on the close, filled at the next open ({DEFAULT_HALF_SPREAD * 1e4:.0f} bp "
        f"half-spread) plus {args.cost * 100:.2f}% cost per side; open position closed at the final close",
        f"- Buy & hold: all capital into {symbol} at the first close, sold at the last close (costs both sides)",
        "",
        table,
        "",
        "## Trades",
        "",
        _markdown(res.trades) if not res.trades.empty else "_No trades._",
        "",
    ])
    (out_dir / f"{symbol}_summary.md").write_text(md, encoding="utf-8")
    metrics = {
        "symbol": symbol,
        "period": period,
        "parameters": {"fast": args.fast, "slow": args.slow, "position_fraction": args.fraction,
                       "starting_capital_usd": args.capital, "cost_per_side": args.cost},
        "strategy": res.strategy_stats.as_dict(),
        "buy_and_hold": res.buy_and_hold_stats.as_dict(),
    }
    (out_dir / f"{symbol}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=lambda x: None).replace("NaN", "null"), encoding="utf-8",
    )
    print(f"\n{symbol}: {period}\n{table}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = [m for s in args.symbols if (m := run_symbol(s, args)) is not None]
    if not results:
        return 1

    rows = ["| Symbol | Period | Strategy return | B&H return | Strategy max DD | B&H max DD | "
            "Strategy Sharpe | B&H Sharpe | Trades |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in results:
        s, b = m["strategy"], m["buy_and_hold"]
        rows.append(f"| {m['symbol']} | {m['period']} | {_fmt_pct(s['total_return'])} | "
                    f"{_fmt_pct(b['total_return'])} | {_fmt_pct(s['max_drawdown'])} | {_fmt_pct(b['max_drawdown'])} | "
                    f"{_fmt_num(s['sharpe_ratio'])} | {_fmt_num(b['sharpe_ratio'])} | {s['num_trades']} |")
    (args.report_dir / "summary.md").write_text(
        f"# SMA {args.fast}/{args.slow} on ETFs\n\n" + "\n".join(rows) + "\n\n"
        + "\n".join(f"![{m['symbol']}]({m['symbol']}_equity_curve.png)" for m in results) + "\n",
        encoding="utf-8",
    )
    print(f"\nReports written to {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
