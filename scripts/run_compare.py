"""Compare classic trading rules against SMA 10/50 and buy & hold, symbol by symbol.

Every rule runs through the same Nautilus engine with the same costs, sizing
and next-open execution, over the same sessions (after a one-year warm-up).
Rules use their standard published settings; nothing is tuned on this data.

Usage:
    python scripts/run_compare.py                                  # public symbols -> reports/compare/
    python scripts/run_compare.py --report-dir reports/portfolio/compare   # include private symbols safely
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.compare import MIN_EVAL_BARS  # noqa: E402
from src.compare import compare_symbol  # noqa: E402
from src.compare import equal_weight_portfolio  # noqa: E402
from src.compare import rank_stability  # noqa: E402
from src.compare import summarise  # noqa: E402
from src.data_loader import DEFAULT_COST_RATE  # noqa: E402
from src.data_loader import DEFAULT_SYMBOLS  # noqa: E402
from src.data_loader import ETFS  # noqa: E402
from src.data_loader import STOCKS  # noqa: E402
from src.data_loader import load_bars  # noqa: E402
from src.data_loader import validate_daily  # noqa: E402
from src.metrics import max_drawdown  # noqa: E402
from src.metrics import sharpe_ratio  # noqa: E402
from src.research import ResearchSettings  # noqa: E402
from src.strategies.signals import STRATEGIES  # noqa: E402
from src.strategies.signals import WARMUP_BARS  # noqa: E402

COLORS = {
    "buy_and_hold": "#8b949e", "sma_10_50": "#d29922", "golden_cross": "#1a7f37", "trend_200": "#0969da",
    "momentum_12_1": "#8250df", "donchian_55_20": "#bf3989", "macd": "#cf222e", "rsi2_reversion": "#57606a",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=None,
                   help="default: the public symbols, or every symbol when --report-dir is under reports/portfolio")
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "compare")
    p.add_argument("--cost", type=float, default=DEFAULT_COST_RATE, help="cost per side (fraction)")
    args = p.parse_args(argv)
    if args.symbols is None:
        private = "portfolio" in args.report_dir.resolve().parts
        args.symbols = list(DEFAULT_SYMBOLS) if private else [*ETFS, *STOCKS]
    return args


def label(key: str) -> str:
    return STRATEGIES[key][0]


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def num(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "|".join(["---"] + ["---:"] * (len(header) - 1))
    return "\n".join(["| " + " | ".join(header) + " |", "|" + align + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


# ------------------------------------------------------------------------- charts
def plot_heatmap(results, path: Path) -> None:
    keys = list(STRATEGIES)
    data = np.array([[r.stats.loc[k, "sharpe_ratio"] for r in results] for k in keys], dtype=float)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(results) + 3), 4.6))
    vmax = max(1.0, float(np.nanmax(np.abs(data))))
    im = ax.imshow(np.ma.masked_invalid(data), cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(results)), [r.symbol for r in results], rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(keys)), [label(k) for k in keys], fontsize=8)
    if len(results) <= 12:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if not np.isnan(data[i, j]):
                    ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Sharpe ratio by rule and symbol (same period, costs, execution)")
    fig.colorbar(im, ax=ax, label="Sharpe (ann.)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_portfolio(curves: dict[str, pd.Series], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for k, eq in curves.items():
        ax.plot(eq.index, eq, color=COLORS[k], lw=2.2 if k in ("buy_and_hold", "sma_10_50") else 1.3,
                ls="--" if k == "buy_and_hold" else "-", label=label(k))
    ax.set_yscale("log")
    ax.set_ylabel("Equity (USD, log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_risk_return(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for k, row in table.iterrows():
        ax.scatter(row["avg_max_drawdown"] * 100, row["avg_cagr"] * 100, s=90, color=COLORS[k], zorder=3)
        ax.annotate(label(k), (row["avg_max_drawdown"] * 100, row["avg_cagr"] * 100), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Average max drawdown (%)  (further right = shallower)")
    ax.set_ylabel("Average annual return (%)")
    ax.set_title("Return vs worst drop, averaged over symbols")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    settings = ResearchSettings(cost_rate=args.cost)
    t0 = time.time()

    results, skipped = [], []
    for s in args.symbols:
        try:
            df = load_bars(args.raw_dir, s)
        except FileNotFoundError:
            print(f"{s}: no data, skipping", file=sys.stderr)
            continue
        validate_daily(df)
        if len(df) < WARMUP_BARS + MIN_EVAL_BARS:
            skipped.append((s, len(df)))
            print(f"{s}: {len(df)} sessions, needs {WARMUP_BARS + MIN_EVAL_BARS}; skipping")
            continue
        res = compare_symbol(df, s, settings)
        res.stats.to_csv(out / f"{s}_compare.csv")
        results.append(res)
        best = res.stats["sharpe_ratio"].idxmax()
        print(f"{s}: {res.start.date()} to {res.end.date()}, best Sharpe: {label(best)}")
    if not results:
        return 1

    table = summarise(results)
    stability = rank_stability(results)
    plot_heatmap(results, out / "sharpe_heatmap.png")
    plot_risk_return(table, out / "risk_return.png")

    # Equal-weight portfolios over the common period of the full-history symbols.
    long_hist = [r for r in results if r.start == min(r.start for r in results)]
    port_set = long_hist if len(long_hist) >= 2 else results
    curves = {k: equal_weight_portfolio(port_set, k) for k in STRATEGIES}
    plot_portfolio(curves, f"Equal-weight portfolio of {len(port_set)} symbols with full history, by rule",
                   out / "portfolio_equity.png")
    port_stats = {
        k: {"total_return": float(c.iloc[-1] / c.iloc[0] - 1), "sharpe_ratio": sharpe_ratio(c),
            "max_drawdown": max_drawdown(c)}
        for k, c in curves.items()
    }

    lines = [
        "# Strategy comparison",
        "",
        f"{len(results)} symbols. Every rule trades the same sessions per symbol (after a {WARMUP_BARS}-session "
        f"warm-up), invests {settings.position_fraction:.0%} of cash, decides at the close and fills at the next "
        f"open (1 bp half-spread) plus {settings.cost_rate * 100:.2f}% per side. Long or cash only.",
        "",
    ]
    if skipped:
        lines += [f"Skipped (too little history; sessions): {', '.join(f'{s} ({n})' for s, n in skipped)}.", ""]
    lines += ["## Overall ranking (averaged across symbols)", ""]
    rows = [[label(k), f"{r['avg_rank']:.1f}", num(r["avg_sharpe"]), pct(r["avg_cagr"]), pct(r["avg_max_drawdown"]),
             f"{r['avg_time_in_market']:.0%}", f"{r['avg_trades']:.0f}",
             f"{int(r['beats_bh_sharpe'])}/{int(r['symbols'])}", f"{int(r['beats_bh_return'])}/{int(r['symbols'])}",
             f"{int(r['shallower_dd_than_bh'])}/{int(r['symbols'])}"]
            for k, r in table.iterrows()]
    lines += [md_table(["Rule", "Avg rank", "Avg Sharpe", "Avg CAGR", "Avg max DD", "In market", "Trades",
                        "Beats B&H Sharpe", "Beats B&H return", "Shallower DD"], rows), ""]
    lines += [
        f"**Ranking stability:** the rank correlation of the rules' Sharpe ratios between the first and second "
        f"half of each period averages **{stability:+.2f}** (1 = same order both halves, 0 = unrelated). "
        + ("Low values mean the 'best' rule is largely luck of the period." if stability < 0.5
           else "The ordering is fairly consistent across halves."),
        "",
        f"## Equal-weight portfolio ({len(port_set)} full-history symbols, from {port_set[0].start.date()})",
        "",
    ]
    rows = [[label(k), pct(v["total_return"]), num(v["sharpe_ratio"]), pct(v["max_drawdown"])]
            for k, v in sorted(port_stats.items(), key=lambda kv: -kv[1]["sharpe_ratio"])]
    lines += [md_table(["Rule", "Return", "Sharpe", "Max DD"], rows), ""]
    lines += ["## Best rule per symbol (by Sharpe)", ""]
    rows = []
    for r in results:
        st = r.stats
        best = st["sharpe_ratio"].idxmax()
        rows.append([r.symbol, f"{r.start.date()} to {r.end.date()}", label(best), num(st.loc[best, "sharpe_ratio"]),
                     num(st.loc["buy_and_hold", "sharpe_ratio"]), num(st.loc["sma_10_50", "sharpe_ratio"]),
                     pct(st.loc[best, "total_return"]), pct(st.loc["buy_and_hold", "total_return"])])
    lines += [md_table(["Symbol", "Period", "Best rule", "Its Sharpe", "B&H Sharpe", "10/50 Sharpe",
                        "Its return", "B&H return"], rows, "---|---|---|" + "|".join(["---:"] * 5)), ""]
    lines += ["## Charts", "", "![Portfolio](portfolio_equity.png)", "",
              "![Risk vs return](risk_return.png) ![Sharpe heatmap](sharpe_heatmap.png)", ""]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({
        "settings": {"cost_per_side": settings.cost_rate, "position_fraction": settings.position_fraction,
                     "warmup_sessions": WARMUP_BARS},
        "strategies": {k: v[0] for k, v in STRATEGIES.items()},
        "summary": table.reset_index().to_dict(orient="records"),
        "rank_stability": stability,
        "portfolio": {"symbols": [r.symbol for r in port_set], "start": str(port_set[0].start.date()),
                      "stats": port_stats},
        "symbols": {r.symbol: {"start": str(r.start.date()), "end": str(r.end.date()),
                               "stats": r.stats.reset_index().to_dict(orient="records")} for r in results},
        "skipped": skipped,
    }, indent=2, default=str).replace("NaN", "null"), encoding="utf-8")
    print(summary.split("## Charts")[0])
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
