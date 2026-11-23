"""Stage 1 research: is the SMA crossover edge real, or an artefact of one backtest?

For each ETF (Alpaca daily data from 2016) this runs:

1. Parameter sensitivity: Sharpe for a grid of fast/slow SMA pairs.
2. Walk-forward optimisation: pick the best pair on 2 years, trade it on the
   next 6 months, roll forward. Compared with fixed 10/50 and buy & hold on the
   same out-of-sample periods.
3. Random-entry baseline: the 10/50 strategy vs random schedules with the same
   number of trades and holding lengths (timing skill vs luck).
4. Cost & delay robustness: 10/50 under higher costs and late execution.

Default execution assumptions: 99% of cash per entry, decided at the close and
filled at the next open (1 bp half-spread), plus 0.03% cost per side.

Usage:
    python scripts/download_data.py
    python scripts/run_research.py            # writes reports/research/
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

from src.data_loader import DEFAULT_COST_RATE  # noqa: E402
from src.data_loader import DEFAULT_SYMBOLS  # noqa: E402
from src.data_loader import load_bars  # noqa: E402
from src.data_loader import validate_daily  # noqa: E402
from src.research import ResearchSettings  # noqa: E402
from src.research import cost_delay_grid  # noqa: E402
from src.research import param_grid  # noqa: E402
from src.research import random_entry_baseline  # noqa: E402
from src.research import sensitivity  # noqa: E402
from src.research import walk_forward  # noqa: E402

COLORS = {"wf": "#1f6feb", "base": "#d29922", "bnh": "#8b949e", "real": "#cf222e"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "research")
    p.add_argument("--train-bars", type=int, default=504, help="walk-forward training window (sessions, ~2 years)")
    p.add_argument("--test-bars", type=int, default=126, help="walk-forward test window (sessions, ~6 months)")
    p.add_argument("--random-samples", type=int, default=200)
    p.add_argument("--cost", type=float, default=DEFAULT_COST_RATE, help="cost per side (fraction)")
    return p.parse_args(argv)


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def num(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def md_table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "|".join(["---"] + ["---:"] * (len(header) - 1))
    lines = ["| " + " | ".join(header) + " |", "|" + align + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------------- charts
def plot_heatmap(sens: pd.DataFrame, symbol: str, path: Path) -> None:
    grid = sens.pivot(index="fast", columns="slow", values="sharpe_ratio")
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    data = np.ma.masked_invalid(grid.values.astype(float))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=max(1.5, float(np.nanmax(grid.values))), aspect="auto")
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.set_xlabel("Slow SMA (days)")
    ax.set_ylabel("Fast SMA (days)")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"{symbol}: Sharpe by SMA pair (full period)")
    fig.colorbar(im, ax=ax, label="Sharpe (ann.)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_walk_forward(wf, symbol: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(wf.equity.index, wf.equity, color=COLORS["wf"], lw=1.8, label="Walk-forward optimised")
    ax.plot(wf.baseline_equity.index, wf.baseline_equity, color=COLORS["base"], lw=1.5, label="Fixed SMA 10/50")
    ax.plot(wf.buy_and_hold.index, wf.buy_and_hold, color=COLORS["bnh"], lw=1.2, ls="--", label="Buy & hold")
    for d in wf.folds["test_start"]:
        ax.axvline(pd.Timestamp(d, tz="UTC"), color="black", lw=0.4, alpha=0.25)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (USD, log scale)")
    ax.set_title(f"{symbol}: out-of-sample equity (vertical lines = fold starts)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_random(rb, symbol: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rb.random_returns * 100, bins=30, color=COLORS["bnh"], alpha=0.8, label="Random schedules")
    ax.axvline(rb.strategy_return * 100, color=COLORS["real"], lw=2,
               label=f"SMA 10/50 (beats {rb.return_percentile:.0%})")
    ax.set_xlabel("Total return (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"{symbol}: SMA timing vs {len(rb.random_returns)} random entry schedules")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def research_symbol(symbol: str, args: argparse.Namespace, settings: ResearchSettings) -> dict:
    df = load_bars(args.raw_dir, symbol)
    data = validate_daily(df)
    grid = param_grid()
    out = args.report_dir
    t0 = time.time()

    sens = sensitivity(df, symbol, grid, settings)
    plot_heatmap(sens, symbol, out / f"{symbol}_sensitivity.png")
    sens.to_csv(out / f"{symbol}_sensitivity.csv", index=False)

    wf = walk_forward(df, symbol, grid, settings, args.train_bars, args.test_bars)
    plot_walk_forward(wf, symbol, out / f"{symbol}_walk_forward.png")
    wf.folds.to_csv(out / f"{symbol}_walk_forward_folds.csv", index=False)

    rb = random_entry_baseline(df, symbol, settings=settings, n_samples=args.random_samples)
    plot_random(rb, symbol, out / f"{symbol}_random_baseline.png")

    robust = cost_delay_grid(df, symbol, settings=settings)
    robust.to_csv(out / f"{symbol}_robustness.csv", index=False)

    s = sens["sharpe_ratio"]
    base_row = sens[(sens.fast == 10) & (sens.slow == 50)].iloc[0]
    print(f"{symbol}: done in {time.time() - t0:.0f}s")
    return {
        "symbol": symbol,
        "data": {k: str(v) for k, v in data.items()},
        "sensitivity": {
            "pairs": len(sens),
            "sharpe_min": float(s.min()),
            "sharpe_median": float(s.median()),
            "sharpe_max": float(s.max()),
            "share_positive_return": float((sens["total_return"] > 0).mean()),
            "sma_10_50_sharpe": float(base_row["sharpe_ratio"]),
            "best_pair": [int(sens.loc[s.idxmax(), "fast"]), int(sens.loc[s.idxmax(), "slow"])],
        },
        "walk_forward": {
            "oos_start": str(wf.folds["test_start"].iloc[0]),
            "oos_end": str(wf.folds["test_end"].iloc[-1]),
            "folds": len(wf.folds),
            "chosen_pairs": [f"{a}/{b}" for a, b in zip(wf.folds["fast"], wf.folds["slow"])],
            **wf.stats,
        },
        "random_baseline": {
            "samples": len(rb.random_returns),
            "strategy_return": rb.strategy_return,
            "strategy_sharpe": rb.strategy_sharpe,
            "random_return_median": float(np.median(rb.random_returns)),
            "random_sharpe_median": float(np.nanmedian(rb.random_sharpes)),
            "return_percentile": rb.return_percentile,
            "sharpe_percentile": rb.sharpe_percentile,
        },
        "robustness": robust.to_dict(orient="records"),
    }


def write_summary(results: list[dict], args: argparse.Namespace, settings: ResearchSettings) -> str:
    parts = [
        "# Stage 1 research summary",
        "",
        f"Execution assumptions: {settings.position_fraction:.0%} of free cash per entry, "
        f"{settings.cost_rate * 100:.2f}% cost per side, decided at the close and filled at the next open "
        "(1 bp half-spread), long only.",
        "",
        "## Walk-forward (out-of-sample only)",
        "",
        f"Train {args.train_bars} sessions, choose the best of {len(param_grid())} SMA pairs by Sharpe, "
        f"trade it for the next {args.test_bars} sessions, roll forward.",
        "",
    ]
    rows = []
    for r in results:
        w = r["walk_forward"]
        for label, key in (("Walk-forward", "walk_forward"), ("Fixed 10/50", "baseline"), ("Buy & hold", "buy_and_hold")):
            st = w[key]
            rows.append([r["symbol"] if key == "walk_forward" else "", label,
                         f"{w['oos_start']} to {w['oos_end']}" if key == "walk_forward" else "",
                         pct(st["total_return"]), pct(st["max_drawdown"]), num(st["sharpe_ratio"])])
    parts += [md_table(["Symbol", "Approach", "OOS period", "Return", "Max DD", "Sharpe"], rows,
                       "---|---|---|---:|---:|---:"), ""]

    parts += ["## Parameter sensitivity (full period)", ""]
    rows = [[r["symbol"], str(r["sensitivity"]["pairs"]), num(r["sensitivity"]["sharpe_min"]),
             num(r["sensitivity"]["sharpe_median"]), num(r["sensitivity"]["sharpe_max"]),
             "/".join(map(str, r["sensitivity"]["best_pair"])), num(r["sensitivity"]["sma_10_50_sharpe"]),
             f"{r['sensitivity']['share_positive_return']:.0%}"] for r in results]
    parts += [md_table(["Symbol", "Pairs", "Sharpe min", "median", "max", "Best pair", "10/50", "Pairs profitable"],
                       rows), ""]

    parts += ["## Random-entry baseline (SMA 10/50, full period)", ""]
    rows = [[r["symbol"], pct(r["random_baseline"]["strategy_return"]),
             pct(r["random_baseline"]["random_return_median"]),
             f"{r['random_baseline']['return_percentile']:.0%}",
             f"{r['random_baseline']['sharpe_percentile']:.0%}"] for r in results]
    parts += [md_table(["Symbol", "Strategy return", "Random median", "Beats (return)", "Beats (Sharpe)"], rows),
              "", f"Random schedules have the same number of trades and holding lengths as the strategy "
              f"({results[0]['random_baseline']['samples']} samples each).", ""]

    parts += ["## Cost and delay robustness (SMA 10/50, full period, Sharpe)", ""]
    for r in results:
        rob = pd.DataFrame(r["robustness"])
        piv = rob.pivot(index="cost_per_side", columns="delay_bars", values="sharpe_ratio")
        rows = [[f"{c * 100:.2f}%"] + [num(v) for v in piv.loc[c]] for c in piv.index]
        parts += [f"**{r['symbol']}**", "",
                  md_table(["Cost / side"] + [f"delay {d}" for d in piv.columns], rows), ""]

    parts += ["## Charts", ""]
    for r in results:
        s = r["symbol"]
        parts += [f"### {s}", "", f"![{s} walk-forward]({s}_walk_forward.png)", "",
                  f"![{s} sensitivity]({s}_sensitivity.png) ![{s} random]({s}_random_baseline.png)", ""]
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    settings = ResearchSettings(cost_rate=args.cost)

    results = []
    for symbol in args.symbols:
        try:
            results.append(research_symbol(symbol, args, settings))
        except FileNotFoundError:
            print(f"{symbol}: no data in {args.raw_dir}, skipping (run scripts/download_data.py)", file=sys.stderr)
    if not results:
        return 1

    summary = write_summary(results, args, settings)
    (args.report_dir / "summary.md").write_text(summary, encoding="utf-8")
    (args.report_dir / "results.json").write_text(
        json.dumps(results, indent=2, default=str).replace("NaN", "null"), encoding="utf-8",
    )
    print(summary.split("## Charts")[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
