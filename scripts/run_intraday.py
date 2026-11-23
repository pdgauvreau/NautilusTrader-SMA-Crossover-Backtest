"""Stage 4: day trading. Do classic intraday rules beat holding after realistic costs?

For each symbol (1-minute bars, regular hours, ~3 years) every rule in
src/intraday/rules.py is traded through Nautilus: decide on a minute's close,
fill at the next minute's open quote, flat by the close every day. Then:

* luck test: random trades with the same per-day structure and holding times,
* cost stress: results at 0-5 bp per side all-in, plus each rule's break-even cost,
* stability: return in each calendar year.

Usage:
    python scripts/download_intraday.py
    python scripts/run_intraday.py                    # writes reports/intraday/
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

from src.data_loader import equity_instrument  # noqa: E402
from src.intraday.data import INTRADAY_DIR  # noqa: E402
from src.intraday.data import load_minute_bars  # noqa: E402
from src.intraday.data import minute_bar_type  # noqa: E402
from src.intraday.data import to_nautilus_minute_bars  # noqa: E402
from src.intraday.data import validate_minutes  # noqa: E402
from src.intraday.evaluate import COST_GRID_BP  # noqa: E402
from src.intraday.evaluate import DEFAULT_COST  # noqa: E402
from src.intraday.evaluate import DEFAULT_HALF_SPREAD  # noqa: E402
from src.intraday.evaluate import breakeven_cost_bp  # noqa: E402
from src.intraday.evaluate import cost_stress  # noqa: E402
from src.intraday.evaluate import random_control  # noqa: E402
from src.intraday.evaluate import run_rule  # noqa: E402
from src.intraday.rules import INTRADAY_RULES  # noqa: E402

COLORS = {"buy_and_hold": "#8b949e", "hold_intraday": "#d29922", "orb_30": "#1f6feb",
          "intraday_momentum": "#8250df", "vwap_reversion": "#cf222e", "gap_fade": "#1a7f37"}
DAY_RULES = [k for k in INTRADAY_RULES if k != "buy_and_hold"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM", "NVDA"])
    p.add_argument("--raw-dir", type=Path, default=INTRADAY_DIR)
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "intraday")
    p.add_argument("--random-samples", type=int, default=100)
    return p.parse_args(argv)


def label(k: str) -> str:
    return INTRADAY_RULES[k][0]


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def num(x: float, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def md_table(header, rows) -> str:
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
                     + ["| " + " | ".join(r) + " |" for r in rows])


def plot_equity(symbol: str, results: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for k, r in results.items():
        ax.plot(r.daily_equity.index, r.daily_equity, color=COLORS[k], lw=2 if k == "buy_and_hold" else 1.3,
                ls="--" if k == "buy_and_hold" else "-", label=label(k))
    ax.axhline(100_000, color="black", lw=0.6, alpha=0.4)
    ax.set_ylabel("Equity (USD)")
    ax.set_title(f"{symbol}: day-trading rules vs buy & hold (next-minute fills, 1 bp per side all-in)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_costs(stress: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for k in DAY_RULES:
        avg = pd.concat([s[k].set_index("cost_bp_per_side")["total_return"] for s in stress.values()], axis=1).mean(axis=1)
        ax.plot(avg.index, avg.values * 100, marker="o", color=COLORS[k], label=label(k))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("All-in cost per side (basis points)")
    ax.set_ylabel("Total return, average over symbols (%)")
    ax.set_title("How much of each rule survives trading costs")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    per_symbol, stress, controls, data_info = {}, {}, {}, {}
    for s in args.symbols:
        try:
            df = load_minute_bars(args.raw_dir, s)
        except FileNotFoundError as exc:
            print(exc, file=sys.stderr)
            continue
        data_info[s] = validate_minutes(df)
        bars = to_nautilus_minute_bars(df, equity_instrument(s), minute_bar_type(s))
        res = {k: run_rule(df, s, k, bars=bars) for k in INTRADAY_RULES}
        per_symbol[s] = res
        stress[s] = {k: cost_stress(df, res[k].target) for k in DAY_RULES}
        controls[s] = {}
        for k in DAY_RULES:
            ctrl = random_control(df, res[k].target, n_samples=args.random_samples)
            st = res[k].stats
            controls[s][k] = {
                "return_percentile": float((ctrl["total_return"] < st["total_return"]).mean()),
                "sharpe_percentile": float((ctrl["sharpe_ratio"].fillna(-np.inf) < st["sharpe_ratio"]).mean()),
                "random_median_return": float(ctrl["total_return"].median()),
                "breakeven_cost_bp": breakeven_cost_bp(df, res[k].target),
            }
        plot_equity(s, res, out / f"{s}_intraday_equity.png")
        pd.DataFrame({k: r.stats for k, r in res.items()}).T.to_csv(out / f"{s}_intraday_stats.csv")
        print(f"{s}: done ({time.time() - t0:.0f}s)")
    if not per_symbol:
        return 1
    plot_costs(stress, out / "cost_stress.png")

    syms = list(per_symbol)
    period = f"{data_info[syms[0]]['first']} to {data_info[syms[0]]['last']}"
    lines = [
        "# Stage 4: day trading",
        "",
        f"1-minute bars, regular hours, {period} ({data_info[syms[0]]['sessions']} sessions). Long only. "
        "Each rule decides on a minute's close and fills at the next minute's open "
        f"({DEFAULT_HALF_SPREAD * 1e4:.1f} bp half-spread + {DEFAULT_COST * 1e4:.1f} bp per side, "
        "about 1 bp per side all-in), and every day-trading rule is flat by the close.",
        "",
        "## Results",
        "",
    ]
    rows = []
    for s in syms:
        for i, k in enumerate(INTRADAY_RULES):
            st = per_symbol[s][k].stats
            rows.append([s if i == 0 else "", label(k), pct(st["total_return"]), num(st["sharpe_ratio"]),
                         pct(st["max_drawdown"]), str(st["trades"]), num(st["avg_trade_bps"], 1),
                         f"{st['win_rate']:.0%}" if not math.isnan(st["win_rate"]) else "n/a",
                         f"{st['time_in_market']:.0%}"])
    lines += [md_table(["Symbol", "Rule", "Return", "Sharpe", "Max DD", "Trades", "Avg trade (bp)", "Win rate",
                        "In market"], rows), ""]

    lines += ["## Is it better than luck, and does it survive costs?", "",
              "*Beats random* = share of random trade schedules (same trades per day, same holding times, "
              "random days and minutes) with a lower return. *Break-even* = all-in cost per side at which "
              "the average trade nets zero; liquid ETFs cost roughly 0.5–1 bp per side, retail market "
              "orders often more.", ""]
    rows = []
    for k in DAY_RULES:
        rows.append([label(k)] + [f"{controls[s][k]['return_percentile']:.0%} / {num(controls[s][k]['breakeven_cost_bp'], 1)} bp"
                                  for s in syms])
    lines += [md_table(["Rule"] + [f"{s}: beats random / break-even" for s in syms], rows), ""]

    lines += ["## Return by year (average over symbols)", ""]
    years = sorted({y for s in syms for y in per_symbol[s]["buy_and_hold"].yearly})
    rows = []
    for k in INTRADAY_RULES:
        rows.append([label(k)] + [pct(float(np.mean([per_symbol[s][k].yearly.get(y, np.nan) for s in syms])))
                                  for y in years])
    lines += [md_table(["Rule"] + [f"{y}{'*' if y in (years[0], years[-1]) else ''}" for y in years], rows),
              "", "\\*Partial year.", ""]

    lines += ["## Charts", "", "![Cost stress](cost_stress.png)", ""]
    lines += [f"![{s}]({s}_intraday_equity.png)" for s in syms] + [""]
    summary = "\n".join(lines)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "results.json").write_text(json.dumps({
        "period": period, "rules": {k: v[0] for k, v in INTRADAY_RULES.items()},
        "costs": {"half_spread": DEFAULT_HALF_SPREAD, "cost_per_side": DEFAULT_COST, "grid_bp": list(COST_GRID_BP)},
        "data": data_info,
        "symbols": {s: {"stats": {k: r.stats for k, r in per_symbol[s].items()},
                        "yearly": {k: r.yearly for k, r in per_symbol[s].items()},
                        "controls": controls[s],
                        "cost_stress": {k: v.to_dict(orient="records") for k, v in stress[s].items()}}
                    for s in syms},
    }, indent=2, default=str).replace("NaN", "null"), encoding="utf-8")
    print(summary.split("## Charts")[0])
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
