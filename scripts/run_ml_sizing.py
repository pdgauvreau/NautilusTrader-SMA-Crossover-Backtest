"""Stage 2b: does an ML model improve the SMA 10/50 rule as a *position-size filter*?

Same SMA entries and exits for every scheme; only the amount invested at each
entry changes. Compared on Sharpe (because smaller bets shrink returns and
drawdowns together), plus a direct check of whether the model's bet sizes
line up with how the trades actually turned out.

Usage:
    python scripts/run_ml_sizing.py       # writes reports/ml_sizing/
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
import pandas as pd  # noqa: E402

from src.data_loader import DEFAULT_COST_RATE  # noqa: E402
from src.data_loader import DEFAULT_SYMBOLS  # noqa: E402
from src.data_loader import load_bars  # noqa: E402
from src.ml.features import build_dataset  # noqa: E402
from src.ml.sizing import HORIZON  # noqa: E402
from src.ml.sizing import SCHEMES  # noqa: E402
from src.ml.sizing import SCHEME_LABELS  # noqa: E402
from src.ml.sizing import VOL_TARGET  # noqa: E402
from src.ml.sizing import compare_sizing  # noqa: E402
from src.ml.sizing import train_models  # noqa: E402
from src.ml.walk_forward import pooled_auc  # noqa: E402
from src.research import ResearchSettings  # noqa: E402

COLORS = {"fixed": "#d29922", "vol_target": "#1a7f37", "ml_size": "#1f6feb", "ml_gate": "#54aeff",
          "ml_size_logreg": "#8250df", "ml_size_shuffled": "#cf222e", "bnh": "#8b949e"}
PLOTTED = ("fixed", "vol_target", "ml_size", "ml_size_shuffled", "bnh")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "ml_sizing")
    p.add_argument("--cost", type=float, default=DEFAULT_COST_RATE, help="cost per side (fraction)")
    return p.parse_args(argv)


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def num(x: float, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def plot_equity(res, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for key in PLOTTED:
        eq = res.equity[key]
        label = "Buy & hold" if key == "bnh" else SCHEME_LABELS[key]
        ax.plot(eq.index, eq, color=COLORS[key], lw=1.1 if key in ("bnh", "ml_size_shuffled") else 1.6,
                ls="--" if key == "bnh" else ("-." if key == "ml_size_shuffled" else "-"), label=label)
    ax.set_yscale("log")
    ax.set_ylabel("Equity (USD, log scale)")
    ax.set_title(f"{res.symbol}: SMA 10/50 timing, different position sizing (out-of-sample)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_bets(entries: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, key in zip(axes, ("ml_size", "ml_size_shuffled")):
        for sym, g in entries.groupby("symbol"):
            ax.scatter(g[key] * 100, g["trade_return"] * 100, s=26, alpha=0.8, label=sym)
        rho = entries[[key, "trade_return"]].corr(method="spearman").iloc[0, 1]
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("Bet size (% of cash)")
        ax.set_title(f"{SCHEME_LABELS[key]}\nSpearman ρ = {rho:+.2f}", fontsize=9)
        ax.set_yscale("symlog", linthresh=20)
    axes[0].set_ylabel("Trade return (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Did bigger bets go to better trades? (every SMA entry, all symbols)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.report_dir
    out.mkdir(parents=True, exist_ok=True)
    settings = ResearchSettings(cost_rate=args.cost)
    t0 = time.time()

    frames = {}
    for s in args.symbols:
        try:
            frames[s] = load_bars(args.raw_dir, s)
        except FileNotFoundError:
            print(f"{s}: no data, skipping", file=sys.stderr)
    if not frames:
        return 1

    models = train_models(frames)
    features = build_dataset(frames)

    results = {"settings": {"horizon_days": HORIZON, "vol_target": VOL_TARGET,
                            "cost_per_side": settings.cost_rate},
               "auc": {k: pooled_auc(m.predictions) for k, m in models.items()},
               "symbols": {}}
    all_entries = []
    for symbol, df in frames.items():
        n_oos = int((models["lgbm"].predictions["symbol"] == symbol).sum())
        if n_oos < 60:
            print(f"{symbol}: only {n_oos} out-of-sample long-state predictions; skipping")
            continue
        res = compare_sizing(symbol, df, models, features, settings)
        plot_equity(res, out / f"{symbol}_equity.png")
        res.entries.assign(symbol=symbol).to_csv(out / f"{symbol}_entries.csv", index=False)
        all_entries.append(res.entries.assign(symbol=symbol))
        results["symbols"][symbol] = {
            "oos_start": str(res.oos_start.date()),
            "oos_end": str(res.equity["fixed"].index[-1].date()),
            "stats": res.stats.reset_index().to_dict(orient="records"),
        }

    entries = pd.concat(all_entries, ignore_index=True)
    plot_bets(entries, out / "bets_vs_outcomes.png")
    results["bet_correlation"] = {
        k: float(entries[[k, "trade_return"]].corr(method="spearman").iloc[0, 1])
        for k in SCHEMES[1:]
    }
    results["n_entries"] = len(entries)

    summary = write_summary(results)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str).replace("NaN", "null"),
                                      encoding="utf-8")
    print(summary.split("## Charts")[0])
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


def write_summary(results: dict) -> str:
    st = results["settings"]
    auc = results["auc"]
    parts = [
        "# Stage 2b: ML as a position-size filter for SMA 10/50",
        "",
        f"The model is trained only on days when SMA10 > SMA50, to predict whether the next "
        f"{st['horizon_days']} days are up (purged walk-forward, out-of-sample). Every scheme uses the same SMA "
        f"entries and exits; only the size of each entry changes. Cost {st['cost_per_side'] * 100:.2f}% per side.",
        "",
        f"Out-of-sample AUC on SMA-long days: LightGBM {auc['lgbm']:.3f}, logistic regression "
        f"{auc['logreg']:.3f}, shuffled-label control {auc['shuffled']:.3f}.",
        "",
        "## Sharpe ratio by sizing scheme",
        "",
    ]
    syms = list(results["symbols"])
    by = {s: {r["scheme"]: r for r in results["symbols"][s]["stats"]} for s in syms}
    rows = []
    for scheme in (*SCHEMES, "bnh"):
        label = "Buy & hold" if scheme == "bnh" else SCHEME_LABELS[scheme]
        sh = [by[s][scheme]["sharpe_ratio"] for s in syms]
        rows.append([label] + [num(x) for x in sh] + [num(sum(sh) / len(sh))])
    parts += [md_table(["Scheme"] + syms + ["Average"], rows), ""]

    parts += ["## Return, drawdown and exposure", ""]
    rows = []
    for s in syms:
        for i, scheme in enumerate((*SCHEMES, "bnh")):
            r = by[s][scheme]
            label = "Buy & hold" if scheme == "bnh" else SCHEME_LABELS[scheme]
            rows.append([s if i == 0 else "", label, pct(r["total_return"]), pct(r["max_drawdown"]),
                         f"{r['avg_exposure']:.0%}", str(r["num_trades"])])
    parts += [md_table(["Symbol", "Scheme", "Return", "Max DD", "Avg exposure", "Trades"], rows), ""]

    parts += ["## Did bigger bets go to better trades?", "",
              f"Spearman correlation between each scheme's bet size and the trade's return, across all "
              f"{results['n_entries']} SMA entries (0 = no relationship):", ""]
    rows = [[SCHEME_LABELS[k], f"{v:+.2f}"] for k, v in results["bet_correlation"].items()]
    parts += [md_table(["Scheme", "Spearman ρ"], rows), ""]

    parts += ["## Charts", "", "![Bets vs outcomes](bets_vs_outcomes.png)", ""]
    for s in syms:
        parts += [f"![{s}]({s}_equity.png)", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
