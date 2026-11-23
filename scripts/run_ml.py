"""Stage 2: can a machine-learning model time ETFs better than a simple trend rule?

Pipeline:
1. Build 17 backward-looking, scale-free features per day for each ETF.
2. Label each day by the sign of the next 5-day return.
3. Purged walk-forward: train one pooled model on 2 years of all symbols, predict
   the next 6 months, roll forward. Every prediction is out-of-sample.
4. Trade the probabilities through Nautilus (enter >= 0.52, exit <= 0.48, 99% of
   cash, next-open fills, 0.03% cost per side) and compare with SMA 10/50 and buy & hold on the
   same days.

Models: LightGBM (main), logistic regression (simple baseline), and LightGBM
trained on shuffled labels (a control that should have no edge at all).

Usage:
    python scripts/run_ml.py              # writes reports/ml/
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
from src.ml.evaluate import compare_on_oos  # noqa: E402
from src.ml.evaluate import backtest_probabilities  # noqa: E402
from src.ml.features import FEATURES  # noqa: E402
from src.ml.features import build_dataset  # noqa: E402
from src.ml.walk_forward import MLConfig  # noqa: E402
from src.ml.walk_forward import pooled_auc  # noqa: E402
from src.ml.walk_forward import walk_forward_predict  # noqa: E402
from src.research import ResearchSettings  # noqa: E402

VARIANTS = {
    "lgbm": ("LightGBM", MLConfig("lgbm")),
    "logreg": ("Logistic regression", MLConfig("logreg")),
    "shuffled": ("LightGBM, shuffled labels (control)", MLConfig("lgbm", shuffle_labels=True)),
}
COLORS = {"lgbm": "#1f6feb", "logreg": "#8250df", "shuffled": "#cf222e", "sma": "#d29922", "bnh": "#8b949e"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=list(DEFAULT_SYMBOLS))
    p.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    p.add_argument("--report-dir", type=Path, default=ROOT / "reports" / "ml")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--enter", type=float, default=0.52)
    p.add_argument("--exit", type=float, default=0.48)
    p.add_argument("--cost", type=float, default=DEFAULT_COST_RATE, help="cost per side (fraction)")
    return p.parse_args(argv)


def pct(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:+.1f}%"


def num(x: float, d: int = 2) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


# ------------------------------------------------------------------------- charts
def plot_equity(symbol: str, curves: dict[str, pd.Series], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels = {"lgbm": "LightGBM", "logreg": "Logistic regression", "shuffled": "Shuffled-label control",
              "sma": "SMA 10/50", "bnh": "Buy & hold"}
    for key, eq in curves.items():
        ax.plot(eq.index, eq, color=COLORS[key], lw=1.0 if key in ("shuffled", "bnh") else 1.6,
                ls="--" if key == "bnh" else ("-." if key == "shuffled" else "-"), label=labels[key])
    ax.set_yscale("log")
    ax.set_ylabel("Equity (USD, log scale)")
    ax.set_title(f"{symbol}: out-of-sample equity, ML vs baselines (costs incl.)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_fold_auc(folds: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.8))
    base = folds["lgbm"]
    x = np.arange(len(base))
    w = 0.27
    for i, key in enumerate(("lgbm", "logreg", "shuffled")):
        ax.bar(x + (i - 1) * w, folds[key]["auc"], width=w, color=COLORS[key], label=VARIANTS[key][0])
    ax.axhline(0.5, color="black", lw=0.8)
    ax.set_xticks(x, [str(d) for d in base["test_start"]], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.35, 0.65)
    ax.set_ylabel("Test AUC")
    ax.set_title("Out-of-sample AUC per walk-forward fold (0.5 = coin flip)")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_importance(importance: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    imp = importance.sort_values()
    ax.barh(imp.index, imp.values * 100, color=COLORS["lgbm"])
    ax.set_xlabel("Share of total gain (%), averaged over folds")
    ax.set_title("LightGBM feature importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(preds: pd.DataFrame, path: Path) -> None:
    known = preds.dropna(subset=["label"])
    bins = np.linspace(0, 1, 11)
    known = known.assign(bin=pd.cut(known["prob"], bins, include_lowest=True))
    g = known.groupby("bin", observed=True).agg(prob=("prob", "mean"), hit=("label", "mean"), n=("label", "size"))
    g = g[g["n"] >= 30]
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, ls=":", label="Perfect calibration")
    ax.axhline(known["label"].mean(), color=COLORS["bnh"], lw=0.8, ls="--", label="Base rate")
    ax.plot(g["prob"], g["hit"], marker="o", color=COLORS["lgbm"], label="LightGBM")
    ax.set_xlabel("Predicted P(up over 5 days)")
    ax.set_ylabel("Actual share up")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 0.8)
    ax.set_title("Calibration (out-of-sample, all symbols)")
    ax.legend(frameon=False, fontsize=8)
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

    frames = {}
    for s in args.symbols:
        try:
            frames[s] = load_bars(args.raw_dir, s)
        except FileNotFoundError:
            print(f"{s}: no data, skipping", file=sys.stderr)
    if not frames:
        return 1
    dataset = build_dataset(frames, horizon=args.horizon)

    wf = {}
    for key, (_, cfg) in VARIANTS.items():
        cfg = MLConfig(cfg.model, horizon=args.horizon, shuffle_labels=cfg.shuffle_labels)
        wf[key] = walk_forward_predict(dataset, cfg)

    results: dict = {
        "settings": {"horizon_days": args.horizon, "enter": args.enter, "exit": args.exit,
                     "cost_per_side": settings.cost_rate, "position_fraction": settings.position_fraction,
                     "features": FEATURES, "train_days": 730, "test_days": 180},
        "classification": {
            key: {"pooled_auc": pooled_auc(w.predictions),
                  "per_symbol_auc": {s: pooled_auc(g) for s, g in w.predictions.groupby("symbol")},
                  "mean_fold_auc": float(w.folds["auc"].mean()),
                  "folds_auc_above_0_5": float((w.folds["auc"] > 0.5).mean())}
            for key, w in wf.items()
        },
        "importance": wf["lgbm"].importance.to_dict(),
        "symbols": {},
    }

    for symbol, df in frames.items():
        curves, stats = {}, {}
        main_probs = wf["lgbm"].predictions.query("symbol == @symbol")["prob"]
        comp = compare_on_oos(df, symbol, main_probs, settings, args.enter, args.exit)
        curves["lgbm"], stats["lgbm"] = comp.ml.equity, comp.ml.stats
        for key in ("logreg", "shuffled"):
            probs = wf[key].predictions.query("symbol == @symbol")["prob"]
            run = backtest_probabilities(df, symbol, probs, settings, args.enter, args.exit)
            curves[key], stats[key] = run.equity, run.stats
        curves["sma"], stats["sma"] = comp.sma.equity, comp.sma.stats
        curves["bnh"], stats["bnh"] = comp.buy_and_hold.equity, comp.buy_and_hold.stats
        plot_equity(symbol, curves, out / f"{symbol}_equity.png")
        comp.thresholds.to_csv(out / f"{symbol}_thresholds.csv", index=False)
        results["symbols"][symbol] = {
            "oos_start": str(comp.ml.equity.index[0].date()),
            "oos_end": str(comp.ml.equity.index[-1].date()),
            "stats": stats,
            "thresholds": comp.thresholds.to_dict(orient="records"),
        }

    plot_fold_auc({k: w.folds for k, w in wf.items()}, out / "fold_auc.png")
    plot_importance(wf["lgbm"].importance, out / "feature_importance.png")
    plot_calibration(wf["lgbm"].predictions, out / "calibration.png")
    wf["lgbm"].folds.to_csv(out / "folds_lgbm.csv", index=False)
    wf["lgbm"].predictions.to_csv(out / "predictions_lgbm.csv")

    summary = write_summary(results, args)
    (out / "summary.md").write_text(summary, encoding="utf-8")
    (out / "results.json").write_text(json.dumps(results, indent=2, default=str).replace("NaN", "null"),
                                      encoding="utf-8")
    print(summary.split("## Charts")[0])
    print(f"Done in {time.time() - t0:.0f}s")
    return 0


def write_summary(results: dict, args: argparse.Namespace) -> str:
    st = results["settings"]
    parts = [
        "# Stage 2 ML summary",
        "",
        f"Label: sign of the next {st['horizon_days']}-day return. {len(st['features'])} backward-looking "
        "features, one model pooled across symbols, purged walk-forward (train 730 days, test 180, "
        f"{st['horizon_days']}-day purge). Trading: enter at P >= {st['enter']}, exit at P <= {st['exit']}, "
        f"{st['position_fraction']:.0%} of cash, {st['cost_per_side'] * 100:.2f}% cost per side.",
        "",
        "## Does the model predict direction? (out-of-sample)",
        "",
    ]
    rows = []
    for key, (label, _) in VARIANTS.items():
        c = results["classification"][key]
        rows.append([label, num(c["pooled_auc"], 3)] + [num(c["per_symbol_auc"].get(s), 3) for s in results["symbols"]]
                    + [f"{c['folds_auc_above_0_5']:.0%}"])
    parts += [md_table(["Model", "Pooled AUC"] + [f"{s} AUC" for s in results["symbols"]] + ["Folds > 0.5"], rows),
              "", "AUC 0.5 is a coin flip. The shuffled-label control shows what pure noise looks like.", ""]

    parts += ["## Does it make money? (out-of-sample, after costs)", ""]
    names = {"lgbm": "LightGBM", "logreg": "Logistic regression", "shuffled": "Shuffled-label control",
             "sma": "SMA 10/50", "bnh": "Buy & hold"}
    rows = []
    for symbol, r in results["symbols"].items():
        for i, key in enumerate(("lgbm", "logreg", "shuffled", "sma", "bnh")):
            s = r["stats"][key]
            rows.append([symbol if i == 0 else "", f"{r['oos_start']} to {r['oos_end']}" if i == 0 else "",
                         names[key], pct(s["total_return"]), pct(s["max_drawdown"]), num(s["sharpe_ratio"]),
                         str(s["num_trades"])])
    parts += [md_table(["Symbol", "OOS period", "Approach", "Return", "Max DD", "Sharpe", "Trades"], rows), ""]

    parts += ["## Threshold sensitivity (LightGBM)", ""]
    rows = []
    for symbol, r in results["symbols"].items():
        for i, t in enumerate(r["thresholds"]):
            rows.append([symbol if i == 0 else "", f"{t['enter']:.2f} / {t['exit']:.2f}", pct(t["total_return"]),
                         num(t["sharpe_ratio"]), str(t["num_trades"]), f"{t['time_in_market']:.0%}"])
    parts += [md_table(["Symbol", "Enter / exit", "Return", "Sharpe", "Trades", "Time in market"], rows), ""]

    imp = sorted(results["importance"].items(), key=lambda kv: -kv[1])[:5]
    parts += ["## Most-used features (LightGBM gain share)", "",
              ", ".join(f"`{k}` {v:.0%}" for k, v in imp), ""]

    parts += ["## Charts", "", "![Fold AUC](fold_auc.png)", "",
              "![Calibration](calibration.png) ![Feature importance](feature_importance.png)", ""]
    for symbol in results["symbols"]:
        parts += [f"![{symbol} equity]({symbol}_equity.png)", ""]
    return "\n".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
