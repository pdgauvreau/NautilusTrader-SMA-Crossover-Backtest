"""Stage 2b: keep the SMA 10/50 timing, let a model (or a rule) decide the position size.

Stage 2 showed a model can't replace the trend rule. This tests a narrower job:
**when the SMA rule says "enter", how much should we bet?**

The model ("meta-label" style):
    * trained only on days when the SMA rule is long (fast SMA > slow SMA),
    * label = was the next 20-day return positive (does the trend continue?),
    * purged walk-forward, so every probability is out-of-sample.

Sizing schemes compared on the *same* SMA entries and exits. All maps and
thresholds below were fixed before looking at results:

==================  =========================================================
fixed               99% of cash on every entry (the Stage 1 strategy)
vol_target          99% x min(1, 15% / annualised 30-day vol). No ML; a control
ml_size             99% x clip((p - 0.25) / 0.5, 0.25, 1): p=0.75 -> full size
ml_gate             99% if p >= 0.5, otherwise skip the entry
ml_size_logreg      ml_size with logistic regression
ml_size_shuffled    ml_size with a model trained on shuffled labels (control)
==================  =========================================================

Sizing applies at entry only; positions are not resized while open.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.metrics import buy_and_hold_equity
from src.metrics import compute_stats
from src.ml.features import build_dataset
from src.ml.walk_forward import MLConfig
from src.ml.walk_forward import WalkForwardPredictions
from src.ml.walk_forward import walk_forward_predict
from src.pipeline import run_pipeline
from src.research import ResearchSettings

HORIZON = 20
MAX_FRACTION = 0.99
VOL_TARGET = 0.15  # annualised; roughly SPY's long-run volatility
SCHEMES = ("fixed", "vol_target", "ml_size", "ml_gate", "ml_size_logreg", "ml_size_shuffled")
SCHEME_LABELS = {
    "fixed": "Fixed 99%",
    "vol_target": "Vol target 15% (no ML)",
    "ml_size": "ML size (LightGBM)",
    "ml_gate": "ML gate (LightGBM)",
    "ml_size_logreg": "ML size (logistic reg.)",
    "ml_size_shuffled": "ML size, shuffled labels (control)",
}


def ml_fraction(p: float) -> float:
    """Map P(trend continues) to a fraction of cash."""
    return MAX_FRACTION * float(np.clip((p - 0.25) / 0.5, 0.25, 1.0))


def gate_fraction(p: float) -> float:
    return MAX_FRACTION if p >= 0.5 else 0.0


def vol_target_fraction(vol_30_daily: float) -> float:
    ann = vol_30_daily * np.sqrt(252)
    return MAX_FRACTION * float(min(1.0, VOL_TARGET / ann)) if ann > 0 else MAX_FRACTION


def long_state_dataset(frames: dict[str, pd.DataFrame], horizon: int = HORIZON) -> pd.DataFrame:
    """Feature/label rows restricted to days when SMA10 > SMA50."""
    ds = build_dataset(frames, horizon=horizon)
    return ds[ds["sma_10_50"] > 0]


def train_models(frames: dict[str, pd.DataFrame], horizon: int = HORIZON) -> dict[str, WalkForwardPredictions]:
    ds = long_state_dataset(frames, horizon)
    configs = {
        "lgbm": MLConfig("lgbm", horizon=horizon),
        "logreg": MLConfig("logreg", horizon=horizon),
        "shuffled": MLConfig("lgbm", horizon=horizon, shuffle_labels=True),
    }
    return {k: walk_forward_predict(ds, cfg) for k, cfg in configs.items()}


def _to_ns_map(series: pd.Series, fn) -> dict[int, float]:
    return {int(ts.value): fn(float(v)) for ts, v in series.items()}


def fraction_maps(
    symbol: str,
    models: dict[str, WalkForwardPredictions],
    features: pd.DataFrame,
) -> dict[str, dict[int, float] | None]:
    """Per-scheme {bar close ns: fraction}. ``None`` means the fixed 99%."""
    def probs(key: str) -> pd.Series:
        p = models[key].predictions
        return p[p["symbol"] == symbol]["prob"]

    vol = features[features["symbol"] == symbol]["vol_30"]
    return {
        "fixed": None,
        "vol_target": _to_ns_map(vol, vol_target_fraction),
        "ml_size": _to_ns_map(probs("lgbm"), ml_fraction),
        "ml_gate": _to_ns_map(probs("lgbm"), gate_fraction),
        "ml_size_logreg": _to_ns_map(probs("logreg"), ml_fraction),
        "ml_size_shuffled": _to_ns_map(probs("shuffled"), ml_fraction),
    }


@dataclass
class SizingResult:
    symbol: str
    oos_start: pd.Timestamp
    equity: dict[str, pd.Series]
    stats: pd.DataFrame  # one row per scheme (+ buy & hold)
    entries: pd.DataFrame  # one row per SMA entry: fractions each scheme used, and the trade's return


def _exposure(res) -> float:
    """Average share of equity held in the coin over the window."""
    pos = res.output.positions
    if pos.empty:
        return 0.0
    eq = res.equity
    held = pd.Series(0.0, index=eq.index)
    for _, row in pos.iterrows():
        o, c = pd.Timestamp(row["ts_opened"]), pd.Timestamp(row["ts_closed"])
        qty = float(str(row["peak_qty"]).split()[0])
        mask = (eq.index >= o) & (eq.index < c)
        held[mask] = qty * res.closes[mask]
    return float((held / eq).mean())


def compare_sizing(
    symbol: str,
    df: pd.DataFrame,
    models: dict[str, WalkForwardPredictions],
    features: pd.DataFrame,
    settings: ResearchSettings = ResearchSettings(),
) -> SizingResult:
    """Run every sizing scheme on the same out-of-sample span for ``symbol``."""
    p = models["lgbm"].predictions
    oos_start = p[p["symbol"] == symbol].index.min()
    maps = fraction_maps(symbol, models, features)

    equity, rows, runs = {}, [], {}
    for scheme in SCHEMES:
        res = run_pipeline(
            df, 10, 50, capital=settings.capital, symbol=symbol,
            position_fraction=MAX_FRACTION, cost_rate=settings.cost_rate,
            trade_start=oos_start, enter_if_above_on_start=True, entry_fractions=maps[scheme],
        )
        runs[scheme] = res
        equity[scheme] = res.equity
        st = res.strategy_stats.as_dict()
        rows.append({"scheme": scheme, **st, "avg_exposure": _exposure(res)})

    closes = runs["fixed"].closes
    bnh = buy_and_hold_equity(closes, settings.capital, settings.cost_rate)
    equity["bnh"] = bnh
    rows.append({"scheme": "bnh", **compute_stats(bnh, [float(bnh.iloc[-1]) - settings.capital],
                                                  settings.capital).as_dict(), "avg_exposure": 1.0})
    stats = pd.DataFrame(rows).set_index("scheme")

    # Per-entry view: what each scheme bet on each of the fixed strategy's trades.
    trades = runs["fixed"].trades
    entry_rows = []
    for opened, closed, ret in zip(trades["opened"], trades["closed"], trades["return"]):
        ns = int(pd.Timestamp(opened, tz="UTC").value)
        row = {"opened": opened, "closed": closed, "trade_return": ret}
        for scheme in SCHEMES[1:]:
            m = maps[scheme]
            row[scheme] = m.get(ns, np.nan) if m is not None else MAX_FRACTION
        entry_rows.append(row)
    return SizingResult(symbol, oos_start, equity, stats, pd.DataFrame(entry_rows))
