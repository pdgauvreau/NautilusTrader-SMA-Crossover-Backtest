"""Research tools: parameter sensitivity, walk-forward optimisation, random baselines.

Everything runs through the same Nautilus pipeline as the single backtest, so
fees, sizing, fills and warmup are identical across experiments.

Conventions
-----------
* Windows are defined in **bars** (rows of the daily DataFrame).
* Every run gets ``warmup`` extra bars *before* its window so the slow SMA is
  ready on the window's first bar; no trades happen during warmup
  (``trade_start``). Each window starts with fresh capital and, if the fast
  SMA is already above the slow one, enters on its first bar (the rule's
  state, "long while fast > slow", carries across window boundaries).
* Selection metric is the annualised Sharpe ratio. Runs that never trade have
  a NaN Sharpe and are ranked last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import field

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.data_loader import DEFAULT_COST_RATE
from src.data_loader import bar_type_for
from src.data_loader import bar_close_times
from src.data_loader import etf_instrument
from src.data_loader import open_quotes
from src.data_loader import to_nautilus_bars
from src.metrics import compute_stats
from src.metrics import buy_and_hold_equity
from src.pipeline import PipelineResult
from src.pipeline import run_pipeline
from src.strategies.scheduled import ScheduledLong
from src.strategies.scheduled import ScheduledLongConfig

DEFAULT_FASTS: tuple[int, ...] = (5, 10, 15, 20, 30, 40)
DEFAULT_SLOWS: tuple[int, ...] = (30, 50, 75, 100, 150, 200)


@dataclass(frozen=True)
class ResearchSettings:
    """Execution assumptions shared by every research run."""

    capital: float = 100_000.0
    position_fraction: float = 0.99  # invest 99% of free cash on each entry
    cost_rate: float = DEFAULT_COST_RATE  # per side: commission + slippage
    execution_delay_bars: int = 0


def param_grid(
    fasts: tuple[int, ...] = DEFAULT_FASTS,
    slows: tuple[int, ...] = DEFAULT_SLOWS,
) -> list[tuple[int, int]]:
    """All (fast, slow) pairs with fast < slow."""
    return [(f, s) for f in fasts for s in slows if f < s]


def run_window(
    df: pd.DataFrame,
    symbol: str,
    fast: int,
    slow: int,
    start: int,
    end: int,
    warmup: int,
    settings: ResearchSettings,
) -> PipelineResult:
    """Backtest bars ``[start, end)`` of ``df`` with ``warmup`` bars before ``start``."""
    lo = max(0, start - warmup)
    window = df.iloc[lo:end]
    # A bar's ts_event is its session close (16:00 New York).
    trade_start = bar_close_times(df)[start]
    return run_pipeline(
        window,
        fast,
        slow,
        capital=settings.capital,
        symbol=symbol,
        position_fraction=settings.position_fraction,
        execution_delay_bars=settings.execution_delay_bars,
        cost_rate=settings.cost_rate,
        trade_start=trade_start if start > 0 else None,
        enter_if_above_on_start=True,
    )


def _score(x: float) -> float:
    return -math.inf if x is None or math.isnan(x) else x


# --------------------------------------------------------------------- sensitivity
def sensitivity(
    df: pd.DataFrame,
    symbol: str,
    grid: list[tuple[int, int]],
    settings: ResearchSettings = ResearchSettings(),
    warmup: int | None = None,
) -> pd.DataFrame:
    """Full-period stats for every parameter pair (one row per pair).

    All pairs trade the same window: bars after the largest slow period, so
    longer SMAs aren't penalised for their longer warmup.
    """
    warmup = max(s for _, s in grid) if warmup is None else warmup
    rows = []
    for fast, slow in grid:
        res = run_window(df, symbol, fast, slow, warmup, len(df), warmup, settings)
        rows.append({"fast": fast, "slow": slow, **res.strategy_stats.as_dict()})
    return pd.DataFrame(rows)


# -------------------------------------------------------------------- walk-forward
@dataclass
class WalkForwardResult:
    folds: pd.DataFrame  # one row per fold
    equity: pd.Series  # stitched out-of-sample equity of the walk-forward strategy
    baseline_equity: pd.Series  # same windows, fixed baseline parameters
    buy_and_hold: pd.Series  # buy & hold over the whole out-of-sample span
    stats: dict[str, dict] = field(default_factory=dict)


def _stitch(curves: list[pd.Series], capital: float) -> pd.Series:
    """Chain per-fold equity curves (each starting from ``capital``) into one."""
    rets = []
    for eq in curves:
        r = eq.pct_change()
        r.iloc[0] = eq.iloc[0] / capital - 1.0
        rets.append(r)
    all_rets = pd.concat(rets)
    return (capital * (1.0 + all_rets).cumprod()).rename("equity")


def walk_forward(
    df: pd.DataFrame,
    symbol: str,
    grid: list[tuple[int, int]],
    settings: ResearchSettings = ResearchSettings(),
    train_bars: int = 504,  # ~2 years of sessions
    test_bars: int = 126,  # ~6 months
    baseline: tuple[int, int] = (10, 50),
    min_test_bars: int = 60,
) -> WalkForwardResult:
    """Rolling walk-forward optimisation.

    For each fold, every pair in ``grid`` is backtested on the training window
    and the best by Sharpe is then traded, untouched, on the following test
    window. Test windows don't overlap, so stitching them gives one honest
    out-of-sample equity curve. Note each fold closes any open position at its
    end, which costs an extra round trip at fold boundaries.
    """
    warmup = max(max(s for _, s in grid), baseline[1])
    capital = settings.capital
    folds, wf_curves, base_curves, closes = [], [], [], []

    train_start = warmup
    while train_start + train_bars + min_test_bars <= len(df):
        train_end = train_start + train_bars
        test_end = min(train_end + test_bars, len(df))

        scores = []
        for fast, slow in grid:
            res = run_window(df, symbol, fast, slow, train_start, train_end, warmup, settings)
            scores.append((_score(res.strategy_stats.sharpe_ratio), fast, slow, res.strategy_stats))
        best_score, fast, slow, train_stats = max(scores, key=lambda t: t[0])

        test = run_window(df, symbol, fast, slow, train_end, test_end, warmup, settings)
        base = run_window(df, symbol, *baseline, train_end, test_end, warmup, settings)
        wf_curves.append(test.equity)
        base_curves.append(base.equity)
        closes.append(test.closes)
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_start": df.index[train_start].date(),
                "test_start": df.index[train_end].date(),
                "test_end": df.index[test_end - 1].date(),
                "fast": fast,
                "slow": slow,
                "train_sharpe": train_stats.sharpe_ratio,
                "test_return": test.strategy_stats.total_return,
                "test_sharpe": test.strategy_stats.sharpe_ratio,
                "test_trades": test.strategy_stats.num_trades,
                "baseline_test_return": base.strategy_stats.total_return,
                "bnh_test_return": test.buy_and_hold_stats.total_return,
            },
        )
        train_start += test_bars

    if not folds:
        raise ValueError(f"Not enough data for one fold ({len(df)} bars)")

    equity = _stitch(wf_curves, capital)
    base_equity = _stitch(base_curves, capital)
    oos_closes = pd.concat(closes)
    bnh = buy_and_hold_equity(oos_closes, capital, settings.cost_rate)

    fold_df = pd.DataFrame(folds)

    def trades(col: str) -> list[float]:
        # Per-fold PnL as the "trade" list keeps win rate meaningful per fold.
        return [r * capital for r in fold_df[col]]

    stats = {
        "walk_forward": compute_stats(equity, trades("test_return"), capital).as_dict(),
        "baseline": compute_stats(base_equity, trades("baseline_test_return"), capital).as_dict(),
        "buy_and_hold": compute_stats(bnh, [float(bnh.iloc[-1]) - capital], capital).as_dict(),
    }
    # Count actual round trips rather than folds for the walk-forward strategy.
    stats["walk_forward"]["num_trades"] = int(fold_df["test_trades"].sum())
    for key in ("baseline", "buy_and_hold"):
        stats[key].pop("num_trades", None)
        stats[key].pop("win_rate", None)
    stats["walk_forward"].pop("win_rate", None)
    stats["walk_forward"]["fold_win_rate"] = float((fold_df["test_return"] > 0).mean())
    stats["walk_forward"]["folds_beating_baseline"] = float(
        (fold_df["test_return"] > fold_df["baseline_test_return"]).mean(),
    )
    return WalkForwardResult(fold_df, equity, base_equity, bnh, stats)


# ------------------------------------------------------------------ random baseline
def random_schedule(
    n_bars: int,
    hold_lengths: list[int],
    rng: np.random.Generator,
    first_bar: int = 0,
) -> tuple[set[int], set[int]]:
    """Random non-overlapping long periods with the given holding lengths.

    Holding lengths are shuffled and separated by random flat gaps (at least
    one bar), all placed within bars ``[first_bar, n_bars)``.
    """
    holds = list(rng.permutation(hold_lengths))
    k = len(holds)
    span = n_bars - first_bar
    free = span - sum(holds) - (k - 1) - 1  # 1-bar minimum gaps, 1 bar spare at end
    if k == 0:
        return set(), set()
    if free < 0:
        raise ValueError("Holding periods don't fit in the window")
    gaps = rng.multinomial(free, [1.0 / (k + 1)] * (k + 1))
    entries, exits = set(), set()
    pos = first_bar + gaps[0]
    for j, h in enumerate(holds):
        entries.add(int(pos))
        exits.add(int(pos + h))
        pos += h + 1 + gaps[j + 1]
    return entries, exits


@dataclass
class RandomBaselineResult:
    strategy_return: float
    strategy_sharpe: float
    random_returns: np.ndarray
    random_sharpes: np.ndarray

    @property
    def return_percentile(self) -> float:
        """Share of random schedules the strategy beat on total return."""
        return float((self.random_returns < self.strategy_return).mean())

    @property
    def sharpe_percentile(self) -> float:
        r = np.nan_to_num(self.random_sharpes, nan=-np.inf)
        return float((r < self.strategy_sharpe).mean())


def random_entry_baseline(
    df: pd.DataFrame,
    symbol: str,
    fast: int = 10,
    slow: int = 50,
    settings: ResearchSettings = ResearchSettings(),
    n_samples: int = 200,
    seed: int = 7,
) -> RandomBaselineResult:
    """Compare the SMA strategy with random schedules of identical trade count and holds.

    Answers: "is the *timing* of the signal better than luck, given the same
    exposure and costs?"
    """
    real = run_pipeline(
        df, fast, slow, capital=settings.capital, symbol=symbol,
        position_fraction=settings.position_fraction, cost_rate=settings.cost_rate,
        execution_delay_bars=settings.execution_delay_bars,
    )
    # Holding lengths in bars, from the strategy's own entry/exit decisions.
    close_ns = [int(t.value) for t in bar_close_times(df)]
    pos = {ns: i for i, ns in enumerate(close_ns)}
    holds, entry_i = [], None
    for ts, sig in real.output.signals:
        if sig.value == "enter_long":
            entry_i = pos[ts]
        elif entry_i is not None:
            holds.append(max(1, pos[ts] - entry_i))
            entry_i = None
    if entry_i is not None:  # still open at the end
        holds.append(max(1, len(close_ns) - 1 - entry_i))

    instrument = etf_instrument(symbol)
    bar_type = bar_type_for(symbol)
    bars = to_nautilus_bars(df, instrument, bar_type)
    quotes = open_quotes(df, instrument)
    closes = real.closes
    rng = np.random.default_rng(seed)
    rets, sharpes = [], []
    for _ in range(n_samples):
        entries, exits = random_schedule(len(bars), holds, rng, first_bar=slow - 1)
        cfg = ScheduledLongConfig(
            bar_type=bar_type, entry_bars=entries, exit_bars=exits,
            position_fraction=settings.position_fraction,
        )
        out = run_backtest(
            bars, instrument, cfg, starting_capital=settings.capital,
            cost_rate=settings.cost_rate, strategy_cls=ScheduledLong, quotes=quotes,
        )
        eq = out.equity.reindex(closes.index).ffill().fillna(settings.capital)
        st = compute_stats(eq, [], settings.capital)
        rets.append(st.total_return)
        sharpes.append(st.sharpe_ratio)

    return RandomBaselineResult(
        strategy_return=real.strategy_stats.total_return,
        strategy_sharpe=real.strategy_stats.sharpe_ratio,
        random_returns=np.array(rets),
        random_sharpes=np.array(sharpes),
    )


# ---------------------------------------------------------------------- robustness
def cost_delay_grid(
    df: pd.DataFrame,
    symbol: str,
    fast: int = 10,
    slow: int = 50,
    costs: tuple[float, ...] = (0.0001, 0.0003, 0.0005, 0.001),
    delays: tuple[int, ...] = (0, 1, 2),
    settings: ResearchSettings = ResearchSettings(),
) -> pd.DataFrame:
    """Fixed-parameter results under higher costs and late execution."""
    rows = []
    for cost in costs:
        for delay in delays:
            s = ResearchSettings(settings.capital, settings.position_fraction, cost, delay)
            res = run_pipeline(
                df, fast, slow, capital=s.capital, symbol=symbol,
                position_fraction=s.position_fraction, cost_rate=s.cost_rate,
                execution_delay_bars=s.execution_delay_bars,
            )
            rows.append(
                {
                    "cost_per_side": cost,
                    "delay_bars": delay,
                    "total_return": res.strategy_stats.total_return,
                    "sharpe_ratio": res.strategy_stats.sharpe_ratio,
                    "max_drawdown": res.strategy_stats.max_drawdown,
                    "num_trades": res.strategy_stats.num_trades,
                },
            )
    return pd.DataFrame(rows)
