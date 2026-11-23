"""Tests for the strategy library, the generic target-position strategy and the comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.compare import compare_symbol
from src.compare import equal_weight_portfolio
from src.compare import rank_stability
from src.compare import summarise
from src.pipeline import run_pipeline
from src.strategies.signals import STRATEGIES
from src.strategies.signals import _month_start_mask
from src.strategies.signals import donchian
from src.strategies.signals import rsi2_reversion
from src.strategies.signals import sma_cross
from tests.conftest import close_times
from tests.conftest import make_ohlcv


def random_walk(n: int, seed: int = 0, drift: float = 0.0004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 300 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_signals_never_use_future_data(key: str) -> None:
    df = make_ohlcv(random_walk(600))
    _, rule = STRATEGIES[key]
    full = rule(df)
    assert full.dtype == bool and len(full) == len(df) and full.index.equals(df.index)
    for cut in (300, 451, 599):
        assert np.array_equal(full.iloc[:cut].to_numpy(), rule(df.iloc[:cut]).to_numpy()), cut
    tampered = df.copy()
    tampered.iloc[400:, :4] *= 2.0  # wildly change the future
    assert np.array_equal(full.iloc[:400].to_numpy(), rule(tampered).iloc[:400].to_numpy())


def test_month_start_mask() -> None:
    idx = pd.DatetimeIndex(["2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02", "2024-03-01"]).tz_localize("UTC")
    assert _month_start_mask(idx).tolist() == [True, False, True, False, True]


def test_donchian_breakout_and_exit() -> None:
    closes = np.r_[np.full(60, 100.0), np.full(5, 120.0), np.full(30, 120.0), np.full(10, 90.0)]
    df = make_ohlcv(closes)
    df["high"] = df["close"]
    df["low"] = df["close"]
    pos = donchian(55, 20)(df)
    assert not pos.iloc[:60].any()
    assert pos.iloc[60]  # first close above the prior 55-day high
    assert pos.iloc[60:95].all()
    assert not pos.iloc[95]  # first close below the prior 20-day low


def test_rsi2_buys_dips_only_in_an_uptrend() -> None:
    up = np.linspace(100, 200, 230)
    dip = np.r_[up, up[-1] * np.array([0.97, 0.94, 0.92]), np.linspace(up[-1] * 0.95, up[-1] * 1.05, 10)]
    pos = rsi2_reversion(make_ohlcv(dip))
    assert not pos.iloc[:230].any()
    assert pos.iloc[230:235].any()  # bought the sharp dip
    assert not pos.iloc[-1]  # sold once back above the 5-day average
    down = make_ohlcv(np.r_[np.linspace(200, 100, 230), [97, 94, 92]])
    assert not rsi2_reversion(down).any()  # below the 200-day SMA: no trades


def test_generic_strategy_matches_original_sma_strategy() -> None:
    """SMA 10/50 through TargetPositionStrategy == the original SmaCrossover (same rule, same execution)."""
    df = make_ohlcv(random_walk(500, seed=3))
    warmup = 252
    cmp = compare_symbol(df, "SPY", keys=["sma_10_50"], warmup=warmup)
    orig = run_pipeline(df, 10, 50, trade_start=close_times(df)[warmup], enter_if_above_on_start=True)
    assert cmp.stats.loc["sma_10_50", "total_return"] == pytest.approx(orig.strategy_stats.total_return, abs=1e-9)
    assert cmp.stats.loc["sma_10_50", "trades"] == orig.strategy_stats.num_trades


def test_compare_symbol_runs_every_rule_on_the_same_window() -> None:
    df = make_ohlcv(random_walk(520, seed=5))
    res = compare_symbol(df, "SPY")
    assert list(res.stats.index) == list(STRATEGIES)
    assert res.start == close_times(df)[252]
    assert all(eq.index[0] == res.start and eq.index[-1] == res.end for eq in res.equity.values())
    bh = res.stats.loc["buy_and_hold"]
    assert bh["trades"] == 1 and bh["time_in_market"] == 1.0
    assert (res.stats["time_in_market"].between(0, 1)).all()


def test_compare_rejects_short_history() -> None:
    with pytest.raises(ValueError):
        compare_symbol(make_ohlcv(random_walk(300)), "SPY")


def test_summary_ranking_and_portfolio() -> None:
    results = [compare_symbol(make_ohlcv(random_walk(520, seed=s)), "SPY") for s in (1, 2)]
    results[1].symbol = "QQQ"
    table = summarise(results)
    assert set(table.index) == set(STRATEGIES)
    assert table["avg_rank"].is_monotonic_increasing
    assert (table.loc["buy_and_hold", ["beats_bh_sharpe", "beats_bh_return"]] == 0).all()
    assert -1 <= rank_stability(results) <= 1
    port = equal_weight_portfolio(results, "buy_and_hold")
    assert port.index.is_monotonic_increasing and len(port) == len(results[0].equity["buy_and_hold"])


def test_sma_cross_names() -> None:
    assert sma_cross(10, 50).__name__ == "sma_10_50"
