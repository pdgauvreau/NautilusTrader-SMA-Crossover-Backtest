"""Tests for Stage 2b: per-entry sizing hook and the sizing rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.features import build_dataset
from src.ml.sizing import MAX_FRACTION
from src.ml.sizing import gate_fraction
from src.ml.sizing import long_state_dataset
from src.ml.sizing import ml_fraction
from src.ml.sizing import vol_target_fraction
from src.pipeline import run_pipeline
from src.strategies.sma_crossover import SmaCrossoverConfig
from tests.conftest import make_ohlcv
from tests.conftest import step_closes


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.0, 0.25), (0.3, 0.25), (0.5, 0.5), (0.625, 0.75), (0.75, 1.0), (0.95, 1.0)],
)
def test_ml_fraction_map(p: float, expected: float) -> None:
    assert ml_fraction(p) == pytest.approx(MAX_FRACTION * expected)


def test_gate_fraction() -> None:
    assert gate_fraction(0.49) == 0.0
    assert gate_fraction(0.5) == MAX_FRACTION


def test_vol_target_fraction_scales_down_only_when_vol_is_high() -> None:
    low = 0.10 / np.sqrt(252)  # 10% annualised: below the 15% target, full size
    high = 0.30 / np.sqrt(252)  # 30% annualised: half size
    assert vol_target_fraction(low) == pytest.approx(MAX_FRACTION)
    assert vol_target_fraction(high) == pytest.approx(MAX_FRACTION * 0.5)


def _entry_ts(df: pd.DataFrame) -> int:
    """Close timestamp (ns) of the bar the default 10/50 strategy enters on."""
    res = run_pipeline(df, 10, 50, position_fraction=0.99)
    return res.output.signals[0][0]


def test_entry_fractions_resize_and_skip_entries() -> None:
    df = make_ohlcv(step_closes(decline_bars=50, total_bars=80))
    ts = _entry_ts(df)
    half = run_pipeline(df, 10, 50, position_fraction=0.99, entry_fractions={ts: 0.5})
    trade = half.trades.iloc[0]
    assert 50_000 - trade["entry_px"] <= trade["entry_px"] * trade["quantity"] <= 50_000

    skipped = run_pipeline(df, 10, 50, position_fraction=0.99, entry_fractions={ts: 0.0})
    assert skipped.trades.empty and skipped.output.signals == []

    # Bars missing from the map fall back to position_fraction.
    default = run_pipeline(df, 10, 50, position_fraction=0.99, entry_fractions={})
    t = default.trades.iloc[0]
    assert 99_000 - t["entry_px"] <= t["entry_px"] * t["quantity"] <= 99_000


def test_entry_fractions_validation(bar_type) -> None:
    with pytest.raises(ValueError):
        SmaCrossoverConfig(bar_type=bar_type, entry_fractions={1: 1.5})


def test_long_state_dataset_keeps_only_fast_above_slow() -> None:
    rng = np.random.default_rng(3)
    df = make_ohlcv(300 * np.exp(np.cumsum(rng.normal(0, 0.012, 500))))
    full = build_dataset({"SPY": df}, horizon=20)
    long_only = long_state_dataset({"SPY": df}, horizon=20)
    assert 0 < len(long_only) < len(full)
    assert (long_only["sma_10_50"] > 0).all()
