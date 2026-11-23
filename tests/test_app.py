"""Tests for the shared pipeline and the Streamlit dashboard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data_loader import DEFAULT_COST_RATE
from src.pipeline import TRADE_COLUMNS
from src.pipeline import run_pipeline
from tests.conftest import make_ohlcv

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
HAS_DATA = (ROOT / "data" / "raw" / "SPY_1d.csv").exists()


def test_pipeline_aligns_curves_and_reconciles() -> None:
    t = np.arange(300)
    closes = 300 + 40 * np.sin(2 * np.pi * t / 120)
    result = run_pipeline(make_ohlcv(closes), fast=10, slow=50, trade_size=0.5, capital=100_000)

    assert len(result.closes) == 300
    assert result.equity.index.equals(result.closes.index)
    assert result.buy_and_hold.index.equals(result.closes.index)
    assert list(result.trades.columns) == TRADE_COLUMNS
    assert result.strategy_stats.num_trades == len(result.trades) > 0
    expected = 100_000 + result.trades["net_pnl_usd"].sum()
    assert result.strategy_stats.final_equity == pytest.approx(expected, abs=0.05)


@pytest.mark.skipif(not HAS_DATA, reason="data/raw has no SPY data; run scripts/download_data.py")
def test_streamlit_app_renders_default_backtest() -> None:
    from streamlit.testing.v1 import AppTest

    from src.data_loader import load_bars

    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]

    # Defaults: SPY, latest calendar year in the data, 10/50, 99% of cash, default cost.
    df = load_bars(ROOT / "data" / "raw", "SPY")
    year = df.index[-1].year
    window = df.loc[f"{year}-01-01":]
    expected = run_pipeline(window, 10, 50, capital=100_000, symbol="SPY", position_fraction=0.99,
                            cost_rate=round(DEFAULT_COST_RATE * 1e4) / 1e4)

    labels = {m.label: m.value for m in at.metric}
    assert labels["Trades"] == str(expected.strategy_stats.num_trades)
    assert labels["Total return"] == f"{expected.strategy_stats.total_return * 100:+.2f}%"
    assert len(at.tabs) == 3
