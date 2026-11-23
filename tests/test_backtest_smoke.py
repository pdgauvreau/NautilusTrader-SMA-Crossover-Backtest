"""Smoke tests: the full backtest runs, accounts consistently, and writes its report."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from src.backtest import realized_pnls
from src.backtest import run_backtest
from tests.conftest import STARTING_CAPITAL
from tests.conftest import money
from tests.conftest import sine_closes

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SCRIPT = ROOT / "scripts" / "run_backtest.py"


@pytest.fixture(scope="module")
def synthetic_run(make_bars, instrument, bar_type):
    from src.strategies.sma_crossover import SmaCrossoverConfig

    bars = make_bars(sine_closes(300, period=100, amplitude=50.0))
    out = run_backtest(bars, instrument, SmaCrossoverConfig(bar_type=bar_type), STARTING_CAPITAL)
    return bars, out


def test_strategy_sees_every_bar(synthetic_run) -> None:
    bars, out = synthetic_run
    # Nautilus only logs handler exceptions, so a crash would show up as missing bars.
    assert out.bars_seen == len(bars)


def test_equity_curve_has_one_point_per_bar(synthetic_run) -> None:
    bars, out = synthetic_run
    assert abs(len(out.equity) - len(bars)) <= 1
    assert out.equity.index.is_monotonic_increasing
    assert out.equity.iloc[0] == pytest.approx(STARTING_CAPITAL)


def test_completes_round_trips_and_ends_flat(synthetic_run) -> None:
    _, out = synthetic_run
    assert len(out.positions) >= 1
    assert (out.positions["side"] == "FLAT").all()
    buys = out.fills[out.fills["side"] == "BUY"]
    sells = out.fills[out.fills["side"] == "SELL"]
    assert len(buys) >= 1
    assert len(buys) == len(sells)


def test_fees_are_charged(synthetic_run) -> None:
    _, out = synthetic_run
    pos = out.positions
    qty = pos["peak_qty"].astype(str).astype(float)
    gross = ((pos["avg_px_close"].astype(float) - pos["avg_px_open"].astype(float)) * qty).sum()
    net = sum(realized_pnls(pos))
    commissions = sum(money(c) for cs in out.fills["commissions"] for c in cs)

    assert commissions > 0
    assert net < gross
    assert gross - net == pytest.approx(commissions, abs=0.01)


def test_final_equity_equals_capital_plus_realized_pnl(synthetic_run) -> None:
    _, out = synthetic_run
    expected = STARTING_CAPITAL + sum(realized_pnls(out.positions))
    assert out.equity.iloc[-1] == pytest.approx(expected, abs=0.01)


def _load_script():
    spec = importlib.util.spec_from_file_location("run_backtest_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not (RAW_DIR / "SPY_1d.csv").exists(), reason="no SPY data in data/raw/ (needs Alpaca keys)")
def test_run_backtest_script_end_to_end(tmp_path: Path) -> None:
    script = _load_script()
    report_dir = tmp_path / "reports"

    assert script.main(["--symbols", "SPY", "--raw-dir", str(RAW_DIR), "--report-dir", str(report_dir)]) == 0

    for name in ("SPY_summary.md", "SPY_metrics.json", "SPY_trades.csv", "SPY_equity_curve.png", "summary.md"):
        path = report_dir / name
        assert path.is_file() and path.stat().st_size > 0, name

    metrics = json.loads((report_dir / "SPY_metrics.json").read_text(encoding="utf-8"))
    assert {"strategy", "buy_and_hold", "parameters"} <= metrics.keys()
    assert metrics["parameters"]["fast"] == 10 and metrics["parameters"]["slow"] == 50
    assert "# SPY backtest summary" in (report_dir / "SPY_summary.md").read_text(encoding="utf-8")


def test_run_backtest_script_on_synthetic_csv(tmp_path: Path) -> None:
    """End-to-end through the CLI with a synthetic SPY CSV in the real Alpaca format."""
    import numpy as np
    import pandas as pd

    raw = tmp_path / "raw"
    raw.mkdir()
    days = pd.bdate_range("2020-01-01", periods=400)
    close = 300 + 30 * np.sin(np.arange(400) / 15)
    frame = pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "open": close, "high": close * 1.01,
                          "low": close * 0.99, "close": close, "volume": 1e6, "vwap": close, "trade_count": 100})
    frame.to_csv(raw / "SPY_1d.csv", index=False)
    report_dir = tmp_path / "reports"
    assert _load_script().main(["--symbols", "SPY", "--raw-dir", str(raw), "--report-dir", str(report_dir)]) == 0
    metrics = json.loads((report_dir / "SPY_metrics.json").read_text(encoding="utf-8"))
    assert metrics["strategy"]["num_trades"] > 0
    assert (report_dir / "SPY_equity_curve.png").stat().st_size > 0
