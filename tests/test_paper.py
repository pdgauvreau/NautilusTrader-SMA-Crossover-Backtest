"""Tests for the paper-trading module (no network: nothing here connects anywhere)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from src.backtest import run_backtest
from src.data_loader import DEFAULT_BAR_TYPE
from src.data_loader import btcusdt_instrument
from src.paper.journal import Journal
from src.paper.journal import read_journal
from src.paper.node import PaperSettings
from src.paper.node import data_client_config
from src.paper.node import sandbox_exec_config
from src.paper.strategy import PaperSmaCrossover
from src.paper.strategy import PaperSmaCrossoverConfig
from tests.conftest import sine_closes

ROOT = Path(__file__).resolve().parents[1]
PAPER_SOURCES = [*(ROOT / "src" / "paper").glob("*.py"), ROOT / "scripts" / "paper_trade.py"]


# ---------------------------------------------------------------------- safety
@pytest.mark.parametrize("path", PAPER_SOURCES, ids=lambda p: p.name)
def test_paper_code_cannot_reach_a_real_exchange(path: Path) -> None:
    src = path.read_text(encoding="utf-8")
    assert "BinanceExecutionClient" not in src  # only the sandbox executes
    assert "api_key" not in src and "api_secret" not in src  # never handles credentials
    # The node itself must be SANDBOX; BinanceEnvironment.LIVE (real public prices) is fine.
    assert re.search(r"(?<!Binance)Environment\.LIVE", src) is None


def test_node_is_built_in_sandbox_environment() -> None:
    src = (ROOT / "src" / "paper" / "node.py").read_text(encoding="utf-8")
    assert "Environment.SANDBOX" in src and "add_simulated_exec_client" in src
    assert ".add_exec_client(" not in src


def test_settings_validation() -> None:
    with pytest.raises(ValueError):
        PaperSettings(region="eu")
    with pytest.raises(ValueError):
        PaperSettings(symbol="BTC/USDT")
    with pytest.raises(ValueError):
        PaperSettings(starting_usdt=0)


def test_node_configs() -> None:
    s = PaperSettings(symbol="ETHUSDT", region="us", starting_usdt=5_000)
    data = data_client_config(s)
    assert data.us is True
    assert data.instrument_provider.load_ids == ["ETHUSDT.BINANCE"]
    ex = sandbox_exec_config(s)
    assert ex.bar_execution is False  # quote-driven fills by default
    assert {m.currency.code: m.as_double() for m in ex.starting_balances} == {"USDT": 5_000.0, "ETH": 0.0}
    assert sandbox_exec_config(PaperSettings(quote_fills=False)).bar_execution is True
    assert data_client_config(PaperSettings(region="com")).us is False


# --------------------------------------------------------------------- journal
def test_journal_appends_and_tolerates_a_torn_last_line(tmp_path: Path) -> None:
    j = Journal(tmp_path / "sub" / "j.jsonl")
    j.write("start", equity=100.0)
    j.write("bar", close=1.5, signal=None)
    with j.path.open("a", encoding="utf-8") as fh:
        fh.write('{"event": "fill", "qty"')  # simulated crash mid-write
    recs = read_journal(j.path)
    assert [r["event"] for r in recs] == ["start", "bar"]
    assert recs[1]["close"] == 1.5 and "ts" in recs[0]


# ------------------------------------------------------- strategy (offline run)
def test_paper_strategy_journals_bars_signals_and_fills(tmp_path: Path) -> None:
    """Run the paper strategy in the backtest engine: same logic, journal written."""
    from tests.conftest import make_ohlcv
    from src.data_loader import to_nautilus_bars
    from nautilus_trader.model import BarType

    instrument = btcusdt_instrument()
    bar_type = BarType.from_str(DEFAULT_BAR_TYPE)
    bars = to_nautilus_bars(make_ohlcv(sine_closes(200)), instrument, bar_type)
    path = tmp_path / "j.jsonl"
    cfg = PaperSmaCrossoverConfig(journal_path=str(path), bar_type=bar_type, position_fraction=0.5,
                                  subscribe_quotes=False)
    out = run_backtest(bars, instrument, cfg, strategy_cls=PaperSmaCrossover)

    recs = read_journal(path)
    events = [r["event"] for r in recs]
    assert events[0] == "start" and events[-1] == "stop"
    bar_recs = [r for r in recs if r["event"] == "bar"]
    assert len(bar_recs) == len(bars) == out.bars_seen
    signals = [r["signal"] for r in bar_recs if r["signal"]]
    assert len(signals) == len(out.signals) > 0
    fills = [r for r in recs if r["event"] == "fill"]
    assert fills and all(r["slippage_vs_close_bps"] == pytest.approx(0, abs=0.01) for r in fills)
    assert bar_recs[0]["fast"] is None and bar_recs[-1]["fast"] is not None


# ---------------------------------------------------------------------- report
def _load_report():
    spec = importlib.util.spec_from_file_location("paper_report", ROOT / "scripts" / "paper_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_report_summary() -> None:
    report = _load_report()
    recs = [
        {"ts": "1", "event": "start", "equity": 10_000.0},
        {"ts": "2", "event": "bar", "bar_time": "t1", "close": 100.0, "fast": 1.0, "slow": 0.5,
         "signal": "enter_long", "position": 0.0, "equity": 10_000.0},
        {"ts": "3", "event": "fill", "side": "BUY", "qty": 1.0, "price": 100.1, "bar_close": 100.0,
         "slippage_vs_close_bps": 10.0, "bid": 99.9, "ask": 100.1, "commission": "0.1 USDT"},
        {"ts": "4", "event": "stop", "equity": 9_999.0, "position": 1.0},
    ]
    s = report.summarise(recs)
    assert s["signals"] == 1 and s["fills"] == 1 and s["position"] == 1.0
    assert s["avg_slippage_bps"] == pytest.approx(10.0)
    assert s["commissions_usdt"] == pytest.approx(0.1)
    assert s["avg_spread_bps"] == pytest.approx(20.0, rel=1e-3)
    assert np.isclose(s["equity_last"], 9_999.0)
