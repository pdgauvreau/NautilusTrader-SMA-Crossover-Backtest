"""ETF-specific behaviour: data loading, session timestamps, next-open fills, sizing, forward test."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.alpaca_data import DATA_URL
from src.alpaca_data import drop_unfinished_session
from src.data_loader import DEFAULT_HALF_SPREAD
from src.data_loader import bar_close_times
from src.data_loader import bar_open_times
from src.data_loader import equity_instrument
from src.data_loader import load_bars
from src.data_loader import open_quotes
from src.data_loader import validate_daily
from src.paper.forward import ForwardState
from src.paper.forward import forward_status
from src.pipeline import run_pipeline
from src.strategies.sma_crossover import affordable_qty
from tests.conftest import close_times
from tests.conftest import make_ohlcv
from tests.conftest import sine_closes
from tests.conftest import step_closes

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ timestamps
def test_session_times_follow_new_york_including_dst() -> None:
    idx = pd.DatetimeIndex(["2025-01-02", "2025-07-03"]).tz_localize("UTC")
    closes = bar_close_times(idx)
    opens = bar_open_times(idx)
    assert [t.strftime("%H:%M") for t in closes] == ["21:00", "20:00"]  # EST, then EDT
    assert [t.strftime("%H:%M") for t in opens] == ["14:30", "13:30"]
    assert all(t.tz_convert("America/New_York").strftime("%H:%M") == "16:00" for t in closes)


# ------------------------------------------------------------- loading & checks
def _write_csv(path: Path, dates, close) -> None:
    pd.DataFrame({"date": dates, "open": close, "high": np.asarray(close) * 1.01, "low": np.asarray(close) * 0.99,
                  "close": close, "volume": 1e6, "vwap": close, "trade_count": 10}).to_csv(path, index=False)


def test_load_bars_and_validate(tmp_path: Path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=30).strftime("%Y-%m-%d")
    _write_csv(tmp_path / "SPY_1d.csv", dates, np.linspace(400, 430, 30))
    df = load_bars(tmp_path, "SPY")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert str(df.index.tz) == "UTC" and len(df) == 30
    summary = validate_daily(df)
    assert summary["rows"] == 30 and summary["max_gap_days"] == 3  # weekends are fine
    assert len(load_bars(tmp_path, "SPY", start="2024-01-10", end="2024-01-12")) == 3


def test_validate_rejects_missing_weeks_and_weekend_rows(tmp_path: Path) -> None:
    days = list(pd.bdate_range("2024-01-02", periods=10).strftime("%Y-%m-%d"))
    _write_csv(tmp_path / "SPY_1d.csv", days[:3] + days[8:], np.linspace(400, 410, 5))  # a 7-day hole
    with pytest.raises(ValueError, match="Gaps"):
        validate_daily(load_bars(tmp_path, "SPY"))
    _write_csv(tmp_path / "QQQ_1d.csv", ["2024-01-05", "2024-01-06"], [400.0, 401.0])  # Saturday
    with pytest.raises(ValueError, match="Weekend"):
        validate_daily(load_bars(tmp_path, "QQQ"))


def test_missing_file_and_unknown_symbol(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_bars(tmp_path, "SPY")
    with pytest.raises(ValueError):
        load_bars(tmp_path, "ZZZZ")


# -------------------------------------------------------------------- execution
def test_open_quotes_bracket_the_open() -> None:
    df = make_ohlcv([400.0, 410.0, 420.0])
    q = open_quotes(df, equity_instrument("SPY"))
    assert len(q) == 3
    assert q[1].bid_price.as_double() == pytest.approx(400.0 * (1 - DEFAULT_HALF_SPREAD), abs=0.01)
    assert q[1].ask_price.as_double() == pytest.approx(400.0 * (1 + DEFAULT_HALF_SPREAD), abs=0.01)
    assert pd.Timestamp(q[0].ts_event, unit="ns", tz="UTC") == bar_open_times(df)[0]


def test_signal_on_close_fills_at_next_open_ask() -> None:
    df = make_ohlcv(step_closes(decline_bars=50, total_bars=80))
    res = run_pipeline(df, 10, 50)  # default: execute at the next open
    decision_ts = pd.Timestamp(res.output.signals[0][0], unit="ns", tz="UTC")
    i = list(close_times(df)).index(decision_ts)
    fill = res.output.fills.iloc[0]
    fill_ts = pd.Timestamp(fill["ts_last"], unit="ns", tz="UTC") if isinstance(fill["ts_last"], (int, np.integer)) \
        else pd.Timestamp(fill["ts_last"])
    assert fill_ts == bar_open_times(df)[i + 1]  # the next session's 09:30 open
    expected_ask = df["open"].iloc[i + 1] * (1 + DEFAULT_HALF_SPREAD)
    assert float(str(fill["avg_px"]).split()[0]) == pytest.approx(expected_ask, abs=0.01)


def test_close_execution_mode_still_available() -> None:
    df = make_ohlcv(step_closes(decline_bars=50, total_bars=80))
    res = run_pipeline(df, 10, 50, execute_at_open=False)
    entry = res.trades.iloc[0]
    assert entry["entry_px"] == pytest.approx(600.0)  # the signal bar's close


def test_affordable_qty_rounds_down() -> None:
    inst = equity_instrument("SPY")
    assert affordable_qty(inst, 99_000, 26_700).as_double() == 3  # 3.7 -> 3, never 4
    assert affordable_qty(inst, 1_000, 500).as_double() == 2
    assert affordable_qty(inst, 499, 500).as_double() == 0
    assert affordable_qty(inst, -5, 500).as_double() == 0


def test_equity_reconciles_with_realised_pnl_after_open_fills() -> None:
    df = make_ohlcv(sine_closes(300, period=60))
    res = run_pipeline(df, 10, 50)
    assert res.strategy_stats.num_trades > 0
    expected = 100_000 + res.trades["net_pnl_usd"].sum()
    assert res.strategy_stats.final_equity == pytest.approx(expected, abs=0.05)


# --------------------------------------------------------------- alpaca client
def test_unfinished_session_is_dropped() -> None:
    df = pd.DataFrame({"date": ["2026-11-20", "2026-11-23"]})
    during = pd.Timestamp("2026-11-23 15:59", tz="America/New_York").tz_convert("UTC")
    after = pd.Timestamp("2026-11-23 16:25", tz="America/New_York").tz_convert("UTC")
    assert drop_unfinished_session(df, during)["date"].tolist() == ["2026-11-20"]
    assert drop_unfinished_session(df, after)["date"].tolist() == ["2026-11-20", "2026-11-23"]


def test_missing_keys_give_a_clear_error(monkeypatch, tmp_path: Path) -> None:
    import src.alpaca_data as alpaca

    for k in alpaca.KEY_VARS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(alpaca, "load_dotenv", lambda path=None: None)
    with pytest.raises(SystemExit, match="ALPACA_API_KEY_ID"):
        alpaca.credentials()


def test_dotenv_reader_does_not_override_environment(monkeypatch, tmp_path: Path) -> None:
    import src.alpaca_data as alpaca

    env = tmp_path / ".env"
    env.write_text("# comment\nALPACA_API_KEY_ID=from_file\nALPACA_API_SECRET_KEY='quoted'\n", encoding="utf-8")
    monkeypatch.setenv("ALPACA_API_KEY_ID", "from_env")
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    alpaca.load_dotenv(env)
    assert os.environ["ALPACA_API_KEY_ID"] == "from_env"
    assert os.environ["ALPACA_API_SECRET_KEY"] == "quoted"


def test_alpaca_client_is_market_data_only() -> None:
    assert DATA_URL.startswith("https://data.alpaca.markets/")
    src = (ROOT / "src" / "alpaca_data.py").read_text(encoding="utf-8")
    assert "/orders" not in src and "paper-api" not in src and "api.alpaca.markets" not in src
    assert ".env" in (ROOT / ".gitignore").read_text(encoding="utf-8")


# ---------------------------------------------------------------- forward test
def test_forward_status_trades_only_after_start_and_keeps_position_open() -> None:
    df = make_ohlcv(sine_closes(260, period=60))
    sessions = df.index.strftime("%Y-%m-%d")
    state = ForwardState.new(sessions[150], ["SPY"], capital_per_symbol=10_000)
    st = forward_status(df, "SPY", state)
    start_open = bar_open_times(df)[151]
    assert all(pd.Timestamp(f["time"]) >= start_open for f in st.fills)
    assert st.last_session == sessions[-1]
    assert st.next_open_action in {"BUY", "SELL", "HOLD"}
    buys = sum(f["qty"] for f in st.fills if f["side"] == "BUY")
    sells = sum(f["qty"] for f in st.fills if f["side"] == "SELL")
    assert st.shares == buys - sells >= 0
    if st.shares:
        assert st.equity != pytest.approx(10_000)


def test_forward_state_roundtrip(tmp_path: Path) -> None:
    s = ForwardState.new("2025-06-30", ["SPY", "QQQ"], fast=10, slow=50)
    s.save(tmp_path / "state.json")
    assert ForwardState.load(tmp_path / "state.json") == s


def test_forward_first_day_joins_uptrend() -> None:
    df = make_ohlcv(np.linspace(300, 400, 120))
    state = ForwardState.new(df.index[-1].strftime("%Y-%m-%d"), ["SPY"])
    st = forward_status(df, "SPY", state)
    assert st.trend == "UP" and st.next_open_action == "BUY" and st.shares == 0


def test_forward_state_adds_new_symbols_with_their_own_start(tmp_path: Path) -> None:
    s = ForwardState.new("2025-06-30", ["SPY", "QQQ"])
    s2 = s.with_symbols(["SPY", "QQQ", "NVDA"], {"NVDA": "2026-11-20"})
    assert s2.symbols == ("SPY", "QQQ", "NVDA")
    assert s2.start_for("SPY") == "2025-06-30" and s2.start_for("NVDA") == "2026-11-20"
    s2.save(tmp_path / "state.json")
    assert ForwardState.load(tmp_path / "state.json") == s2
    # Old state files without "starts" still load.
    (tmp_path / "old.json").write_text(
        '{"start_session": "2025-06-30", "symbols": ["SPY"], "fast": 10, "slow": 50, '
        '"capital_per_symbol": 10000.0, "cost_rate": 0.0003, "created": ""}', encoding="utf-8")
    assert ForwardState.load(tmp_path / "old.json").start_for("SPY") == "2025-06-30"


# ------------------------------------------------------------ private symbols
def test_local_symbols_file_parsing(tmp_path: Path) -> None:
    from src.data_loader import load_local_symbols

    f = tmp_path / "symbols.local.csv"
    f.write_text("# my holdings\nTICKER,VENUE\naapl, nasdaq\nIBM,NYSE  # comment\n\n", encoding="utf-8")
    assert load_local_symbols(f) == {"AAPL": "NASDAQ", "IBM": "NYSE"}
    assert load_local_symbols(tmp_path / "missing.csv") == {}
    f.write_text("AAPL,MOON\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_local_symbols(f)


def test_local_symbols_file_is_git_ignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "symbols.local.csv" in gitignore


def test_trim_before_last_trading_gap() -> None:
    from src.alpaca_data import trim_before_last_gap

    dates = ["2022-02-24", "2022-02-25", "2024-10-21", "2024-10-22"]  # 2.5-year suspension
    df = pd.DataFrame({"date": dates, "close": [1.0, 2.0, 20.0, 21.0]})
    out, dropped = trim_before_last_gap(df)
    assert dropped == 2 and out["date"].tolist() == ["2024-10-21", "2024-10-22"]
    weekend = pd.DataFrame({"date": ["2024-01-05", "2024-01-08"], "close": [1.0, 1.0]})
    assert trim_before_last_gap(weekend)[1] == 0
