"""Tests for Stage 1 research features: sizing, delay, trade start, schedules, walk-forward."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import bar_type_for
from src.data_loader import equity_instrument
from src.pipeline import run_pipeline
from src.research import ResearchSettings
from src.research import param_grid
from src.research import random_schedule
from src.research import walk_forward
from tests.conftest import close_times
from tests.conftest import make_ohlcv
from tests.conftest import sine_closes
from tests.conftest import step_closes

ROOT = Path(__file__).resolve().parents[1]


def _first_buy_index(result) -> int:
    """Index (into closes) of the first BUY signal."""
    ts = pd.Timestamp(result.output.signals[0][0], unit="ns", tz="UTC")
    return int(result.closes.index.get_loc(ts))


# --------------------------------------------------------------- strategy options
def test_execution_delay_shifts_entry_by_n_bars() -> None:
    df = make_ohlcv(step_closes(decline_bars=50, total_bars=80))
    on_time = run_pipeline(df, 10, 50)
    late = run_pipeline(df, 10, 50, execution_delay_bars=2)
    assert _first_buy_index(late) == _first_buy_index(on_time) + 2


def test_position_fraction_sizes_by_cash() -> None:
    df = make_ohlcv(step_closes(decline_bars=50, total_bars=80))
    res = run_pipeline(df, 10, 50, capital=100_000, position_fraction=0.5)
    trade = res.trades.iloc[0]
    notional = trade["entry_px"] * trade["quantity"]
    # Whole shares, bought at the next open's ask: within one share of half the cash.
    assert 50_000 - trade["entry_px"] <= notional <= 50_000


def test_trade_start_blocks_earlier_trades_and_trims_curves() -> None:
    df = make_ohlcv(sine_closes(300))
    start = close_times(df)[150]
    res = run_pipeline(df, 10, 50, position_fraction=0.99, trade_start=start)
    assert res.closes.index[0] == start
    assert res.equity.index[0] == start
    assert all(pd.Timestamp(ts, unit="ns", tz="UTC") >= start for ts, _ in res.output.signals)


def test_enter_if_above_on_start_joins_existing_uptrend() -> None:
    # Prices rise steadily, so fast > slow long before trade_start and no new cross happens.
    df = make_ohlcv(np.linspace(200, 400, 200))
    start = close_times(df)[120]
    waits = run_pipeline(df, 10, 50, position_fraction=0.99, trade_start=start)
    joins = run_pipeline(df, 10, 50, position_fraction=0.99, trade_start=start, enter_if_above_on_start=True)
    assert waits.output.signals == []
    assert pd.Timestamp(joins.output.signals[0][0], unit="ns", tz="UTC") == start


def test_config_rejects_bad_research_options(bar_type) -> None:
    from src.strategies.sma_crossover import SmaCrossoverConfig

    with pytest.raises(ValueError):
        SmaCrossoverConfig(bar_type=bar_type, position_fraction=1.5)
    with pytest.raises(ValueError):
        SmaCrossoverConfig(bar_type=bar_type, execution_delay_bars=-1)


# ------------------------------------------------------------------- instruments
@pytest.mark.parametrize(("symbol", "venue"), [("SPY", "ARCA"), ("QQQ", "NASDAQ"), ("IWM", "ARCA"),
                                              ("NVDA", "NASDAQ"), ("SOFI", "NASDAQ")])
def test_equity_instruments(symbol: str, venue: str) -> None:
    inst = equity_instrument(symbol)
    assert str(inst.id) == f"{symbol}.{venue}"
    assert inst.price_precision == 4 and inst.size_precision == 0
    assert str(inst.quote_currency) == "USD"
    assert str(bar_type_for(symbol)) == f"{symbol}.{venue}-1-DAY-LAST-EXTERNAL"


def test_unknown_symbol_raises() -> None:
    with pytest.raises(ValueError):
        equity_instrument("BTCUSDT")


def test_cost_rate_increases_fees() -> None:
    df = make_ohlcv(sine_closes(300))
    cheap = run_pipeline(df, 10, 50, position_fraction=0.99, cost_rate=0.0001)
    dear = run_pipeline(df, 10, 50, position_fraction=0.99, cost_rate=0.003)
    assert dear.strategy_stats.final_equity < cheap.strategy_stats.final_equity
    assert dear.buy_and_hold_stats.final_equity < cheap.buy_and_hold_stats.final_equity


# ---------------------------------------------------------------- random schedule
@pytest.mark.parametrize("seed", range(5))
def test_random_schedule_preserves_holds_and_never_overlaps(seed: int) -> None:
    holds = [5, 30, 12, 1, 60]
    entries, exits = random_schedule(400, holds, np.random.default_rng(seed), first_bar=49)
    pairs = sorted(zip(sorted(entries), sorted(exits)))
    assert sorted(x - e for e, x in pairs) == sorted(holds)
    assert pairs[0][0] >= 49 and pairs[-1][1] < 400
    for (_, prev_exit), (next_entry, _) in zip(pairs, pairs[1:]):
        assert next_entry > prev_exit


def test_random_schedule_rejects_impossible_layout() -> None:
    with pytest.raises(ValueError):
        random_schedule(50, [40, 40], np.random.default_rng(0))


# -------------------------------------------------------------------- walk-forward
def test_walk_forward_folds_are_sequential_and_out_of_sample() -> None:
    df = make_ohlcv(sine_closes(700, period=60))
    grid = param_grid((5, 10), (20, 30))
    wf = walk_forward(df, "SPY", grid, ResearchSettings(), train_bars=200, test_bars=100)

    folds = wf.folds
    assert len(folds) >= 3
    # Each test window starts after its training window and test windows don't overlap.
    assert (pd.to_datetime(folds["test_start"]) > pd.to_datetime(folds["train_start"])).all()
    assert (pd.to_datetime(folds["test_start"]).iloc[1:].values
            > pd.to_datetime(folds["test_end"]).iloc[:-1].values).all()
    assert set(zip(folds["fast"], folds["slow"])) <= set(grid)
    # Stitched curve covers exactly the test bars, with no duplicates.
    assert wf.equity.index.is_unique and wf.equity.index.is_monotonic_increasing
    assert len(wf.equity) == len(wf.buy_and_hold)
    assert wf.equity.iloc[0] == pytest.approx(100_000, rel=0.01)


# ------------------------------------------------------------------------ CLI args
def test_downloader_args() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import download_data

    args = download_data.parse_args(["--symbols", "spy", "QQQ", "--start", "2018-01-01"])
    assert args.symbols == ["SPY", "QQQ"] and args.start == "2018-01-01"
    from src.data_loader import DEFAULT_SYMBOLS

    assert download_data.parse_args([]).symbols == list(DEFAULT_SYMBOLS)
    assert DEFAULT_SYMBOLS[:5] == ("SPY", "QQQ", "IWM", "NVDA", "SOFI")  # public symbols first
    with pytest.raises(SystemExit):
        download_data.parse_args(["--symbols", "../etc"])
    with pytest.raises(SystemExit):
        download_data.parse_args(["--start", "01/02/2020"])
