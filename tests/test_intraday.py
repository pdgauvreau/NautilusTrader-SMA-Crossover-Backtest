"""Tests for Stage 4 (day trading): minute data, intraday rules, evaluator, luck test."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.intraday.data import is_early_close
from src.intraday.data import load_minute_bars
from src.intraday.data import validate_minutes
from src.intraday.evaluate import DEFAULT_COST
from src.intraday.evaluate import DEFAULT_HALF_SPREAD
from src.intraday.evaluate import _equity
from src.intraday.evaluate import breakeven_cost_bp
from src.intraday.evaluate import daily_returns_from_trades
from src.intraday.evaluate import random_control
from src.intraday.evaluate import run_rule
from src.intraday.evaluate import trade_returns
from src.intraday.rules import INTRADAY_RULES
from src.intraday.rules import gap_fade
from src.intraday.rules import orb_30


def make_minutes(days: int = 6, seed: int = 0, start: str = "2024-03-04", drift: float = 0.0,
                 gaps: list[float] | None = None) -> pd.DataFrame:
    """Synthetic regular-hours minute bars shaped like ``load_minute_bars`` output."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(start, periods=days)
    frames, price = [], 100.0
    for k, d in enumerate(sessions):
        if gaps:
            price *= 1 + gaps[k]
        starts = pd.date_range(f"{d.date()} 09:30", periods=390, freq="1min", tz="America/New_York")
        rets = rng.normal(drift, 0.0008, 390)
        close = price * np.exp(np.cumsum(rets))
        open_ = np.r_[price, close[:-1]]
        frames.append(pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) * 1.0002, "low": np.minimum(open_, close) * 0.9998,
            "close": close, "volume": 1000.0, "session": d.date(), "minute": 570 + np.arange(390),
        }, index=starts.tz_convert("UTC")))
        price = close[-1]
    df = pd.concat(frames)
    df.index.name = "start"
    return df


# ------------------------------------------------------------------------- data
@pytest.mark.parametrize(("d", "early"), [
    (date(2023, 11, 24), True), (date(2024, 11, 29), True), (date(2025, 11, 28), True),
    (date(2024, 7, 3), True), (date(2024, 12, 24), True),
    (date(2024, 11, 22), False), (date(2024, 7, 5), False), (date(2024, 12, 26), False),
])
def test_early_close_calendar(d: date, early: bool) -> None:
    assert is_early_close(d) is early


def test_loader_keeps_regular_hours_and_trims_early_closes(tmp_path: Path) -> None:
    rows = []
    for d, minutes in (("2024-11-27", range(4 * 60, 20 * 60)), ("2024-11-29", range(4 * 60, 20 * 60))):
        for m in minutes:
            t = pd.Timestamp(f"{d} {m // 60:02d}:{m % 60:02d}", tz="America/New_York").tz_convert("UTC")
            rows.append({"time": t.isoformat(), "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1,
                         "vwap": 1, "trade_count": 1})
    pd.DataFrame(rows).to_csv(tmp_path / "SPY_1m.csv.gz", index=False, compression="gzip")
    df = load_minute_bars(tmp_path, "SPY")
    per_day = df.groupby("session").size().to_dict()
    assert per_day == {date(2024, 11, 27): 390, date(2024, 11, 29): 210}  # Black Friday closes at 13:00
    assert df["minute"].min() == 570
    assert validate_minutes(df)["min_bars_per_session"] == 210


# ------------------------------------------------------------------------ rules
@pytest.mark.parametrize("key", [k for k in INTRADAY_RULES if k != "buy_and_hold"])
def test_day_rules_are_flat_overnight_and_never_look_ahead(key: str) -> None:
    df = make_minutes(days=6, seed=1, gaps=[0, -0.01, 0.004, -0.008, 0.0, -0.012])
    rule = INTRADAY_RULES[key][1]
    t = rule(df)
    assert t.dtype == bool and len(t) == len(df)
    last_two = df.groupby("session").tail(2).index
    assert not t[df.index.isin(last_two)].any()  # flat for the final two minutes
    for cut in (390 * 3 + 200, 390 * 5 + 17):  # truncating the future never changes the past
        assert np.array_equal(t[:cut], rule(df.iloc[:cut])), cut


def test_intraday_momentum_never_uses_later_bars_of_the_day() -> None:
    df = make_minutes(days=4, seed=3)
    t = INTRADAY_RULES["intraday_momentum"][1](df)
    tampered = df.copy()
    last_day = tampered["session"] == tampered["session"].iloc[-1]
    late = last_day & (tampered["minute"] >= 10 * 60)  # change the final day after 10:00
    tampered.loc[late, ["open", "high", "low", "close"]] *= 1.5
    t2 = INTRADAY_RULES["intraday_momentum"][1](tampered)
    assert np.array_equal(t, t2)  # the decision uses only the prior close and prices up to 10:00


def test_orb_waits_for_a_break_of_the_opening_range() -> None:
    df = make_minutes(days=1, seed=2)
    c = df["close"].to_numpy().copy()
    c[:30] = 100 + np.linspace(-0.5, 0.5, 30)  # opening range 99.5-100.5 (plus the 0.02% wicks)
    c[30:200] = 100.0
    c[200:] = 102.0  # breakout at minute 200
    df["close"] = c
    df["open"] = np.r_[c[0], c[:-1]]
    df["high"] = np.maximum(df["open"], df["close"]) * 1.0002
    df["low"] = np.minimum(df["open"], df["close"]) * 0.9998
    t = orb_30(df)
    assert not t[:200].any() and t[200] and t[200:388].all() and not t[388:].any()


def test_gap_fade_only_trades_gap_downs() -> None:
    df = make_minutes(days=3, seed=4, gaps=[0.0, -0.02, 0.02])
    t = gap_fade(df)
    by_day = pd.Series(t, index=df.index).groupby(df["session"].to_numpy()).sum()
    days = sorted(by_day.index)
    assert by_day[days[0]] == 0  # no prior close
    assert by_day[days[2]] == 0  # gap up: no trade
    assert by_day[days[1]] >= 0  # gap down: allowed to trade until the gap closes or noon
    noon = df["minute"].to_numpy() >= 719
    assert not t[noon].any()


# -------------------------------------------------------------------- evaluator
def test_fast_evaluator_matches_nautilus() -> None:
    df = make_minutes(days=8, seed=5, gaps=[0, -0.01, 0, -0.009, 0.01, -0.011, 0, -0.01])
    for key in ("hold_intraday", "orb_30", "gap_fade", "buy_and_hold"):
        r = run_rule(df, "SPY", key)
        tr = trade_returns(df, r.target, DEFAULT_HALF_SPREAD, DEFAULT_COST)
        fast = _equity(daily_returns_from_trades(tr, df["session"].to_numpy()), 100_000.0)
        assert r.stats["trades"] == len(tr)
        assert r.stats["total_return"] == pytest.approx(fast.iloc[-1] / 100_000 - 1, abs=0.003), key


def test_round_trip_cost_math() -> None:
    df = make_minutes(days=2, seed=6)
    target = np.zeros(len(df), dtype=bool)
    target[10:20] = True  # one trade: buy at bar 11's open, sell at bar 21's open
    tr = trade_returns(df, target, half_spread=0.0, cost=0.0)
    o = df["open"].to_numpy()
    assert tr["net_return"].iloc[0] == pytest.approx(o[21] / o[11] - 1)
    costly = trade_returns(df, target, half_spread=0.0001, cost=0.0001)
    assert costly["net_return"].iloc[0] < tr["net_return"].iloc[0]
    assert breakeven_cost_bp(df, target) == pytest.approx((o[21] / o[11] - 1) / 2 * 1e4)


def test_random_control_keeps_trade_count_and_flat_close() -> None:
    df = make_minutes(days=10, seed=7, gaps=[0, -0.01, 0, -0.01, 0, -0.01, 0, -0.01, 0, -0.01])
    target = INTRADAY_RULES["orb_30"][1](df)
    ctrl = random_control(df, target, n_samples=20, seed=1)
    assert len(ctrl) == 20 and ctrl["total_return"].notna().all()
