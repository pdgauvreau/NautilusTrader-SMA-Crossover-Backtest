"""Tests for Stage 2: features, labels, purged walk-forward and the ML strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.evaluate import backtest_probabilities
from src.ml.features import FEATURES
from src.ml.features import WARMUP_BARS
from src.ml.features import build_dataset
from src.ml.features import make_features
from src.ml.features import make_labels
from src.ml.walk_forward import MLConfig
from src.ml.walk_forward import pooled_auc
from src.ml.walk_forward import walk_forward_predict
from src.strategies.ml_signal import MLSignalConfig
from src.strategies.ml_signal import ml_decision
from src.strategies.sma_crossover import Signal
from tests.conftest import close_times
from tests.conftest import make_ohlcv


def random_walk(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 300 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))


# ------------------------------------------------------------------------ features
def test_features_do_not_use_future_data() -> None:
    df = make_ohlcv(random_walk(400))
    full = make_features(df)
    # Wildly change everything after row 300: rows 0..300 must not move.
    tampered = df.copy()
    tampered.iloc[301:, :5] *= 3.0
    assert np.allclose(full.iloc[:301].values, make_features(tampered).iloc[:301].values, equal_nan=True)
    assert np.allclose(full.iloc[:301].values, make_features(df.iloc[:301]).values, equal_nan=True)


def test_features_are_indexed_by_bar_close_and_ready_after_warmup() -> None:
    df = make_ohlcv(random_walk(300))
    f = make_features(df)
    assert list(f.columns) == FEATURES
    assert (f.index == close_times(df)).all()
    assert f.iloc[WARMUP_BARS:].notna().all().all()
    assert f.iloc[: WARMUP_BARS - 1]["close_sma200"].isna().all()


def test_labels_are_forward_returns() -> None:
    closes = np.array([100, 110, 99, 120, 130, 125, 140], dtype=float)
    lab = make_labels(make_ohlcv(closes), horizon=2)
    assert lab["fwd_ret"].iloc[0] == pytest.approx(np.log(99 / 100))
    assert lab["label"].iloc[0] == 0.0
    assert lab["label"].iloc[1] == 1.0  # 110 -> 120
    assert lab["fwd_ret"].iloc[-2:].isna().all()


def test_build_dataset_stacks_symbols_without_warmup_rows() -> None:
    frames = {"SPY": make_ohlcv(random_walk(300, 1)), "QQQ": make_ohlcv(random_walk(260, 2))}
    ds = build_dataset(frames, horizon=5)
    assert ds.groupby("symbol").size().to_dict() == {"QQQ": 60, "SPY": 100}
    assert ds[FEATURES].notna().all().all()
    assert ds.index.is_monotonic_increasing


# -------------------------------------------------------------------- walk-forward
def synthetic_dataset(n_days: int = 1200, signal: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Random features; the label depends on ``ret_1`` with strength ``signal``."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2019-01-01", periods=n_days, freq="D", tz="UTC")
    x = pd.DataFrame(rng.normal(size=(n_days, len(FEATURES))), index=idx, columns=FEATURES)
    logit = signal * x["ret_1"] + rng.normal(size=n_days)
    x["fwd_ret"] = logit * 0.01
    x["label"] = (logit > 0).astype(float)
    x.insert(0, "symbol", "SPY")
    return x


def test_walk_forward_purges_and_predicts_only_the_future() -> None:
    cfg = MLConfig("logreg", horizon=5, train_days=300, test_days=100)
    res = walk_forward_predict(synthetic_dataset(), cfg)
    for f in res.folds.itertuples():
        # Training ends at least `horizon` days before testing starts.
        assert (pd.Timestamp(f.test_start) - pd.Timestamp(f.train_end)).days > cfg.horizon
        assert pd.Timestamp(f.train_start) < pd.Timestamp(f.train_end)
    preds = res.predictions
    assert preds.index.min() >= pd.Timestamp(res.folds["test_start"].iloc[0], tz="UTC")
    assert not preds.reset_index().duplicated(["close_time", "symbol"] if "close_time" in
                                              preds.reset_index() else ["index", "symbol"]).any()
    assert preds["prob"].between(0, 1).all()


@pytest.mark.parametrize("model", ["logreg", "lgbm"])
def test_model_finds_a_planted_signal(model: str) -> None:
    res = walk_forward_predict(synthetic_dataset(signal=3.0), MLConfig(model, train_days=300, test_days=100))
    assert pooled_auc(res.predictions) > 0.8


def test_model_finds_nothing_in_noise_and_shuffled_labels_kill_signal() -> None:
    noise = walk_forward_predict(synthetic_dataset(signal=0.0), MLConfig("logreg", train_days=300, test_days=100))
    assert abs(pooled_auc(noise.predictions) - 0.5) < 0.06
    shuffled = walk_forward_predict(
        synthetic_dataset(signal=3.0), MLConfig("logreg", train_days=300, test_days=100, shuffle_labels=True),
    )
    assert abs(pooled_auc(shuffled.predictions) - 0.5) < 0.06


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward_predict(synthetic_dataset(), MLConfig("xgboost"))


# ------------------------------------------------------------------------ strategy
@pytest.mark.parametrize(
    ("prob", "is_long", "expected"),
    [
        (None, False, None),
        (0.60, False, Signal.ENTER_LONG),
        (0.51, False, None),  # inside the hysteresis band
        (0.40, False, None),  # already flat
        (0.40, True, Signal.EXIT_LONG),
        (0.49, True, None),  # inside the band, stay long
        (0.70, True, None),
    ],
)
def test_ml_decision(prob, is_long, expected) -> None:
    assert ml_decision(prob, is_long, 0.52, 0.48) is expected


def test_ml_config_validates_thresholds(bar_type) -> None:
    with pytest.raises(ValueError):
        MLSignalConfig(bar_type=bar_type, probabilities={}, enter_threshold=0.4, exit_threshold=0.6)


def test_ml_strategy_trades_on_probabilities() -> None:
    df = make_ohlcv(np.linspace(300, 400, 60))
    close_ts = close_times(df)
    probs = pd.Series(np.where(np.arange(60) < 30, 0.7, 0.3), index=close_ts)
    run = backtest_probabilities(df, "SPY", probs)
    assert run.stats["num_trades"] == 1
    assert run.num_signals == 2  # enter on bar 0, exit on bar 30
    # Long through a steady rise, flat afterwards: profit, and flat equity in the second half.
    assert run.stats["total_return"] > 0.1
    assert run.equity.iloc[31:].nunique() == 1
