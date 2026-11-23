"""Purged walk-forward training: every prediction comes from a model that never saw its future.

For each fold:

* **train** on all symbols' rows dated ``[train_start, test_start - horizon)``.
  The last ``horizon`` days before the test window are *purged*: their labels
  are forward returns that would overlap the test period.
* **predict** every row in ``[test_start, test_start + test_days)``.
* roll forward by ``test_days``.

Hyperparameters are fixed up front and deliberately conservative (shallow
trees, many samples per leaf). They are *not* tuned on this data, because tuning
them here would just be a slower way to overfit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.features import FEATURES

MODELS = ("lgbm", "logreg")


@dataclass(frozen=True)
class MLConfig:
    model: str = "lgbm"
    horizon: int = 5  # label = sign of the next `horizon`-day return
    train_days: int = 730
    test_days: int = 180
    min_test_days: int = 60
    seed: int = 7
    shuffle_labels: bool = False  # sanity check: should destroy any edge


def make_model(kind: str, seed: int = 7):
    if kind == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
        )
    if kind == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
    raise ValueError(f"Unknown model {kind!r}; choose from {MODELS}")


@dataclass
class WalkForwardPredictions:
    predictions: pd.DataFrame  # index close time; symbol, prob, fwd_ret, label, fold
    folds: pd.DataFrame  # one row per fold with sizes and test AUC/accuracy
    importance: pd.Series  # mean feature importance (LightGBM gain share), empty for logreg


def walk_forward_predict(dataset: pd.DataFrame, cfg: MLConfig = MLConfig()) -> WalkForwardPredictions:
    """Out-of-sample probabilities that the next ``horizon``-day return is positive."""
    dates = dataset.index.unique().sort_values()
    day = pd.Timedelta(days=1)
    rng = np.random.default_rng(cfg.seed)

    preds, folds, importances = [], [], []
    test_start = dates[0] + cfg.train_days * day
    while test_start + cfg.min_test_days * day <= dates[-1] + day:
        test_end = test_start + cfg.test_days * day  # exclusive
        train_start = test_start - cfg.train_days * day
        purge_from = test_start - cfg.horizon * day

        train = dataset[(dataset.index >= train_start) & (dataset.index < purge_from)].dropna(subset=["label"])
        test = dataset[(dataset.index >= test_start) & (dataset.index < test_end)]
        if train.empty or test.empty:
            break

        y = train["label"].to_numpy()
        if cfg.shuffle_labels:
            y = rng.permutation(y)
        model = make_model(cfg.model, cfg.seed)
        model.fit(train[FEATURES], y)
        prob = model.predict_proba(test[FEATURES])[:, 1]

        fold_no = len(folds) + 1
        out = test[["symbol", "fwd_ret", "label"]].copy()
        out["prob"] = prob
        out["fold"] = fold_no
        preds.append(out)

        known = out.dropna(subset=["label"])
        auc = roc_auc_score(known["label"], known["prob"]) if known["label"].nunique() == 2 else np.nan
        folds.append(
            {
                "fold": fold_no,
                "train_start": train_start.date(),
                "train_end": (purge_from - day).date(),
                "test_start": test_start.date(),
                "test_end": (min(test_end, dates[-1] + day) - day).date(),
                "n_train": len(train),
                "n_test": len(test),
                "base_rate": float(known["label"].mean()),
                "auc": float(auc),
                "accuracy": float(((known["prob"] > 0.5) == (known["label"] > 0.5)).mean()),
                "mean_prob": float(out["prob"].mean()),
            },
        )
        if cfg.model == "lgbm":
            gain = model.booster_.feature_importance(importance_type="gain")
            importances.append(pd.Series(gain / gain.sum(), index=FEATURES))

        test_start = test_end

    if not folds:
        raise ValueError("Not enough data for a single walk-forward fold")
    importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False) if importances \
        else pd.Series(dtype=float)
    return WalkForwardPredictions(pd.concat(preds).sort_index(kind="stable"), pd.DataFrame(folds), importance)


def pooled_auc(predictions: pd.DataFrame) -> float:
    known = predictions.dropna(subset=["label"])
    return float(roc_auc_score(known["label"], known["prob"]))
