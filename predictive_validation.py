"""Fold-safe walk-forward validation for next-session return regimes.

The class boundaries are re-estimated inside every training fold and then
applied unchanged to that fold's validation labels. This avoids using later
outer-training targets to define earlier validation classes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeClassifier


@dataclass
class PurgedDateFold:
    fold: int
    train: pd.DataFrame
    validation: pd.DataFrame
    q_low: float
    q_high: float
    purged_dates: tuple[pd.Timestamp, ...]


def _assign_regime(values: pd.Series, q_low: float, q_high: float) -> pd.Series:
    return pd.Series(
        np.where(values < q_low, "Low", np.where(values < q_high, "Medium", "High")),
        index=values.index,
        name="_fold_regime",
    )


def iter_purged_date_folds(
    data: pd.DataFrame,
    date_col: str = "Date_dt",
    target_date_col: str = "Target_Date",
    target_col: str = "Next_Day_Return",
    n_splits: int = 5,
    purge_sessions: int = 1,
) -> Iterator[PurgedDateFold]:
    """Yield expanding-window folds with atomic date groups and target purging."""
    required = {date_col, target_date_col, target_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if purge_sessions < 1:
        raise ValueError("purge_sessions must be at least 1 for a next-session target")

    frame = data.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[target_date_col] = pd.to_datetime(frame[target_date_col], errors="coerce")
    if frame[[date_col, target_date_col, target_col]].isna().any().any():
        raise ValueError("Date, Target_Date and target values must be non-null for CV")

    dates = np.array(sorted(frame[date_col].unique()))
    if len(dates) <= n_splits + purge_sessions:
        raise ValueError("Not enough unique dates for requested folds and purge")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    for fold_no, (train_idx, validation_idx) in enumerate(splitter.split(dates), start=1):
        pre_purge_train_dates = dates[train_idx]
        if len(pre_purge_train_dates) <= purge_sessions:
            raise ValueError(f"Fold {fold_no} has no training dates after purge")
        kept_train_dates = pre_purge_train_dates[:-purge_sessions]
        purged_dates = tuple(pd.Timestamp(x) for x in pre_purge_train_dates[-purge_sessions:])
        validation_dates = dates[validation_idx]

        train_fold = frame.loc[frame[date_col].isin(kept_train_dates)].copy()
        validation_fold = frame.loc[frame[date_col].isin(validation_dates)].copy()
        if train_fold.empty or validation_fold.empty:
            raise ValueError(f"Fold {fold_no} is empty")
        if train_fold[date_col].nunique() != len(kept_train_dates):
            raise ValueError(f"Fold {fold_no}: training dates are not fully represented")
        if validation_fold[date_col].nunique() != len(validation_dates):
            raise ValueError(f"Fold {fold_no}: validation dates are not fully represented")
        if train_fold[target_date_col].max() >= validation_fold[date_col].min():
            raise ValueError(f"Fold {fold_no}: target realization overlaps validation dates")

        # Crucial: thresholds are learned from this fold's kept training dates only.
        q_low = float(train_fold[target_col].quantile(1 / 3))
        q_high = float(train_fold[target_col].quantile(2 / 3))
        train_fold["_fold_regime"] = _assign_regime(train_fold[target_col], q_low, q_high)
        validation_fold["_fold_regime"] = _assign_regime(validation_fold[target_col], q_low, q_high)
        yield PurgedDateFold(fold_no, train_fold, validation_fold, q_low, q_high, purged_dates)


def evaluate_decision_tree_grid(
    data: pd.DataFrame,
    feature_cols: tuple[str, ...] = (
        "Lag1_Return", "Lag1_Volatility", "Lag1_Volume_Log", "Month",
    ),
    depths: tuple[int, ...] = (3, 4, 5, 6),
    min_samples_leaf_values: tuple[int, ...] = (20, 30, 40, 50, 75),
    criteria: tuple[str, ...] = ("entropy", "gini"),
    n_splits: int = 5,
    purge_sessions: int = 1,
) -> pd.DataFrame:
    """Evaluate a Decision Tree grid using fold-local tertile thresholds.

    Hyperparameters are ranked by mean fold accuracy, matching the original
    experiment's selection objective. Macro-F1 and balanced accuracy are also
    reported so a majority-class trade-off is visible.
    """
    missing_features = set(feature_cols) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {sorted(missing_features)}")
    folds = list(iter_purged_date_folds(data, n_splits=n_splits, purge_sessions=purge_sessions))
    rows: list[dict] = []

    for depth, min_leaf, criterion in product(depths, min_samples_leaf_values, criteria):
        fold_accuracy: list[float] = []
        fold_macro_f1: list[float] = []
        fold_balanced_accuracy: list[float] = []
        all_true: list[str] = []
        all_pred: list[str] = []
        fold_details: list[str] = []

        for fold in folds:
            X_train = fold.train.loc[:, feature_cols]
            X_val = fold.validation.loc[:, feature_cols]
            if X_train.isna().any().any() or X_val.isna().any().any():
                raise ValueError("Features contain missing values; handle imputation inside each fold")
            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=int(depth),
                min_samples_leaf=int(min_leaf),
                random_state=42,
            )
            y_fold_train = fold.train["_fold_regime"]
            y_fold_val = fold.validation["_fold_regime"]
            model.fit(X_train, y_fold_train)
            pred = model.predict(X_val)
            fold_accuracy.append(float(accuracy_score(y_fold_val, pred)))
            fold_macro_f1.append(float(f1_score(y_fold_val, pred, average="macro", zero_division=0)))
            fold_balanced_accuracy.append(float(balanced_accuracy_score(y_fold_val, pred)))
            all_true.extend(y_fold_val.astype(str).tolist())
            all_pred.extend(pd.Series(pred).astype(str).tolist())
            fold_details.append(f"{accuracy_score(y_fold_val, pred):.4f}")

        rows.append({
            "Max Depth": int(depth),
            "Min Samples Leaf": int(min_leaf),
            "Criterion": criterion,
            "Mean Fold Accuracy (%)": float(np.mean(fold_accuracy) * 100),
            "Std Fold Accuracy (pp)": float(np.std(fold_accuracy, ddof=1) * 100) if len(fold_accuracy) > 1 else 0.0,
            "Mean Fold Macro F1": float(np.mean(fold_macro_f1)),
            "Mean Fold Balanced Accuracy (%)": float(np.mean(fold_balanced_accuracy) * 100),
            "Pooled OOF Accuracy (%)": float(accuracy_score(all_true, all_pred) * 100),
            "Fold Accuracies": ", ".join(fold_details),
            "CV Folds": len(folds),
            "Purge Sessions": purge_sessions,
        })

    return pd.DataFrame(rows).sort_values(
        ["Mean Fold Accuracy (%)", "Mean Fold Macro F1"], ascending=False
    ).reset_index(drop=True)
