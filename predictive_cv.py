"""
QuantPulse — Predictive Methodology & Cross-Validation Suite
Harmonizes purged time-series cross-validation and serial-dependence-aware
moving-block bootstrap evaluation.
"""

from __future__ import annotations

from typing import Dict, Generator, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from predictive_validation import (
    PurgedDateFold,
    iter_purged_date_folds,
    evaluate_decision_tree_grid,
)
from predictive_inference import (
    paired_moving_block_bootstrap,
    render_streamlit_bootstrap_result,
)


class PurgedDateTimeSeriesSplit:
    """
    Time-Series Cross-Validation with Date Grouping and 1-Session Target Boundary Purging.

    Key Invariants:
    1. Chronological order: Training fold dates strictly precede validation fold dates.
    2. Date grouping: All stocks sharing the same trading date are assigned to the same fold.
    3. Boundary Purge: The boundary session between train and validation is purged, guaranteeing
       that the maximum target date in the training fold strictly precedes the first date in the
       validation fold (max(train_target_date) < min(val_date)).
    4. Fold-Local Thresholds: Classification tertile boundaries are estimated strictly on each
       fold's training partition.
    """

    def __init__(self, n_splits: int = 5, purge_sessions: int = 1):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.purge_sessions = purge_sessions

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "Date_dt",
        target_date_col: str = "Target_Date",
        target_col: str = "Next_Day_Return",
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generates (train_indices, val_indices) for each fold, delegating to iter_purged_date_folds.
        """
        for fold in iter_purged_date_folds(
            data=df,
            date_col=date_col,
            target_date_col=target_date_col,
            target_col=target_col,
            n_splits=self.n_splits,
            purge_sessions=self.purge_sessions,
        ):
            yield fold.train.index.to_numpy(), fold.validation.index.to_numpy()


def paired_date_block_bootstrap(
    y_true: Iterable,
    pred_a: Iterable,
    pred_b: Iterable,
    dates: Iterable,
    block_lengths: tuple[int, ...] = (3, 5, 10),
    n_boot: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Alias / wrapper for paired_moving_block_bootstrap preserving serial dependence across trading dates.
    """
    return paired_moving_block_bootstrap(
        y_true=y_true,
        y_pred_reference=pred_b,
        y_pred_candidate=pred_a,
        dates=dates,
        block_lengths=block_lengths,
        n_resamples=n_boot,
        random_state=random_state,
    )


def compute_comprehensive_classification_metrics(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    model_name: str,
    split_name: str = "Chronological 80:20 (146 Clusters)",
) -> Dict[str, Union[str, float]]:
    """
    Computes multi-class evaluation metrics including Balanced Accuracy,
    Macro F1, and Weighted F1.
    """
    labels = ["Low", "Medium", "High"]
    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    w_f1 = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    m_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    w_prec = precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    w_rec = recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)

    return {
        "Algorithm": model_name,
        "Validation Split": split_name,
        "Accuracy (%)": round(acc, 2),
        "Balanced Acc (%)": round(bal_acc, 2),
        "Macro F1": round(m_f1, 4),
        "Weighted F1": round(w_f1, 4),
        "F1-Score": round(w_f1, 4),
        "Precision": round(w_prec, 4),
        "Recall": round(w_rec, 4),
    }
