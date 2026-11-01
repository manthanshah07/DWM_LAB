from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from predictive_validation import iter_purged_date_folds, evaluate_decision_tree_grid

HERE = Path(__file__).resolve().parent
_candidates = [HERE / "stock_market_data.csv", HERE.parent / "stock_market_data.csv"]
DATA_PATH = next((p for p in _candidates if p.is_file()), _candidates[0])


class FoldSafeValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_PATH.is_file():
            raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
        raw = pd.read_csv(DATA_PATH)
        raw["Date_dt"] = pd.to_datetime(raw["Date"])
        raw = raw.sort_values(["Ticker", "Date_dt"]).reset_index(drop=True)
        groups = []
        for _, g0 in raw.groupby("Ticker", sort=False):
            g = g0.copy().reset_index(drop=True)
            g["Next_Day_Return"] = (g["Close_Price"].shift(-1) - g["Close_Price"]) / g["Close_Price"] * 100
            g["Target_Date"] = g["Date_dt"].shift(-1)
            g["Lag1_Return"] = (g["Close_Price"] - g["Close_Price"].shift(1)) / g["Close_Price"].shift(1) * 100
            g["Lag1_Volatility"] = (g["High_Price"] - g["Low_Price"]) / g["Open_Price"] * 100
            g["Lag1_Volume_Log"] = np.log1p(g["Volume"])
            g["Month"] = g["Date_dt"].dt.month
            g["SMA5_Return"] = g["Lag1_Return"].rolling(5).mean()
            groups.append(g)
        frame = pd.concat(groups, ignore_index=True)
        frame = frame.dropna(subset=["Next_Day_Return", "Target_Date", "Lag1_Return", "Lag1_Volatility", "Lag1_Volume_Log", "Month", "SMA5_Return"])
        unique_dates = sorted(frame["Date_dt"].unique())
        cut = int(len(unique_dates) * 0.8)
        outer_train_dates = set(unique_dates[:cut])
        outer_train = frame[frame["Date_dt"].isin(outer_train_dates)].copy()
        outer_test_start = outer_train["Target_Date"].max()
        cls.train = outer_train.reset_index(drop=True)

    def test_all_folds_are_date_grouped_and_purged(self):
        folds = list(iter_purged_date_folds(self.train))
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertLess(fold.train["Target_Date"].max(), fold.validation["Date_dt"].min())
            self.assertFalse(set(fold.train["Date_dt"]).intersection(set(fold.validation["Date_dt"])))
            self.assertEqual(fold.train.groupby("Date_dt").size().nunique(), 1)
            self.assertEqual(fold.validation.groupby("Date_dt").size().nunique(), 1)

    def test_fold_thresholds_are_learned_from_that_fold_train_only(self):
        folds = list(iter_purged_date_folds(self.train))
        fold = folds[0]
        expected_low = fold.train["Next_Day_Return"].quantile(1 / 3)
        expected_high = fold.train["Next_Day_Return"].quantile(2 / 3)
        self.assertAlmostEqual(fold.q_low, expected_low)
        self.assertAlmostEqual(fold.q_high, expected_high)
        # The validation period's own target quantiles should generally differ.
        val_low = fold.validation["Next_Day_Return"].quantile(1 / 3)
        val_high = fold.validation["Next_Day_Return"].quantile(2 / 3)
        self.assertTrue(abs(fold.q_low - val_low) + abs(fold.q_high - val_high) > 0)

    def test_small_grid_runs_and_returns_all_reporting_metrics(self):
        result = evaluate_decision_tree_grid(
            self.train, depths=(3, 4), min_samples_leaf_values=(30,),
            criteria=("entropy",), n_splits=5, purge_sessions=1,
        )
        self.assertEqual(len(result), 2)
        for column in ["Mean Fold Accuracy (%)", "Std Fold Accuracy (pp)", "Mean Fold Macro F1",
                       "Mean Fold Balanced Accuracy (%)", "Pooled OOF Accuracy (%)"]:
            self.assertIn(column, result.columns)
        self.assertTrue(result["Mean Fold Accuracy (%)"].between(0, 100).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
