"""
Unit tests for predictive modelling methodology, fold-local threshold calibration,
time-series validation, date clustering, moving-block bootstrap, and metric reconciliation.
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from app import load_and_prepare_data, train_predictive_models
from predictive_validation import iter_purged_date_folds, evaluate_decision_tree_grid
from predictive_inference import paired_moving_block_bootstrap
from predictive_cv import (
    PurgedDateTimeSeriesSplit,
    compute_comprehensive_classification_metrics,
)


class TestPredictiveMethodology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (
            cls.df,
            cls.stock_df,
            cls.basket_bool,
            cls.df_ml,
            cls.train_df,
            cls.test_df,
            cls.latest_features_df,
            cls.q1,
            cls.q2,
        ) = load_and_prepare_data()

    def test_01_holdout_split_and_purge(self):
        """Verify the chronological 80:20 split and the 1-session holdout purge."""
        unique_dates = sorted(self.df_ml["Date_dt"].unique())
        n_dates = len(unique_dates)
        split_idx = int(n_dates * 0.8)

        # 587 train dates, 146 test dates, 1 boundary purge date
        train_dates = set(unique_dates[:split_idx])
        self.assertEqual(len(train_dates), 587)
        self.assertEqual(len(self.train_df["Date_dt"].unique()), 587)

        max_train_date = self.train_df["Date_dt"].max()
        max_train_target = self.train_df["Target_Date"].max()
        min_test_date = self.test_df["Date_dt"].min()

        # The target date of the last train session must strictly precede the first test date
        self.assertLess(max_train_date, min_test_date)
        self.assertLess(max_train_target, min_test_date)

        # Exactly 1 trading day (2026-03-06) is purged
        purged_dates = set(unique_dates) - set(self.train_df["Date_dt"].unique()) - set(self.test_df["Date_dt"].unique())
        self.assertEqual(len(purged_dates), 1)
        self.assertIn(pd.Timestamp("2026-03-06"), purged_dates)

        # Test set contains exactly 146 unique dates and 2,920 records (20 stocks/date)
        test_dates = self.test_df["Date_dt"].unique()
        self.assertEqual(len(test_dates), 146)
        self.assertEqual(len(self.test_df), 2920)

    def test_02_date_clustering_structure(self):
        """Verify that every date in train and test contains all 20 equities without cross-sectional splits."""
        train_counts = self.train_df.groupby("Date_dt")["Ticker"].count()
        test_counts = self.test_df.groupby("Date_dt")["Ticker"].count()

        # Every trading date in train and test must have all 20 equities
        self.assertTrue((train_counts == 20).all())
        self.assertTrue((test_counts == 20).all())

        # No date overlap between train and test
        overlap = set(self.train_df["Date_dt"]).intersection(set(self.test_df["Date_dt"]))
        self.assertEqual(len(overlap), 0)

    def test_03_fold_local_threshold_calibration_and_invariants(self):
        """Verify that iter_purged_date_folds enforces fold-local thresholds, chronology, and zero target overlap."""
        folds = list(iter_purged_date_folds(self.train_df, n_splits=5, purge_sessions=1))
        self.assertEqual(len(folds), 5)

        total_val_dates = 0
        for fold in folds:
            # 1. Non-empty partitions
            self.assertGreater(len(fold.train), 0)
            self.assertGreater(len(fold.validation), 0)

            # 2. Date grouping: no date appears in both train and validation
            tr_dates = set(fold.train["Date_dt"])
            va_dates = set(fold.validation["Date_dt"])
            self.assertEqual(len(tr_dates.intersection(va_dates)), 0)

            # 3. All equities for each date are intact (multiples of 20)
            self.assertEqual(len(fold.train) % 20, 0)
            self.assertEqual(len(fold.validation) % 20, 0)

            # 4. Strict Chronology
            self.assertLess(fold.train["Date_dt"].max(), fold.validation["Date_dt"].min())

            # 5. Purged Target-Boundary Separation: max train target < min val date
            max_tr_tgt = fold.train["Target_Date"].max()
            min_va_date = fold.validation["Date_dt"].min()
            self.assertLess(
                max_tr_tgt,
                min_va_date,
                f"Fold {fold.fold} target overlap: {max_tr_tgt} >= {min_va_date}",
            )

            # 6. Fold-local thresholds: q_low and q_high are calibrated strictly on this fold's train data
            expected_q_low = float(fold.train["Next_Day_Return"].quantile(1 / 3))
            expected_q_high = float(fold.train["Next_Day_Return"].quantile(2 / 3))
            self.assertAlmostEqual(fold.q_low, expected_q_low, places=6)
            self.assertAlmostEqual(fold.q_high, expected_q_high, places=6)

            # 7. Check that validation regimes were assigned using the fold-local thresholds
            val_low = fold.validation.loc[fold.validation["_fold_regime"] == "Low", "Next_Day_Return"]
            val_high = fold.validation.loc[fold.validation["_fold_regime"] == "High", "Next_Day_Return"]
            if not val_low.empty:
                self.assertLess(val_low.max(), fold.q_low)
            if not val_high.empty:
                self.assertGreaterEqual(val_high.min(), fold.q_high)

            total_val_dates += fold.validation["Date_dt"].nunique()

        # In standard TimeSeriesSplit(n_splits=5) on 587 dates:
        # val_size = 587 // 6 = 97 dates per fold.
        # Total validation dates = 5 * 97 = 485 dates.
        # Initial training buffer = 587 - 485 = 102 dates.
        self.assertEqual(total_val_dates, 485)

    def test_04_classification_threshold_leakage_free(self):
        """Verify that outer holdout classification tertile boundaries q1 and q2 are computed strictly on training data."""
        expected_q1 = float(self.train_df["Next_Day_Return"].quantile(1 / 3))
        expected_q2 = float(self.train_df["Next_Day_Return"].quantile(2 / 3))

        self.assertAlmostEqual(self.q1, expected_q1, places=6)
        self.assertAlmostEqual(self.q2, expected_q2, places=6)

        test_low_max = self.test_df.loc[self.test_df["Return_Category"] == "Low", "Next_Day_Return"].max()
        test_high_min = self.test_df.loc[self.test_df["Return_Category"] == "High", "Next_Day_Return"].min()

        self.assertLess(test_low_max, expected_q1)
        self.assertGreaterEqual(test_high_min, expected_q2)

    def test_05_reconciliation_authoritative_regression_metrics(self):
        """Verify that regression models match authoritative production metrics."""
        mb = train_predictive_models(self.train_df, self.test_df)

        # Baseline: MAE = 1.1985%, RMSE = 1.6247%
        self.assertAlmostEqual(mb["base_mae"], 1.1985, places=3)
        self.assertAlmostEqual(mb["base_rmse"], 1.6247, places=3)
        self.assertAlmostEqual(mb["base_r2"], -0.001629, places=5)

        # SLR: MAE = 1.2005%, RMSE = 1.6264%, R2 = -0.003635
        self.assertAlmostEqual(mb["slr_mae"], 1.2005, places=3)
        self.assertAlmostEqual(mb["slr_rmse"], 1.6264, places=3)
        self.assertAlmostEqual(mb["slr_r2"], -0.003635, places=5)

        # MLR: MAE = 1.2004%, RMSE = 1.6272%, R2 = -0.004658
        self.assertAlmostEqual(mb["mlr_mae"], 1.2004, places=3)
        self.assertAlmostEqual(mb["mlr_rmse"], 1.6272, places=3)
        self.assertAlmostEqual(mb["mlr_r2"], -0.004658, places=5)

        # Every regression model has R2 <= 0 against out-of-sample holdout
        self.assertLessEqual(mb["slr_r2"], 0.0)
        self.assertLessEqual(mb["mlr_r2"], 0.0)
        self.assertLessEqual(mb["ridge_r2"], 0.0)

    def test_06_reconciliation_authoritative_classification_metrics(self):
        """Verify that classification models match authoritative production metrics including Balanced Acc and Macro F1."""
        mb = train_predictive_models(self.train_df, self.test_df)
        clf_df = mb["clf_metrics"]

        # Tuned DT: Accuracy = 37.64%, Balanced Acc = 38.29%, Macro F1 = 0.3742, Weighted F1 = 0.3735
        dt_tuned_row = clf_df[clf_df["Algorithm"] == "Tuned Decision Tree (Depth 4, Leaf 30)"].iloc[0]
        self.assertAlmostEqual(dt_tuned_row["Accuracy (%)"], 37.64, places=1)
        self.assertAlmostEqual(dt_tuned_row["Balanced Acc (%)"], 38.29, places=1)
        self.assertAlmostEqual(dt_tuned_row["Macro F1"], 0.3742, places=3)
        self.assertAlmostEqual(dt_tuned_row["Weighted F1"], 0.3735, places=3)

        # Baseline DT: Accuracy = 36.37%, Balanced Acc = 37.56%, Macro F1 = 0.3596, Weighted F1 = 0.3585
        dt_base_row = clf_df[clf_df["Algorithm"] == "Entropy Decision Tree (Baseline, Depth 5)"].iloc[0]
        self.assertAlmostEqual(dt_base_row["Accuracy (%)"], 36.37, places=1)
        self.assertAlmostEqual(dt_base_row["Balanced Acc (%)"], 37.56, places=1)
        self.assertAlmostEqual(dt_base_row["Macro F1"], 0.3596, places=3)
        self.assertAlmostEqual(dt_base_row["Weighted F1"], 0.3585, places=3)

        # Gaussian Naive Bayes: Accuracy = 31.88%, Balanced Acc = 34.99%, Macro F1 = 0.2938
        nb_row = clf_df[clf_df["Algorithm"] == "Gaussian Naïve Bayes (Baseline)"].iloc[0]
        self.assertAlmostEqual(nb_row["Accuracy (%)"], 31.88, places=1)
        self.assertAlmostEqual(nb_row["Balanced Acc (%)"], 34.99, places=1)
        self.assertAlmostEqual(nb_row["Macro F1"], 0.2938, places=3)

        # Logistic Regression: Accuracy = 36.16%, Balanced Acc = 37.68%, Macro F1 = 0.3610
        lr_row = clf_df[clf_df["Algorithm"] == "L2 Regularized Logistic Regression"].iloc[0]
        self.assertAlmostEqual(lr_row["Accuracy (%)"], 36.16, places=1)
        self.assertAlmostEqual(lr_row["Balanced Acc (%)"], 37.68, places=1)
        self.assertAlmostEqual(lr_row["Macro F1"], 0.3610, places=3)

    def test_07_paired_moving_block_bootstrap_sensitivity(self):
        """Verify that paired_moving_block_bootstrap correctly evaluates blocks 3, 5, 10 and all CIs span zero."""
        clf_cols = ["Lag1_Return", "Lag1_Volatility", "Lag1_Volume_Log", "Month"]
        dt_base = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42).fit(
            self.train_df[clf_cols], self.train_df["Return_Category"]
        )
        dt_tuned = DecisionTreeClassifier(criterion="entropy", max_depth=4, min_samples_leaf=30, random_state=42).fit(
            self.train_df[clf_cols], self.train_df["Return_Category"]
        )

        pred_base = dt_base.predict(self.test_df[clf_cols])
        pred_tuned = dt_tuned.predict(self.test_df[clf_cols])
        y_true = self.test_df["Return_Category"].to_numpy()
        dates = self.test_df["Date_dt"]

        boot_df = paired_moving_block_bootstrap(
            y_true=y_true,
            y_pred_reference=pred_base,
            y_pred_candidate=pred_tuned,
            dates=dates,
            block_lengths=(3, 5, 10),
            n_resamples=500,  # quick test check
            random_state=42,
        )

        self.assertEqual(len(boot_df), 3)
        self.assertListEqual(boot_df["Block Length (trading days)"].tolist(), [3, 5, 10])

        for _, row in boot_df.iterrows():
            self.assertAlmostEqual(row["Accuracy Difference (pp)"], 1.2671, places=2)
            # Crucial: 95% CI lower <= 0 and upper >= 0 for all block lengths
            self.assertLessEqual(row["95% CI Lower (pp)"], 0.0)
            self.assertGreaterEqual(row["95% CI Upper (pp)"], 0.0)
            # p-value > 0.05
            self.assertGreater(row["Two-Sided Bootstrap p-value"], 0.05)


if __name__ == "__main__":
    unittest.main()
