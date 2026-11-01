"""Reproduce QuantPulse's classification holdout metrics and temporal inference.

Read-only with respect to the source CSV. All outputs go to --output-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from predictive_inference import paired_moving_block_bootstrap
from predictive_validation import evaluate_decision_tree_grid, iter_purged_date_folds


FEATURES = ["Lag1_Return", "Lag1_Volatility", "Lag1_Volume_Log", "Month"]
LABELS = ["Low", "Medium", "High"]


def prepare_data(csv_path: Path):
    source = pd.read_csv(csv_path)
    source["Date_dt"] = pd.to_datetime(source["Date"], errors="raise")
    source = source.sort_values(["Ticker", "Date_dt"]).reset_index(drop=True)
    groups = []
    for _, group0 in source.groupby("Ticker", sort=False):
        group = group0.copy().reset_index(drop=True)
        group["Next_Day_Return"] = (
            (group["Close_Price"].shift(-1) - group["Close_Price"])
            / group["Close_Price"] * 100
        )
        group["Target_Date"] = group["Date_dt"].shift(-1)
        group["Lag1_Return"] = (
            (group["Close_Price"] - group["Close_Price"].shift(1))
            / group["Close_Price"].shift(1) * 100
        )
        group["Lag1_Intraday"] = (
            (group["Close_Price"] - group["Open_Price"]) / group["Open_Price"] * 100
        )
        group["Lag1_Volatility"] = (
            (group["High_Price"] - group["Low_Price"]) / group["Open_Price"] * 100
        )
        group["Lag1_Volume_Log"] = np.log1p(group["Volume"])
        group["SMA5_Return"] = group["Lag1_Return"].rolling(window=5).mean()
        group["Month"] = group["Date_dt"].dt.month
        groups.append(group)

    supervised = pd.concat(groups, ignore_index=True)
    model_features = [
        "Lag1_Return", "Lag1_Intraday", "Lag1_Volatility",
        "Lag1_Volume_Log", "SMA5_Return", "Month",
    ]
    supervised = supervised.dropna(
        subset=["Next_Day_Return", "Target_Date"] + model_features
    ).reset_index(drop=True)

    unique_dates = np.array(sorted(supervised["Date_dt"].unique()))
    split_idx = int(len(unique_dates) * 0.8)
    outer_train_dates = set(unique_dates[:split_idx])
    train = supervised[supervised["Date_dt"].isin(outer_train_dates)].copy().reset_index(drop=True)
    max_train_target_date = train["Target_Date"].max()
    test = supervised[supervised["Date_dt"] > max_train_target_date].copy().reset_index(drop=True)

    q_low = float(train["Next_Day_Return"].quantile(1 / 3))
    q_high = float(train["Next_Day_Return"].quantile(2 / 3))
    train["Return_Category"] = np.where(
        train["Next_Day_Return"] < q_low, "Low",
        np.where(train["Next_Day_Return"] < q_high, "Medium", "High"),
    )
    test["Return_Category"] = np.where(
        test["Next_Day_Return"] < q_low, "Low",
        np.where(test["Next_Day_Return"] < q_high, "Medium", "High"),
    )
    return source, supervised, train, test, q_low, q_high


def classification_row(name, y_true, y_pred, model):
    return {
        "Model": name,
        "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
        "Balanced Accuracy (%)": balanced_accuracy_score(y_true, y_pred) * 100,
        "Macro F1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "Weighted Precision": precision_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "Weighted Recall": recall_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0),
        "Parameters": str(model),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce QuantPulse classification and validation results")
    parser.add_argument("csv", nargs="?", default="stock_market_data.csv")
    parser.add_argument("--output-dir", default="quantpulse_validation_output")
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        parser.error(f"CSV not found: {csv_path}")

    source, supervised, train, test, q_low, q_high = prepare_data(csv_path)
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["Return_Category"], test["Return_Category"]

    models = {
        "Majority Baseline": None,
        "Gaussian Naive Bayes": GaussianNB(),
        "Logistic Regression (scaled, C=0.1)": make_pipeline(
            StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, random_state=42)
        ),
        "Decision Tree (depth=5)": DecisionTreeClassifier(
            criterion="entropy", max_depth=5, random_state=42
        ),
        "Random Forest (100 trees, depth=4, leaf=30)": RandomForestClassifier(
            n_estimators=100, max_depth=4, min_samples_leaf=30,
            random_state=42, n_jobs=-1,
        ),
        "Tuned Decision Tree (depth=4, leaf=30)": DecisionTreeClassifier(
            criterion="entropy", max_depth=4, min_samples_leaf=30, random_state=42
        ),
    }

    majority = y_train.value_counts().idxmax()
    predictions: dict[str, np.ndarray] = {"Majority Baseline": np.repeat(majority, len(y_test))}
    rows: list[dict] = []
    rows.append(classification_row("Majority Baseline", y_test, predictions["Majority Baseline"], f"predict={majority}"))
    fitted = {}
    for name, model in list(models.items())[1:]:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        fitted[name] = model
        rows.append(classification_row(name, y_test, pred, model.get_params() if hasattr(model, "get_params") else str(model)))

    metrics = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    ref_name = "Decision Tree (depth=5)"
    candidate_name = "Tuned Decision Tree (depth=4, leaf=30)"
    bootstrap = paired_moving_block_bootstrap(
        y_true=y_test.to_numpy(),
        y_pred_reference=predictions[ref_name],
        y_pred_candidate=predictions[candidate_name],
        dates=test["Date_dt"],
        block_lengths=(3, 5, 10),
        n_resamples=args.bootstrap_replicates,
        random_state=42,
    )

    cv_grid = evaluate_decision_tree_grid(train, n_splits=5, purge_sessions=1)
    cv_grid.to_csv(Path(args.output_dir) / "purged_cv_grid.csv", index=False) if Path(args.output_dir).exists() else None

    folds_out = []
    for fold in iter_purged_date_folds(train, n_splits=5, purge_sessions=1):
        folds_out.append({
            "Fold": fold.fold,
            "Train Dates": fold.train["Date_dt"].nunique(),
            "Train Rows": len(fold.train),
            "Purged Dates": ", ".join(d.strftime("%Y-%m-%d") for d in fold.purged_dates),
            "Max Train Target Date": fold.train["Target_Date"].max().strftime("%Y-%m-%d"),
            "Validation Dates": fold.validation["Date_dt"].nunique(),
            "Validation Rows": len(fold.validation),
            "Min Validation Date": fold.validation["Date_dt"].min().strftime("%Y-%m-%d"),
            "Fold q1": fold.q_low,
            "Fold q2": fold.q_high,
        })
    folds_df = pd.DataFrame(folds_out)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(outdir / "classification_holdout_metrics.csv", index=False)
    bootstrap.to_csv(outdir / "moving_block_bootstrap.csv", index=False)
    cv_grid.to_csv(outdir / "purged_cv_grid.csv", index=False)
    folds_df.to_csv(outdir / "purged_cv_folds.csv", index=False)

    holdout_predictions = pd.DataFrame({
        "Date": test["Date_dt"].dt.strftime("%Y-%m-%d"),
        "Ticker": test["Ticker"],
        "Actual": y_test,
        "Predicted_Baseline_DT": predictions[ref_name],
        "Predicted_Tuned_DT": predictions[candidate_name],
    })
    holdout_predictions.to_csv(outdir / "paired_classifier_holdout_predictions.csv", index=False)

    cm = confusion_matrix(y_test, predictions[candidate_name], labels=LABELS)
    report = {
        "source_csv": str(csv_path.resolve()),
        "outer_train_rows": len(train),
        "outer_train_dates": int(train["Date_dt"].nunique()),
        "test_rows": len(test),
        "test_dates": int(test["Date_dt"].nunique()),
        "train_date_min": train["Date_dt"].min().strftime("%Y-%m-%d"),
        "train_date_max": train["Date_dt"].max().strftime("%Y-%m-%d"),
        "max_train_target_date": train["Target_Date"].max().strftime("%Y-%m-%d"),
        "test_feature_date_min": test["Date_dt"].min().strftime("%Y-%m-%d"),
        "test_feature_date_max": test["Date_dt"].max().strftime("%Y-%m-%d"),
        "outer_train_thresholds": {"q1": q_low, "q2": q_high},
        "class_metrics": metrics.to_dict(orient="records"),
        "tuned_tree_confusion_matrix_labels_low_medium_high": cm.tolist(),
        "moving_block_bootstrap": bootstrap.to_dict(orient="records"),
        "purged_cv_top_10": cv_grid.head(10).to_dict(orient="records"),
        "cv_fold_details": folds_df.to_dict(orient="records"),
        "evaluation_warning": (
            "The historical holdout metrics have already been inspected in prior model experiments; "
            "treat them as exploratory retrospective comparisons, not a pristine confirmatory test."
        ),
    }
    (outdir / "quantpulse_validation_report.json").write_text(json.dumps(report, indent=2, default=str))

    print("CLASSIFICATION HOLDOUT METRICS")
    print(metrics.drop(columns=["Parameters"]).round(4).to_string(index=False))
    print("\nDATE-BLOCK BOOTSTRAP SENSITIVITY")
    print(bootstrap.round(4).to_string(index=False))
    print("\nPURGED CV TOP 10 (fold-local thresholds)")
    print(cv_grid.head(10).round(4).to_string(index=False))
    print(f"\nOutputs saved to: {outdir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
