from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from quantpulse_quality import audit_stock_data

HERE = Path(__file__).resolve().parent
_candidates = [HERE / "stock_market_data.csv", HERE.parent / "stock_market_data.csv"]
DATA_PATH = next((p for p in _candidates if p.is_file()), _candidates[0])


class QuantPulseQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DATA_PATH.is_file():
            raise FileNotFoundError(f"Expected dataset at {DATA_PATH}")
        cls.data = pd.read_csv(DATA_PATH)

    def test_authoritative_csv_has_no_hard_quality_failures(self):
        result = audit_stock_data(self.data, DATA_PATH)
        self.assertEqual(result.failed_count, 0, result.checks.to_string(index=False))

    def test_dataset_scope_is_reconciled(self):
        result = audit_stock_data(self.data, DATA_PATH)
        self.assertEqual(result.summary["rows"], 14800)
        self.assertEqual(result.summary["dates"], 740)
        self.assertEqual(result.summary["tickers"], 20)
        self.assertEqual(result.summary["sectors"], 5)
        self.assertEqual(result.summary["date_min"], "2023-10-12")
        self.assertEqual(result.summary["date_max"], "2026-10-12")

    def test_intraday_return_definition_matches_all_rows(self):
        result = audit_stock_data(self.data)
        row = result.checks.loc[result.checks["Check"] == "Daily_Return definition"].iloc[0]
        self.assertEqual(row["Status"], "PASS")
        self.assertIn("100.00%", row["Observed"])

    def test_duplicate_fact_grain_is_detected(self):
        damaged = pd.concat([self.data, self.data.iloc[[0]]], ignore_index=True)
        result = audit_stock_data(damaged)
        row = result.checks.loc[result.checks["Check"] == "Date-ticker uniqueness"].iloc[0]
        self.assertEqual(row["Status"], "FAIL")
        self.assertEqual(int(row["Observed"]), 1)

    def test_invalid_ohlc_is_detected(self):
        damaged = self.data.copy()
        damaged.loc[0, "High_Price"] = damaged.loc[0, "Low_Price"] - 1
        result = audit_stock_data(damaged)
        row = result.checks.loc[result.checks["Check"] == "OHLC envelope"].iloc[0]
        self.assertEqual(row["Status"], "FAIL")

    def test_sector_dimension_conflict_is_detected(self):
        damaged = self.data.copy()
        first_ticker = damaged.loc[0, "Ticker"]
        idx = damaged.index[damaged["Ticker"] == first_ticker][0]
        damaged.loc[idx, "Sector"] = "Conflicting Sector"
        result = audit_stock_data(damaged)
        row = result.checks.loc[result.checks["Check"] == "Ticker-to-sector consistency"].iloc[0]
        self.assertEqual(row["Status"], "FAIL")

    def test_missing_provenance_is_a_warning_not_a_data_authenticity_claim(self):
        result = audit_stock_data(self.data)
        row = result.checks.loc[result.checks["Check"] == "Source provenance metadata"].iloc[0]
        self.assertEqual(row["Status"], "WARN")
        self.assertIn("does not establish", row["Details"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
