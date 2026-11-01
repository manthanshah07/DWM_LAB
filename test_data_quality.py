"""
Unit tests for QuantPulse data quality checks, OHLC envelope,
business key uniqueness, panel completeness, and provenance verification.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path

from quantpulse_quality import audit_stock_data, _file_sha256


class TestDataQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_path = Path("stock_market_data.csv")
        cls.df = pd.read_csv(cls.source_path)

    def test_01_authoritative_dataset_passes_quality_audit(self):
        """Verify that authoritative stock_market_data.csv has zero hard failures."""
        res = audit_stock_data(self.df, source_path=self.source_path)
        self.assertTrue(res.passed)
        self.assertEqual(res.failed_count, 0)
        # Warning is expected for missing provenance metadata in raw CSV
        self.assertEqual(res.warning_count, 1)

    def test_02_detects_duplicate_fact_grain(self):
        """Verify detection of duplicate fact grain (Date + Ticker uniqueness)."""
        corrupted = self.df.copy()
        # Duplicate the first row
        corrupted = pd.concat([corrupted, corrupted.iloc[[0]]], ignore_index=True)
        res = audit_stock_data(corrupted)
        self.assertFalse(res.passed)
        self.assertGreater(res.failed_count, 0)

        dup_check = res.checks[res.checks["Check"] == "Date-ticker uniqueness"].iloc[0]
        self.assertEqual(dup_check["Status"], "FAIL")
        self.assertEqual(dup_check["Severity"], "Critical")

    def test_03_detects_malformed_dates(self):
        """Verify detection of unparseable/malformed dates."""
        corrupted = self.df.copy()
        corrupted.loc[0, "Date"] = "not-a-valid-date-2024"
        res = audit_stock_data(corrupted)
        self.assertFalse(res.passed)

        date_check = res.checks[res.checks["Check"] == "Date parseability"].iloc[0]
        self.assertEqual(date_check["Status"], "FAIL")

    def test_04_detects_invalid_ohlc_bars(self):
        """Verify detection of invalid OHLC price envelope (e.g. High < Open or Low > Close)."""
        corrupted = self.df.copy()
        # Violate High envelope: High < Open
        corrupted.loc[0, "High_Price"] = corrupted.loc[0, "Open_Price"] - 10.0
        res = audit_stock_data(corrupted)
        self.assertFalse(res.passed)

        ohlc_check = res.checks[res.checks["Check"] == "OHLC envelope"].iloc[0]
        self.assertEqual(ohlc_check["Status"], "FAIL")

    def test_05_detects_negative_prices(self):
        """Verify detection of non-positive OHLC prices."""
        corrupted = self.df.copy()
        corrupted.loc[0, "Low_Price"] = -5.0
        res = audit_stock_data(corrupted)
        self.assertFalse(res.passed)

        price_check = res.checks[res.checks["Check"] == "Positive OHLC prices"].iloc[0]
        self.assertEqual(price_check["Status"], "FAIL")

    def test_06_detects_ticker_sector_conflicts(self):
        """Verify detection when a single ticker maps to multiple ambiguous sectors."""
        corrupted = self.df.copy()
        first_ticker = corrupted.loc[0, "Ticker"]
        corrupted.loc[0, "Sector"] = "Fabricated_Alternative_Sector"
        res = audit_stock_data(corrupted)
        self.assertFalse(res.passed)

        sector_check = res.checks[res.checks["Check"] == "Ticker-to-sector consistency"].iloc[0]
        self.assertEqual(sector_check["Status"], "FAIL")

    def test_07_detects_incomplete_date_panels(self):
        """Verify detection when a date panel has missing tickers."""
        # Drop one stock observation from the first date
        first_date = self.df.loc[0, "Date"]
        idx_to_drop = self.df[self.df["Date"] == first_date].index[0]
        corrupted = self.df.drop(index=idx_to_drop).reset_index(drop=True)

        res = audit_stock_data(corrupted)
        panel_check = res.checks[res.checks["Check"] == "Complete date panel"].iloc[0]
        self.assertEqual(panel_check["Status"], "WARN")
        self.assertEqual(panel_check["Severity"], "Medium")

    def test_08_detects_daily_return_formula_inconsistency(self):
        """Verify detection of Daily_Return calculation deviations."""
        corrupted = self.df.copy()
        # Corrupt Daily_Return for first row
        corrupted.loc[0, "Daily_Return"] = 99.99
        res = audit_stock_data(corrupted)

        ret_check = res.checks[res.checks["Check"] == "Daily_Return definition"].iloc[0]
        self.assertEqual(ret_check["Status"], "WARN")
        self.assertEqual(ret_check["Severity"], "Medium")

    def test_09_detects_missing_provenance_metadata(self):
        """Verify that absence of source provenance metadata is flagged as a WARN."""
        res = audit_stock_data(self.df)
        prov_check = res.checks[res.checks["Check"] == "Source provenance metadata"].iloc[0]
        self.assertEqual(prov_check["Status"], "WARN")
        self.assertEqual(prov_check["Severity"], "Medium")

    def test_10_file_sha256_fingerprint_verified(self):
        """Verify SHA-256 fingerprint matches source file bytes."""
        expected_sha = "914753232eb5e677716d7a65a4b1cd2ec48826380dbcfa5ba585a9832b93263b"
        computed_sha = _file_sha256(self.source_path)
        self.assertEqual(computed_sha, expected_sha)


if __name__ == "__main__":
    unittest.main()
