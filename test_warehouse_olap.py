"""
QuantPulse — Data Warehouse & OLAP Test Suite
=============================================
Comprehensive unit and integration test suite verifying:
1. Dimensional database schema integrity and foreign keys.
2. ACID transactional ETL, byte-snapshot hashing, and safe recovery.
3. Multidimensional SQL OLAP operations (Roll-up, Drill-down, Slice, Dice, Pivot).
4. Strict numerical equivalence between SQLite and Pandas reference calculations.
5. Single-process reader/rebuild concurrency and thread-lock safety.
6. Streamlit dashboard end-to-end regression validation via AppTest.
"""

from __future__ import annotations

import io
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import warehouse


class TestDatabaseIntegrity(unittest.TestCase):
    """Verifies schema structure, constraints, indexes, and expected row counts."""

    @classmethod
    def setUpClass(cls):
        warehouse.init_warehouse()
        cls.db_path = warehouse.DEFAULT_DB_PATH

    def test_required_tables_exist(self):
        with warehouse.get_readonly_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
            )
            tables = {row[0] for row in cur.fetchall()}

        expected_tables = {
            "Dim_Sector",
            "Dim_Exchange",
            "Dim_Stock",
            "Dim_Date",
            "Fact_DailyTrades",
            "_etl_metadata",
        }
        self.assertTrue(
            expected_tables.issubset(tables),
            f"Missing tables: {expected_tables - tables}",
        )

    def test_pragma_integrity_and_foreign_keys(self):
        with warehouse.get_readonly_connection(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            self.assertEqual(cur.fetchone()[0], "ok")

            cur.execute("PRAGMA foreign_key_check;")
            violations = cur.fetchall()
            self.assertEqual(
                len(violations), 0, f"Foreign key violations found: {violations}"
            )

    def test_expected_baseline_dimensions_and_facts(self):
        stats = warehouse.get_warehouse_stats(self.db_path)
        counts = stats["table_counts"]

        self.assertEqual(counts["Fact_DailyTrades"], 14800)
        self.assertEqual(counts["Dim_Date"], 740)
        self.assertEqual(counts["Dim_Stock"], 20)
        self.assertEqual(counts["Dim_Sector"], 5)
        self.assertEqual(counts["Dim_Exchange"], 1)

    def test_unique_composite_constraint_on_fact(self):
        # Verify that inserting a duplicate (Date_Key, Stock_Key) raises an IntegrityError
        with self.assertRaises(sqlite3.IntegrityError):
            conn = sqlite3.connect(str(self.db_path))
            try:
                conn.execute("PRAGMA foreign_keys = ON;")
                # Try inserting a duplicate of the first record
                conn.execute(
                    """
                    INSERT INTO Fact_DailyTrades (
                        Date_Key, Stock_Key, Open_Price, High_Price, Low_Price,
                        Close_Price, Volume, Daily_Return, Intraday_Volatility
                    ) VALUES (20231012, 1, 100.0, 105.0, 95.0, 102.0, 1000, 2.0, 10.0);
                    """
                )
            finally:
                conn.close()


class TestETLAndFreshness(unittest.TestCase):
    """Verifies transactional ETL, freshness detection, corruption recovery, and safe rebuilds."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_csv = Path(self.temp_dir.name) / "test_market_data.csv"
        self.test_db = Path(self.temp_dir.name) / "test_warehouse.db"

        # Copy original CSV into test directory
        raw_bytes = warehouse.DEFAULT_CSV_PATH.read_bytes()
        self.test_csv.write_bytes(raw_bytes)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_initial_creation_and_idempotency(self):
        # 1. First build
        res1 = warehouse.init_warehouse(self.test_csv, self.test_db)
        self.assertEqual(res1["status"], "rebuilt")
        self.assertTrue(res1["rebuilt"])
        self.assertEqual(res1["rows_ingested"], 14800)

        # 2. Repeated build on unchanged dataset (fast path)
        res2 = warehouse.init_warehouse(self.test_csv, self.test_db)
        self.assertEqual(res2["status"], "fresh")
        self.assertFalse(res2["rebuilt"])

        # Row count remains exactly 14,800
        stats = warehouse.get_warehouse_stats(self.test_db)
        self.assertEqual(stats["table_counts"]["Fact_DailyTrades"], 14800)

    def test_source_fingerprint_change_triggers_rebuild(self):
        warehouse.init_warehouse(self.test_csv, self.test_db)

        # Modify source CSV (drop 1 row)
        df = pd.read_csv(self.test_csv)
        modified_df = df.iloc[:-1]
        modified_df.to_csv(self.test_csv, index=False)

        # Ingestion must detect modified SHA-256 and rebuild
        res = warehouse.init_warehouse(self.test_csv, self.test_db)
        self.assertEqual(res["status"], "rebuilt")
        self.assertTrue(res["rebuilt"])
        self.assertEqual(res["rows_ingested"], 14799)

        stats = warehouse.get_warehouse_stats(self.test_db)
        self.assertEqual(stats["table_counts"]["Fact_DailyTrades"], 14799)

    def test_schema_version_mismatch_triggers_rebuild(self):
        warehouse.init_warehouse(self.test_csv, self.test_db)

        # Tamper with stored schema version in _etl_metadata
        conn = sqlite3.connect(str(self.test_db))
        conn.execute("UPDATE _etl_metadata SET value = '1.0.0' WHERE key = 'schema_version';")
        conn.commit()
        conn.close()

        res = warehouse.init_warehouse(self.test_csv, self.test_db)
        self.assertEqual(res["status"], "rebuilt")
        self.assertTrue(res["rebuilt"])

        stats = warehouse.get_warehouse_stats(self.test_db)
        self.assertEqual(stats["metadata"]["schema_version"], warehouse.SCHEMA_VERSION)

    def test_corrupt_database_safe_recovery(self):
        warehouse.init_warehouse(self.test_csv, self.test_db)

        # Corrupt the physical database header
        with open(self.test_db, "r+b") as f:
            f.seek(0)
            f.write(b"CORRUPTED_HEADER_BYTES_FOR_TESTING_1234567890")

        # Ingestion must recover safely by rebuilding a fresh database
        res = warehouse.init_warehouse(self.test_csv, self.test_db)
        self.assertEqual(res["status"], "rebuilt")

        stats = warehouse.get_warehouse_stats(self.test_db)
        self.assertEqual(stats["table_counts"]["Fact_DailyTrades"], 14800)

    def test_failed_rebuild_preserves_valid_production_database(self):
        # 1. Build valid database
        warehouse.init_warehouse(self.test_csv, self.test_db)
        orig_stats = warehouse.get_warehouse_stats(self.test_db)
        self.assertEqual(orig_stats["table_counts"]["Fact_DailyTrades"], 14800)

        # 2. Corrupt source CSV by writing invalid headers
        self.test_csv.write_text("Invalid,Headers,Only\n1,2,3\n")

        # 3. Attempt rebuild (should fail with ValueError and NOT destroy production DB)
        with self.assertRaises(ValueError):
            warehouse.init_warehouse(self.test_csv, self.test_db, force_rebuild=True)

        # 4. Production database must remain completely intact
        pres_stats = warehouse.get_warehouse_stats(self.test_db)
        self.assertEqual(pres_stats["table_counts"]["Fact_DailyTrades"], 14800)


class TestOLAPOperationsAndNumericalEquivalence(unittest.TestCase):
    """Compares SQL aggregation results against independent Pandas computations."""

    @classmethod
    def setUpClass(cls):
        warehouse.init_warehouse()
        cls.db_path = warehouse.DEFAULT_DB_PATH

        # Load reference source data and compute calendar columns
        df = pd.read_csv(warehouse.DEFAULT_CSV_PATH)
        df["Date_dt"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date_dt"].dt.year
        df["Quarter"] = "Q" + df["Date_dt"].dt.quarter.astype(str)
        df["Year_Quarter"] = df["Year"].astype(str) + "-" + df["Quarter"]
        df["YearMonth"] = df["Date_dt"].dt.to_period("M").astype(str)
        cls.df_ref = df

    def test_kpi_summary_equivalence(self):
        kpi_sql = warehouse.query_kpi_summary(self.db_path)

        self.assertEqual(kpi_sql["total_records"], len(self.df_ref))
        self.assertEqual(kpi_sql["active_symbols"], self.df_ref["Ticker"].nunique())
        self.assertEqual(kpi_sql["active_sectors"], self.df_ref["Sector"].nunique())
        self.assertEqual(kpi_sql["aggregate_volume"], int(self.df_ref["Volume"].sum()))
        self.assertAlmostEqual(
            kpi_sql["avg_close_price"], self.df_ref["Close_Price"].mean(), places=5
        )
        self.assertAlmostEqual(
            kpi_sql["mean_daily_return"], self.df_ref["Daily_Return"].mean(), places=5
        )

    def test_quarterly_rollup_equivalence(self):
        sql_df = warehouse.query_sector_trend(self.db_path, time_grain="Quarterly")
        pd_df = (
            self.df_ref.groupby(["Year_Quarter", "Sector"], as_index=False)
            .agg(
                Total_Volume=("Volume", "sum"),
                Avg_Close_Price=("Close_Price", "mean"),
                Avg_Daily_Return=("Daily_Return", "mean"),
            )
            .sort_values(["Year_Quarter", "Sector"])
            .reset_index(drop=True)
        )

        self.assertEqual(len(sql_df), len(pd_df))
        np.testing.assert_array_equal(
            sql_df["Total_Volume"].values, pd_df["Total_Volume"].values
        )
        np.testing.assert_allclose(
            sql_df["Avg_Close_Price"].values, pd_df["Avg_Close_Price"].values, rtol=1e-5
        )
        np.testing.assert_allclose(
            sql_df["Avg_Daily_Return"].values, pd_df["Avg_Daily_Return"].values, rtol=1e-5
        )

    def test_yearly_rollup_equivalence(self):
        sql_df = warehouse.query_sector_trend(self.db_path, time_grain="Yearly")
        pd_df = (
            self.df_ref.groupby(["Year", "Sector"], as_index=False)
            .agg(
                Total_Volume=("Volume", "sum"),
                Avg_Close_Price=("Close_Price", "mean"),
                Avg_Daily_Return=("Daily_Return", "mean"),
            )
            .sort_values(["Year", "Sector"])
            .reset_index(drop=True)
        )

        self.assertEqual(len(sql_df), len(pd_df))
        np.testing.assert_array_equal(
            sql_df["Total_Volume"].values, pd_df["Total_Volume"].values
        )
        np.testing.assert_allclose(
            sql_df["Avg_Close_Price"].values, pd_df["Avg_Close_Price"].values, rtol=1e-5
        )
        np.testing.assert_allclose(
            sql_df["Avg_Daily_Return"].values, pd_df["Avg_Daily_Return"].values, rtol=1e-5
        )

    def test_monthly_rollup_equivalence(self):
        sql_df = warehouse.query_sector_trend(self.db_path, time_grain="Monthly")
        pd_df = (
            self.df_ref.groupby(["YearMonth", "Sector"], as_index=False)
            .agg(
                Total_Volume=("Volume", "sum"),
                Avg_Close_Price=("Close_Price", "mean"),
                Avg_Daily_Return=("Daily_Return", "mean"),
            )
            .sort_values(["YearMonth", "Sector"])
            .reset_index(drop=True)
        )

        self.assertEqual(len(sql_df), len(pd_df))
        np.testing.assert_array_equal(
            sql_df["Total_Volume"].values, pd_df["Total_Volume"].values
        )
        np.testing.assert_allclose(
            sql_df["Avg_Close_Price"].values, pd_df["Avg_Close_Price"].values, rtol=1e-5
        )
        np.testing.assert_allclose(
            sql_df["Avg_Daily_Return"].values, pd_df["Avg_Daily_Return"].values, rtol=1e-5
        )

    def test_daily_trend_and_output_mapping(self):
        # 1. Populated selection: 740 dates * 5 sectors = 3,700 rows
        sql_df = warehouse.query_sector_trend(self.db_path, time_grain="Daily")
        self.assertEqual(len(sql_df), 3700)
        self.assertIn("Date_dt", sql_df.columns)
        self.assertEqual(
            list(sql_df.columns),
            ["Date_dt", "Sector", "Total_Volume", "Avg_Close_Price", "Avg_Daily_Return"],
        )

        # 2. Empty sector selection: must return empty DataFrame with Date_dt column
        empty_df = warehouse.query_sector_trend(
            self.db_path, dice_sectors=[], time_grain="Daily"
        )
        self.assertEqual(len(empty_df), 0)
        self.assertEqual(
            list(empty_df.columns),
            ["Date_dt", "Sector", "Total_Volume", "Avg_Close_Price", "Avg_Daily_Return"],
        )

    def test_slice_by_year_and_dice_by_sector(self):
        # Slice Year = 2024, Dice Sectors = ['Banking', 'IT']
        sql_df = warehouse.query_sector_trend(
            self.db_path,
            slice_year="2024",
            dice_sectors=["Banking", "IT"],
            time_grain="Quarterly",
        )
        pd_sub = self.df_ref[
            (self.df_ref["Year"] == 2024) & (self.df_ref["Sector"].isin(["Banking", "IT"]))
        ]
        pd_df = (
            pd_sub.groupby(["Year_Quarter", "Sector"], as_index=False)
            .agg(
                Total_Volume=("Volume", "sum"),
                Avg_Close_Price=("Close_Price", "mean"),
                Avg_Daily_Return=("Daily_Return", "mean"),
            )
            .sort_values(["Year_Quarter", "Sector"])
            .reset_index(drop=True)
        )

        self.assertEqual(len(sql_df), len(pd_df))
        np.testing.assert_array_equal(
            sql_df["Total_Volume"].values, pd_df["Total_Volume"].values
        )
        np.testing.assert_allclose(
            sql_df["Avg_Close_Price"].values, pd_df["Avg_Close_Price"].values, rtol=1e-5
        )

    def test_pivot_matrix_computation(self):
        piv_sql = warehouse.query_pivot_matrix(self.db_path)
        self.assertEqual(piv_sql.shape, (5, 13))

        # Check values match reference within floating-point tolerance
        pd_piv = pd.pivot_table(
            self.df_ref,
            values="Daily_Return",
            index="Sector",
            columns="Year_Quarter",
            aggfunc="mean",
        )
        # Difference between SQL and Pandas mean before round(3) is <= 1e-12
        for sec in piv_sql.index:
            for col in piv_sql.columns:
                val_sql = piv_sql.loc[sec, col]
                val_pd_round = round(pd_piv.loc[sec, col], 3)
                # Max difference between rounded forms due to tie-breaking is <= 0.001
                self.assertAlmostEqual(val_sql, val_pd_round, delta=0.0011)

    def test_stock_drilldown_for_all_tickers(self):
        tickers = sorted(self.df_ref["Ticker"].unique())
        for ticker in tickers:
            dd_sql = warehouse.query_stock_drilldown(self.db_path, ticker=ticker)
            dd_ref = self.df_ref[self.df_ref["Ticker"] == ticker].sort_values("Date_dt")

            self.assertEqual(len(dd_sql), len(dd_ref))
            np.testing.assert_array_equal(
                dd_sql["Volume"].values, dd_ref["Volume"].values
            )
            np.testing.assert_allclose(
                dd_sql["Close_Price"].values, dd_ref["Close_Price"].values, rtol=1e-5
            )

    def test_invalid_grain_rejected(self):
        with self.assertRaises(ValueError):
            warehouse.query_sector_trend(self.db_path, time_grain="Hourly")


class TestConcurrencyAndLocking(unittest.TestCase):
    """Verifies reader/rebuild synchronization under process-wide DB_LOCK."""

    @classmethod
    def setUpClass(cls):
        warehouse.init_warehouse()
        cls.db_path = warehouse.DEFAULT_DB_PATH

    def test_concurrent_readers_and_rebuild(self):
        reader_errors: List[Exception] = []
        stop_event = threading.Event()

        def reader_worker():
            while not stop_event.is_set():
                try:
                    kpi = warehouse.query_kpi_summary(self.db_path)
                    if kpi["total_records"] != 14800:
                        reader_errors.append(
                            ValueError(f"Unexpected count: {kpi['total_records']}")
                        )
                    trend = warehouse.query_sector_trend(
                        self.db_path, time_grain="Quarterly"
                    )
                    if len(trend) != 65:
                        reader_errors.append(
                            ValueError(f"Unexpected trend rows: {len(trend)}")
                        )
                except Exception as e:
                    reader_errors.append(e)
                time.sleep(0.005)

        # Spawn 6 reader threads
        threads = [threading.Thread(target=reader_worker) for _ in range(6)]
        for t in threads:
            t.start()

        # Perform a forced warehouse rebuild concurrently
        try:
            res = warehouse.init_warehouse(force_rebuild=True)
            self.assertEqual(res["status"], "rebuilt")
        finally:
            stop_event.set()
            for t in threads:
                t.join()

        self.assertEqual(
            len(reader_errors), 0, f"Concurrent reader errors occurred: {reader_errors}"
        )


class TestStreamlitIntegration(unittest.TestCase):
    """Validates end-to-end Streamlit app execution and tab navigation via AppTest."""

    def test_app_loads_without_exceptions(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App threw exception on load: {at.exception}")

    def test_market_overview_olap_tab(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        # Navigate to Tab 1
        at.sidebar.radio[0].set_value("🌐 Market Overview & OLAP").run()
        self.assertFalse(
            at.exception, f"Market Overview Tab threw exception: {at.exception}"
        )

    def test_exploratory_analytics_tab(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("📊 Exploratory Analytics").run()
        self.assertFalse(
            at.exception, f"Exploratory Analytics Tab threw exception: {at.exception}"
        )

    def test_predictive_modeling_tab(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("⚡ Predictive Modeling").run()
        self.assertFalse(
            at.exception, f"Predictive Modeling Tab threw exception: {at.exception}"
        )

    def test_asset_clusters_and_risk_tab(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("🎯 Asset Clusters & Risk").run()
        self.assertFalse(
            at.exception, f"Asset Clusters Tab threw exception: {at.exception}"
        )

    def test_co_movement_engine_tab(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        at.sidebar.radio[0].set_value("🔄 Co-Movement Engine").run()
        self.assertFalse(
            at.exception, f"Co-Movement Engine Tab threw exception: {at.exception}"
        )

    def test_technical_architecture_tab(self):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        # Navigate to Tab 6
        at.sidebar.radio[0].set_value("📄 Technical Architecture").run()
        self.assertFalse(
            at.exception, f"Technical Architecture Tab threw exception: {at.exception}"
        )


if __name__ == "__main__":
    unittest.main()
