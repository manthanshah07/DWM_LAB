"""
QuantPulse — SQLite Dimensional Data Warehouse & OLAP Engine
============================================================
Implements a normalized Snowflake schema for stock market data,
ACID-compliant transactional ETL with byte-snapshot SHA-256 freshness tracking,
process-wide reader/rebuild synchronization via threading.Lock, and
parameterized SQL queries powering multidimensional OLAP operations.
"""

from __future__ import annotations

import hashlib
import io
import platform
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema Constants & Dimensional Mappings
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "2.0.0"
DEFAULT_DB_PATH = Path("dwm_stock_warehouse.db")
DEFAULT_CSV_PATH = Path("stock_market_data.csv")

# Whitelist of allowed OLAP aggregation grains
ALLOWED_GRAINS = {"Yearly", "Quarterly", "Monthly", "Daily"}

# Grain to SQL dimension column expression
TEMPORAL_DIM_COL = {
    "Yearly": "d.Year",
    "Quarterly": "d.Year_Quarter",
    "Monthly": "d.Year_Month",
    "Daily": "d.Full_Date",
}

# Grain to Output DataFrame column mapping
TEMPORAL_OUTPUT_MAP = {
    "Yearly": "Year",
    "Quarterly": "Year_Quarter",
    "Monthly": "YearMonth",
    "Daily": "Date_dt",
}

# Authentic company names for the 20 large-cap NSE equities
COMPANY_METADATA: Dict[str, str] = {
    "BAJAJ-AUTO": "Bajaj Auto Ltd.",
    "BRITANNIA": "Britannia Industries Ltd.",
    "HCLTECH": "HCL Technologies Ltd.",
    "HDFCBANK": "HDFC Bank Ltd.",
    "HINDUNILVR": "Hindustan Unilever Ltd.",
    "ICICIBANK": "ICICI Bank Ltd.",
    "INFY": "Infosys Ltd.",
    "ITC": "ITC Ltd.",
    "KOTAKBANK": "Kotak Mahindra Bank Ltd.",
    "M&M": "Mahindra & Mahindra Ltd.",
    "MARUTI": "Maruti Suzuki India Ltd.",
    "NESTLEIND": "Nestle India Ltd.",
    "NTPC": "NTPC Ltd.",
    "ONGC": "Oil and Natural Gas Corporation Ltd.",
    "POWERGRID": "Power Grid Corporation of India Ltd.",
    "RELIANCE": "Reliance Industries Ltd.",
    "SBIN": "State Bank of India",
    "TATAMOTORS": "Tata Motors Passenger Vehicles Ltd.",
    "TCS": "Tata Consultancy Services Ltd.",
    "WIPRO": "Wipro Ltd.",
}

# Process-wide lock coordinating readers and rebuilds within a single Python process
# (Note: Multi-process coordination requires external/OS distributed locks)
DB_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Thread-Synchronized Connection Management
# ---------------------------------------------------------------------------
@contextmanager
def get_readonly_connection(db_path: Path | str, timeout: float = 10.0):
    """
    Acquires the process-wide lock, yields an immutable read-only SQLite
    connection, and guarantees the connection is closed before releasing the lock.
    """
    path = Path(db_path).resolve()
    with DB_LOCK:
        conn = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro",
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# DDL: Normalized Snowflake Schema
# ---------------------------------------------------------------------------
DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS Dim_Sector (
        Sector_Key   INTEGER PRIMARY KEY AUTOINCREMENT,
        Sector_Name  TEXT NOT NULL UNIQUE COLLATE NOCASE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Dim_Exchange (
        Exchange_Key   INTEGER PRIMARY KEY AUTOINCREMENT,
        Exchange_Name  TEXT NOT NULL UNIQUE COLLATE NOCASE
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Dim_Stock (
        Stock_Key     INTEGER PRIMARY KEY AUTOINCREMENT,
        Ticker        TEXT NOT NULL UNIQUE COLLATE NOCASE,
        Company_Name  TEXT NOT NULL,
        Sector_Key    INTEGER NOT NULL,
        Exchange_Key  INTEGER NOT NULL,
        FOREIGN KEY (Sector_Key)   REFERENCES Dim_Sector(Sector_Key),
        FOREIGN KEY (Exchange_Key) REFERENCES Dim_Exchange(Exchange_Key)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Dim_Date (
        Date_Key     INTEGER PRIMARY KEY, -- Smart Key YYYYMMDD
        Full_Date    TEXT NOT NULL UNIQUE,
        Day          INTEGER NOT NULL,
        Month        INTEGER NOT NULL,
        Month_Name   TEXT NOT NULL,
        Quarter      INTEGER NOT NULL,
        Year         INTEGER NOT NULL,
        Year_Quarter TEXT NOT NULL,
        Year_Month   TEXT NOT NULL,
        Day_of_Week  TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS Fact_DailyTrades (
        Trade_ID            INTEGER PRIMARY KEY AUTOINCREMENT,
        Date_Key            INTEGER NOT NULL,
        Stock_Key           INTEGER NOT NULL,
        Open_Price          REAL NOT NULL,
        High_Price          REAL NOT NULL,
        Low_Price           REAL NOT NULL,
        Close_Price         REAL NOT NULL,
        Volume              INTEGER NOT NULL,
        Daily_Return        REAL NOT NULL,
        Intraday_Volatility REAL NOT NULL,
        FOREIGN KEY (Date_Key)  REFERENCES Dim_Date(Date_Key),
        FOREIGN KEY (Stock_Key) REFERENCES Dim_Stock(Stock_Key),
        CONSTRAINT uq_session_stock UNIQUE (Date_Key, Stock_Key)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS _etl_metadata (
        key         TEXT PRIMARY KEY,
        value       TEXT NOT NULL,
        updated_at  TEXT NOT NULL
    );
    """,
    # Explicit secondary index on Stock_Key (Date_Key is covered as leading column of uq_session_stock autoindex)
    """
    CREATE INDEX IF NOT EXISTS idx_fact_stock ON Fact_DailyTrades(Stock_Key);
    """,
    # Dimension outrigger and temporal query indexes
    """
    CREATE INDEX IF NOT EXISTS idx_dim_stock_sector ON Dim_Stock(Sector_Key);
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_dim_date_year ON Dim_Date(Year);
    """,
]


# ---------------------------------------------------------------------------
# ETL & Source Freshness Detection
# ---------------------------------------------------------------------------
def load_source_snapshot(csv_path: Path | str) -> Tuple[bytes, str, pd.DataFrame]:
    """
    Reads the source CSV file into an immutable in-memory byte snapshot.
    Derives SHA-256 fingerprint and parsed DataFrame from the identical byte
    buffer, eliminating Time-of-Check to Time-of-Use (TOCTOU) file modification races.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Source dataset not found at {path}")

    raw_bytes = path.read_bytes()
    source_hash = hashlib.sha256(raw_bytes).hexdigest()
    df = pd.read_csv(io.BytesIO(raw_bytes))

    # Validate essential schema columns
    required_cols = {
        "Date",
        "Ticker",
        "Sector",
        "Exchange",
        "Open_Price",
        "High_Price",
        "Low_Price",
        "Close_Price",
        "Volume",
        "Daily_Return",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Source CSV missing required columns: {sorted(missing)}")

    return raw_bytes, source_hash, df


def is_warehouse_fresh(
    db_path: Path | str, current_hash: str, parsed_row_count: int
) -> bool:
    """
    Validates whether the existing warehouse database is fresh, healthy, and consistent
    with the source dataset. Executes a live SELECT COUNT(*) on Fact_DailyTrades.
    """
    path = Path(db_path)
    if not path.exists():
        return False

    try:
        with get_readonly_connection(path) as conn:
            cur = conn.cursor()

            # 1. Schema version
            cur.execute("SELECT value FROM _etl_metadata WHERE key = 'schema_version'")
            r = cur.fetchone()
            if not r or r[0] != SCHEMA_VERSION:
                return False

            # 2. Source SHA-256
            cur.execute("SELECT value FROM _etl_metadata WHERE key = 'source_sha256'")
            r = cur.fetchone()
            if not r or r[0] != current_hash:
                return False

            # 3. Stored source & fact row counts
            cur.execute("SELECT value FROM _etl_metadata WHERE key = 'source_row_count'")
            r = cur.fetchone()
            if not r or int(r[0]) != parsed_row_count:
                return False

            cur.execute("SELECT value FROM _etl_metadata WHERE key = 'fact_row_count'")
            r = cur.fetchone()
            if not r or int(r[0]) != parsed_row_count:
                return False

            # 4. Live fact table row count
            cur.execute("SELECT COUNT(*) FROM Fact_DailyTrades")
            r = cur.fetchone()
            if not r or int(r[0]) != parsed_row_count:
                return False

            # 5. Lightweight integrity check
            cur.execute("PRAGMA quick_check")
            r = cur.fetchone()
            if not r or r[0] != "ok":
                return False

            # 6. Foreign key integrity check
            cur.execute("PRAGMA foreign_key_check")
            fk_violations = cur.fetchall()
            if fk_violations:
                return False

            return True
    except Exception:
        return False


def init_warehouse(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    db_path: Path | str = DEFAULT_DB_PATH,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Idempotent database initialization and transactional ETL pipeline.
    If the warehouse is fresh and matches the source snapshot, returns immediately (fast path).
    Otherwise, builds a clean staging database, executes PRAGMA wal_checkpoint(TRUNCATE),
    and atomically swaps it into production under DB_LOCK.
    """
    csv_p = Path(csv_path)
    db_p = Path(db_path)

    # 1. Load immutable byte snapshot
    _, source_hash, source_df = load_source_snapshot(csv_p)
    parsed_row_count = len(source_df)

    # 2. Check freshness (fast path)
    if not force_rebuild and is_warehouse_fresh(db_p, source_hash, parsed_row_count):
        return {
            "status": "fresh",
            "db_path": str(db_p),
            "rows_ingested": parsed_row_count,
            "rebuilt": False,
        }

    # 3. Staging build path
    staging_db = db_p.with_name(f"{db_p.stem}_staging.db")
    for ext in ("", "-wal", "-shm"):
        f = staging_db.with_name(staging_db.name + ext)
        if f.exists():
            f.unlink()

    # 4. Build staging warehouse transactionally
    conn = sqlite3.connect(str(staging_db))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")

        # Create schema
        for stmt in DDL_STATEMENTS:
            conn.execute(stmt)

        # A. Populate Dim_Sector
        unique_sectors = sorted(source_df["Sector"].astype(str).unique())
        conn.executemany(
            "INSERT INTO Dim_Sector (Sector_Name) VALUES (?);",
            [(s,) for s in unique_sectors],
        )
        sector_map = {
            row[1]: row[0]
            for row in conn.execute(
                "SELECT Sector_Key, Sector_Name FROM Dim_Sector;"
            ).fetchall()
        }

        # B. Populate Dim_Exchange
        unique_exchanges = sorted(source_df["Exchange"].astype(str).unique())
        conn.executemany(
            "INSERT INTO Dim_Exchange (Exchange_Name) VALUES (?);",
            [(e,) for e in unique_exchanges],
        )
        exchange_map = {
            row[1]: row[0]
            for row in conn.execute(
                "SELECT Exchange_Key, Exchange_Name FROM Dim_Exchange;"
            ).fetchall()
        }

        # C. Populate Dim_Stock
        unique_stocks = (
            source_df[["Ticker", "Sector", "Exchange"]]
            .drop_duplicates()
            .sort_values("Ticker")
        )
        stock_insert_data = []
        for _, r in unique_stocks.iterrows():
            ticker = str(r["Ticker"])
            sec_key = sector_map[str(r["Sector"])]
            ex_key = exchange_map[str(r["Exchange"])]
            comp_name = COMPANY_METADATA.get(ticker, f"{ticker} Ltd.")
            stock_insert_data.append((ticker, comp_name, sec_key, ex_key))

        conn.executemany(
            """
            INSERT INTO Dim_Stock (Ticker, Company_Name, Sector_Key, Exchange_Key)
            VALUES (?, ?, ?, ?);
            """,
            stock_insert_data,
        )
        stock_map = {
            row[1]: row[0]
            for row in conn.execute(
                "SELECT Stock_Key, Ticker FROM Dim_Stock;"
            ).fetchall()
        }

        # D. Populate Dim_Date (740 distinct trading sessions)
        date_series = pd.to_datetime(source_df["Date"].drop_duplicates().sort_values())
        date_insert_data = []
        for dt in date_series:
            d_key = int(dt.strftime("%Y%m%d"))
            full_date = dt.strftime("%Y-%m-%d")
            day = int(dt.day)
            month = int(dt.month)
            month_name = dt.month_name()
            quarter = int(dt.quarter)
            year = int(dt.year)
            year_quarter = f"{year}-Q{quarter}"
            year_month = dt.strftime("%Y-%m")
            day_of_week = dt.day_name()
            date_insert_data.append(
                (
                    d_key,
                    full_date,
                    day,
                    month,
                    month_name,
                    quarter,
                    year,
                    year_quarter,
                    year_month,
                    day_of_week,
                )
            )

        conn.executemany(
            """
            INSERT INTO Dim_Date (
                Date_Key, Full_Date, Day, Month, Month_Name,
                Quarter, Year, Year_Quarter, Year_Month, Day_of_Week
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            date_insert_data,
        )

        # E. Populate Fact_DailyTrades
        # Calculate Intraday Volatility: ((High - Low) / Open) * 100 rounded to 2 decimals
        volatility = (
            (
                (source_df["High_Price"] - source_df["Low_Price"])
                / source_df["Open_Price"]
            )
            * 100
        ).round(2)

        date_dt = pd.to_datetime(source_df["Date"])
        date_keys = date_dt.dt.strftime("%Y%m%d").astype(int).values
        stock_keys = source_df["Ticker"].map(stock_map).values

        fact_rows = list(
            zip(
                date_keys.tolist(),
                stock_keys.tolist(),
                source_df["Open_Price"].astype(float).values.tolist(),
                source_df["High_Price"].astype(float).values.tolist(),
                source_df["Low_Price"].astype(float).values.tolist(),
                source_df["Close_Price"].astype(float).values.tolist(),
                source_df["Volume"].astype(int).values.tolist(),
                source_df["Daily_Return"].astype(float).values.tolist(),
                volatility.astype(float).values.tolist(),
            )
        )

        conn.executemany(
            """
            INSERT INTO Fact_DailyTrades (
                Date_Key, Stock_Key, Open_Price, High_Price, Low_Price,
                Close_Price, Volume, Daily_Return, Intraday_Volatility
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            fact_rows,
        )

        # F. Insert Metadata
        now_ts = datetime.now(timezone.utc).isoformat()
        metadata_records = [
            ("schema_version", SCHEMA_VERSION, now_ts),
            ("source_sha256", source_hash, now_ts),
            ("source_row_count", str(parsed_row_count), now_ts),
            ("fact_row_count", str(parsed_row_count), now_ts),
            ("ingestion_timestamp", now_ts, now_ts),
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO _etl_metadata (key, value, updated_at) VALUES (?, ?, ?);",
            metadata_records,
        )

        conn.commit()

        # 5. Staging Validation Checks
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Fact_DailyTrades")
        stg_count = cur.fetchone()[0]
        if stg_count != parsed_row_count:
            raise RuntimeError(
                f"Row count mismatch in staging: expected {parsed_row_count}, got {stg_count}"
            )

        cur.execute("PRAGMA integrity_check")
        ic = cur.fetchone()[0]
        if ic != "ok":
            raise RuntimeError(f"Integrity check failed in staging: {ic}")

        cur.execute("PRAGMA foreign_key_check")
        fk_errs = cur.fetchall()
        if fk_errs:
            raise RuntimeError(f"Foreign key violations in staging: {fk_errs}")

        # 6. Checkpoint staging database to truncate WAL frames into main file
        cp_result = conn.execute("PRAGMA wal_checkpoint(TRUNCATE);").fetchone()
        if cp_result[0] != 0:
            raise RuntimeError(f"Staging WAL checkpoint failed: {cp_result}")

    finally:
        conn.close()

    # 7. Safe replacement under process-wide lock
    with DB_LOCK:
        # Clean up staging sidecars
        for ext in ("-wal", "-shm"):
            sidecar = staging_db.with_name(staging_db.name + ext)
            if sidecar.exists():
                sidecar.unlink()

        # Checkpoint and clean up production sidecars if existing
        if db_p.exists():
            try:
                p_conn = sqlite3.connect(str(db_p))
                try:
                    p_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                finally:
                    p_conn.close()
            except Exception:
                pass

            for ext in ("-wal", "-shm"):
                sidecar = db_p.with_name(db_p.name + ext)
                if sidecar.exists():
                    sidecar.unlink()

        # Atomic file system replacement
        staging_db.replace(db_p)

    return {
        "status": "rebuilt",
        "db_path": str(db_p),
        "rows_ingested": parsed_row_count,
        "rebuilt": True,
    }


# ---------------------------------------------------------------------------
# Parameterized SQL OLAP Operations
# ---------------------------------------------------------------------------
def _build_olap_where(
    slice_year: str = "All Years",
    dice_sectors: Optional[List[str]] = None,
    dice_exchange: str = "All",
) -> Tuple[str, List[Any]]:
    """
    Builds parameterized SQL WHERE clauses and bound parameter lists.
    Guarantees zero raw string interpolation of user input.
    """
    clauses: List[str] = []
    params: List[Any] = []

    # Temporal slice
    if slice_year != "All Years":
        clauses.append("d.Year = ?")
        params.append(int(slice_year))

    # Categorical dice: Sectors
    if dice_sectors is not None:
        if len(dice_sectors) == 0:
            # Force empty result safely
            clauses.append("1 = 0")
        else:
            placeholders = ", ".join(["?"] * len(dice_sectors))
            clauses.append(f"sec.Sector_Name IN ({placeholders})")
            params.extend(dice_sectors)

    # Categorical dice: Exchange
    if dice_exchange != "All":
        clauses.append("ex.Exchange_Name = ?")
        params.append(dice_exchange)

    where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return where_sql, params


def query_kpi_summary(
    db_path: Path | str = DEFAULT_DB_PATH,
    slice_year: str = "All Years",
    dice_sectors: Optional[List[str]] = None,
    dice_exchange: str = "All",
) -> Dict[str, Any]:
    """
    Executes SQL aggregation to retrieve top-level market summary KPIs:
    total records, distinct stocks, distinct sectors, aggregate volume,
    mean close price, and average daily return.
    """
    if dice_sectors is not None and len(dice_sectors) == 0:
        return {
            "total_records": 0,
            "active_symbols": 0,
            "active_sectors": 0,
            "aggregate_volume": 0,
            "avg_close_price": 0.0,
            "mean_daily_return": 0.0,
        }

    where_sql, params = _build_olap_where(slice_year, dice_sectors, dice_exchange)

    query = f"""
    SELECT
        COUNT(f.Trade_ID)                  AS total_records,
        COUNT(DISTINCT s.Stock_Key)        AS active_symbols,
        COUNT(DISTINCT s.Sector_Key)       AS active_sectors,
        COALESCE(SUM(f.Volume), 0)         AS aggregate_volume,
        COALESCE(AVG(f.Close_Price), 0.0)  AS avg_close_price,
        COALESCE(AVG(f.Daily_Return), 0.0) AS mean_daily_return
    FROM Fact_DailyTrades f
    JOIN Dim_Stock s     ON f.Stock_Key = s.Stock_Key
    JOIN Dim_Sector sec  ON s.Sector_Key = sec.Sector_Key
    JOIN Dim_Exchange ex ON s.Exchange_Key = ex.Exchange_Key
    JOIN Dim_Date d      ON f.Date_Key = d.Date_Key
    {where_sql};
    """

    with get_readonly_connection(db_path) as conn:
        cur = conn.cursor()
        cur.execute(query, params)
        row = cur.fetchone()

    return {
        "total_records": int(row["total_records"]),
        "active_symbols": int(row["active_symbols"]),
        "active_sectors": int(row["active_sectors"]),
        "aggregate_volume": int(row["aggregate_volume"]),
        "avg_close_price": float(row["avg_close_price"]),
        "mean_daily_return": float(row["mean_daily_return"]),
    }


def query_sector_trend(
    db_path: Path | str = DEFAULT_DB_PATH,
    slice_year: str = "All Years",
    dice_sectors: Optional[List[str]] = None,
    dice_exchange: str = "All",
    time_grain: str = "Quarterly",
) -> pd.DataFrame:
    """
    Performs OLAP roll-up / slice / dice aggregations across sectors.
    Supports Yearly, Quarterly, Monthly, and Daily aggregation grains.
    Guarantees proper 'Date_dt' column mapping for Daily grain and
    defensive empty DataFrame schema for empty selections.
    """
    if time_grain not in ALLOWED_GRAINS:
        raise ValueError(
            f"Invalid time grain '{time_grain}'. Must be one of {sorted(ALLOWED_GRAINS)}"
        )

    time_col = TEMPORAL_DIM_COL[time_grain]
    output_grain_col = TEMPORAL_OUTPUT_MAP[time_grain]

    expected_columns = [
        output_grain_col,
        "Sector",
        "Total_Volume",
        "Avg_Close_Price",
        "Avg_Daily_Return",
    ]

    # Handle empty sector selection defensively
    if dice_sectors is not None and len(dice_sectors) == 0:
        return pd.DataFrame(columns=expected_columns)

    where_sql, params = _build_olap_where(slice_year, dice_sectors, dice_exchange)

    query = f"""
    SELECT
        {time_col}             AS Time_Bucket,
        sec.Sector_Name        AS Sector,
        SUM(f.Volume)          AS Total_Volume,
        AVG(f.Close_Price)     AS Avg_Close_Price,
        AVG(f.Daily_Return)    AS Avg_Daily_Return
    FROM Fact_DailyTrades f
    JOIN Dim_Stock s     ON f.Stock_Key = s.Stock_Key
    JOIN Dim_Sector sec  ON s.Sector_Key = sec.Sector_Key
    JOIN Dim_Exchange ex ON s.Exchange_Key = ex.Exchange_Key
    JOIN Dim_Date d      ON f.Date_Key = d.Date_Key
    {where_sql}
    GROUP BY {time_col}, sec.Sector_Name
    ORDER BY {time_col} ASC, sec.Sector_Name ASC;
    """

    with get_readonly_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame(columns=expected_columns)

    # Rename temporal bucket to target dashboard column
    df = df.rename(columns={"Time_Bucket": output_grain_col})
    return df


def query_pivot_matrix(
    db_path: Path | str = DEFAULT_DB_PATH,
    slice_year: str = "All Years",
    dice_sectors: Optional[List[str]] = None,
    dice_exchange: str = "All",
    metric: str = "Daily_Return",
) -> pd.DataFrame:
    """
    Performs OLAP Pivoting: Sector vs Year_Quarter.
    The underlying aggregation is computed in SQL; pandas is used strictly for
    reshaping the 2D relational result into the matrix format.
    """
    if dice_sectors is not None and len(dice_sectors) == 0:
        return pd.DataFrame()

    allowed_metrics = {
        "Daily_Return": "AVG(f.Daily_Return)",
        "Close_Price": "AVG(f.Close_Price)",
        "Volume": "SUM(f.Volume)",
        "Intraday_Volatility": "AVG(f.Intraday_Volatility)",
    }
    metric_expr = allowed_metrics.get(metric, "AVG(f.Daily_Return)")

    where_sql, params = _build_olap_where(slice_year, dice_sectors, dice_exchange)

    query = f"""
    SELECT
        d.Year_Quarter         AS Year_Quarter,
        sec.Sector_Name        AS Sector,
        {metric_expr}          AS Metric_Val
    FROM Fact_DailyTrades f
    JOIN Dim_Stock s     ON f.Stock_Key = s.Stock_Key
    JOIN Dim_Sector sec  ON s.Sector_Key = sec.Sector_Key
    JOIN Dim_Exchange ex ON s.Exchange_Key = ex.Exchange_Key
    JOIN Dim_Date d      ON f.Date_Key = d.Date_Key
    {where_sql}
    GROUP BY d.Year_Quarter, sec.Sector_Name
    ORDER BY d.Year_Quarter ASC, sec.Sector_Name ASC;
    """

    with get_readonly_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if df.empty:
        return pd.DataFrame()

    pivot_matrix = (
        df.pivot_table(
            index="Sector",
            columns="Year_Quarter",
            values="Metric_Val",
            aggfunc="mean",
        )
        .round(3)
    )
    return pivot_matrix


def query_stock_drilldown(
    db_path: Path | str = DEFAULT_DB_PATH,
    ticker: str = "TCS",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Executes OLAP drill-down to individual stock trading records at daily granularity.
    Maps date column to 'Date_dt' and returns rows chronologically sorted.
    """
    clauses = ["s.Ticker = ?"]
    params: List[Any] = [ticker]

    if start_date:
        clauses.append("d.Full_Date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("d.Full_Date <= ?")
        params.append(end_date)

    where_sql = "WHERE " + " AND ".join(clauses)

    query = f"""
    SELECT
        d.Full_Date           AS Date_dt,
        s.Ticker              AS Ticker,
        sec.Sector_Name       AS Sector,
        f.Open_Price          AS Open_Price,
        f.High_Price          AS High_Price,
        f.Low_Price           AS Low_Price,
        f.Close_Price         AS Close_Price,
        f.Volume              AS Volume,
        f.Daily_Return        AS Daily_Return,
        f.Intraday_Volatility AS Intraday_Volatility
    FROM Fact_DailyTrades f
    JOIN Dim_Stock s    ON f.Stock_Key = s.Stock_Key
    JOIN Dim_Sector sec ON s.Sector_Key = sec.Sector_Key
    JOIN Dim_Date d     ON f.Date_Key = d.Date_Key
    {where_sql}
    ORDER BY d.Date_Key ASC;
    """

    with get_readonly_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=params)

    if not df.empty:
        df["Date_dt"] = pd.to_datetime(df["Date_dt"])
    return df


def get_filter_options(db_path: Path | str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Retrieves distinct years, sectors, exchanges, and ticker hierarchies
    directly from the dimensional warehouse tables.
    """
    with get_readonly_connection(db_path) as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(DISTINCT Date_Key) FROM Dim_Date;")
        num_sessions = int(cur.fetchone()[0])

        cur.execute("SELECT DISTINCT Year FROM Dim_Date ORDER BY Year ASC;")
        years = [row[0] for row in cur.fetchall()]

        cur.execute("SELECT Sector_Name FROM Dim_Sector ORDER BY Sector_Name ASC;")
        sectors = [row[0] for row in cur.fetchall()]

        cur.execute("SELECT Exchange_Name FROM Dim_Exchange ORDER BY Exchange_Name ASC;")
        exchanges = [row[0] for row in cur.fetchall()]

        cur.execute(
            """
            SELECT s.Ticker, sec.Sector_Name
            FROM Dim_Stock s
            JOIN Dim_Sector sec ON s.Sector_Key = sec.Sector_Key
            ORDER BY s.Ticker ASC;
            """
        )
        stock_rows = cur.fetchall()

    tickers_by_sector: Dict[str, List[str]] = {s: [] for s in sectors}
    all_tickers = []
    for ticker, sec_name in stock_rows:
        all_tickers.append(ticker)
        if sec_name in tickers_by_sector:
            tickers_by_sector[sec_name].append(ticker)

    return {
        "num_sessions": num_sessions,
        "years": years,
        "sectors": sectors,
        "exchanges": exchanges,
        "all_tickers": sorted(all_tickers),
        "tickers_by_sector": tickers_by_sector,
    }


def get_warehouse_stats(db_path: Path | str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Retrieves live physical database file size, verified table row counts,
    indexes, and dynamically detected runtime system specs.
    """
    path = Path(db_path)
    file_size_bytes = path.stat().st_size if path.exists() else 0

    table_counts: Dict[str, int] = {}
    metadata: Dict[str, str] = {}
    indexes: List[str] = []

    if path.exists():
        with get_readonly_connection(path) as conn:
            cur = conn.cursor()
            for tbl in [
                "Fact_DailyTrades",
                "Dim_Date",
                "Dim_Stock",
                "Dim_Sector",
                "Dim_Exchange",
            ]:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {tbl};")
                    table_counts[tbl] = int(cur.fetchone()[0])
                except Exception:
                    table_counts[tbl] = 0

            try:
                cur.execute("SELECT key, value FROM _etl_metadata;")
                for k, v in cur.fetchall():
                    metadata[k] = v
            except Exception:
                pass

            try:
                cur.execute(
                    "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
                )
                indexes = [r[0] for r in cur.fetchall()]
            except Exception:
                pass

    runtime_info = {
        "python_version": sys.version.split()[0],
        "python_compiler": (
            sys.version.split("(")[-1].strip(")") if "(" in sys.version else "Unknown"
        ),
        "sqlite_version": sqlite3.sqlite_version,
        "os_platform": platform.platform(),
        "cpu_architecture": platform.machine(),
    }

    return {
        "db_path": str(path),
        "exists": path.exists(),
        "file_size_bytes": file_size_bytes,
        "file_size_kb": round(file_size_bytes / 1024, 2),
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "table_counts": table_counts,
        "metadata": metadata,
        "indexes": indexes,
        "runtime_info": runtime_info,
    }
