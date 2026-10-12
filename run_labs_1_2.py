import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path


COMPANY_METADATA = {
    "TCS": ("Tata Consultancy Services Ltd.", "IT", "IT Services & Consulting"),
    "INFY": ("Infosys Ltd.", "IT", "IT Services & Consulting"),
    "WIPRO": ("Wipro Ltd.", "IT", "IT Services & Consulting"),
    "HCLTECH": ("HCL Technologies Ltd.", "IT", "IT Services & Consulting"),
    "HDFCBANK": ("HDFC Bank Ltd.", "Banking", "Private Sector Banking & Financial Services"),
    "ICICIBANK": ("ICICI Bank Ltd.", "Banking", "Private Sector Banking & Financial Services"),
    "SBIN": ("State Bank of India", "Banking", "Public & Private Banking Services"),
    "KOTAKBANK": ("Kotak Mahindra Bank Ltd.", "Banking", "Private Sector Banking & Financial Services"),
    "TATAMOTORS": ("Tata Motors Passenger Vehicles Ltd.", "Auto", "Automotive & Mobility Manufacturing"),
    "MARUTI": ("Maruti Suzuki India Ltd.", "Auto", "Automotive & Mobility Manufacturing"),
    "M&M": ("Mahindra & Mahindra Ltd.", "Auto", "Automotive & Mobility Manufacturing"),
    "BAJAJ-AUTO": ("Bajaj Auto Ltd.", "Auto", "Automotive & Mobility Manufacturing"),
    "RELIANCE": ("Reliance Industries Ltd.", "Energy", "Oil, Gas & Power Generation"),
    "ONGC": ("Oil and Natural Gas Corporation Ltd.", "Energy", "Oil, Gas & Power Generation"),
    "NTPC": ("NTPC Ltd.", "Energy", "Oil, Gas & Power Generation"),
    "POWERGRID": ("Power Grid Corporation of India Ltd.", "Energy", "Oil, Gas & Power Generation"),
    "ITC": ("ITC Ltd.", "FMCG", "Fast-Moving Consumer Goods"),
    "HINDUNILVR": ("Hindustan Unilever Ltd.", "FMCG", "Fast-Moving Consumer Goods"),
    "NESTLEIND": ("Nestle India Ltd.", "FMCG", "Fast-Moving Consumer Goods"),
    "BRITANNIA": ("Britannia Industries Ltd.", "FMCG", "Fast-Moving Consumer Goods"),
}

SECTOR_CATEGORIES = {
    "IT": "Information Technology & Software Services",
    "Banking": "Banking & Financial Intermediation",
    "Auto": "Automobile & Commercial Vehicles",
    "Energy": "Energy, Oil & Power Utilities",
    "FMCG": "Consumer Staples & Packaged Foods",
}


def df_to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)


def run_experiments_1_and_2():
    print("=" * 80)
    print("DWM EXPERIMENTS 01 & 02: DATA WAREHOUSE SCHEMA DESIGN & OLAP EXECUTION")
    print("=" * 80)

    # Load CSV
    df = pd.read_csv("stock_market_data.csv")
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["TimeKey"] = df["Date_dt"].dt.strftime("%Y%m%d").astype(int)
    df["Day"] = df["Date_dt"].dt.day
    df["Month"] = df["Date_dt"].dt.month
    df["MonthName"] = df["Date_dt"].dt.month_name()
    df["Quarter"] = "Q" + df["Date_dt"].dt.quarter.astype(str)
    df["Year"] = df["Date_dt"].dt.year
    df["DayOfWeek"] = df["Date_dt"].dt.day_name()
    df["Volatility"] = (
        ((df["High_Price"] - df["Low_Price"]) / df["Open_Price"]) * 100
    ).round(2)

    # Connect to SQLite warehouse database (also persisted on disk for inspection)
    db_path = Path("dwm_stock_warehouse.db")
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # 1. Execute Experiment 01 DDL (lab1_schema.sql)
    schema_sql = Path("lab1_schema.sql").read_text(encoding="utf-8")
    cur.executescript(schema_sql)
    print("[Exp 01] Executed lab1_schema.sql: Created Star, Snowflake, and Fact Constellation tables.")

    # 2. Populate Dim_Time
    dim_time_df = (
        df[["TimeKey", "Date", "Day", "Month", "MonthName", "Quarter", "Year", "DayOfWeek"]]
        .drop_duplicates()
        .sort_values("TimeKey")
        .reset_index(drop=True)
    )
    dim_time_df.to_sql("Dim_Time", conn, if_exists="append", index=False)

    # 3. Populate Dim_Exchange
    dim_exchange_df = pd.DataFrame(
        [{"ExchangeKey": 1, "ExchangeName": "NSE", "Country": "India", "Currency": "INR"}]
    )
    dim_exchange_df.to_sql("Dim_Exchange", conn, if_exists="append", index=False)

    # 4. Populate Dim_Sector (Snowflake)
    sectors_sorted = sorted(SECTOR_CATEGORIES.keys())
    dim_sector_df = pd.DataFrame(
        [
            {
                "SectorKey": idx + 1,
                "SectorName": sec,
                "IndustryCategory": SECTOR_CATEGORIES[sec],
            }
            for idx, sec in enumerate(sectors_sorted)
        ]
    )
    dim_sector_df.to_sql("Dim_Sector", conn, if_exists="append", index=False)
    sector_key_map = dict(zip(dim_sector_df["SectorName"], dim_sector_df["SectorKey"]))

    # 5. Populate Dim_Stock (Star) & Dim_Stock_Snowflake (Snowflake)
    tickers_sorted = sorted(df["Ticker"].unique())
    dim_stock_rows = []
    dim_stock_sf_rows = []
    for idx, ticker in enumerate(tickers_sorted):
        comp_name, sec_name, _ = COMPANY_METADATA[ticker]
        stock_key = idx + 1
        dim_stock_rows.append(
            {
                "StockKey": stock_key,
                "Ticker": ticker,
                "CompanyName": comp_name,
                "Sector": sec_name,
                "Exchange": "NSE",
            }
        )
        dim_stock_sf_rows.append(
            {
                "StockKey": stock_key,
                "Ticker": ticker,
                "CompanyName": comp_name,
                "SectorKey": sector_key_map[sec_name],
                "ExchangeKey": 1,
            }
        )

    dim_stock_df = pd.DataFrame(dim_stock_rows)
    dim_stock_sf_df = pd.DataFrame(dim_stock_sf_rows)
    dim_stock_df.to_sql("Dim_Stock", conn, if_exists="append", index=False)
    dim_stock_sf_df.to_sql("Dim_Stock_Snowflake", conn, if_exists="append", index=False)
    stock_key_map = dict(zip(dim_stock_df["Ticker"], dim_stock_df["StockKey"]))

    # 6. Populate Fact_DailyTrades (Star) & Fact_DailyTrades_Snowflake (Snowflake)
    fact_df = pd.DataFrame(
        {
            "TimeKey": df["TimeKey"],
            "StockKey": df["Ticker"].map(stock_key_map),
            "ExchangeKey": 1,
            "OpenPrice": df["Open_Price"],
            "HighPrice": df["High_Price"],
            "LowPrice": df["Low_Price"],
            "ClosePrice": df["Close_Price"],
            "Volume": df["Volume"],
            "DailyReturn": df["Daily_Return"],
            "Volatility": df["Volatility"],
        }
    )
    fact_df.to_sql("Fact_DailyTrades", conn, if_exists="append", index=False)
    fact_df.to_sql("Fact_DailyTrades_Snowflake", conn, if_exists="append", index=False)

    # 7. Populate Fact_QuarterlyFinancials (Fact Constellation Schema)
    # Synthesize realistic quarterly fundamentals aligned with quarter-end TimeKeys
    np.random.seed(42)
    q_end_dates = (
        dim_time_df.groupby(["Year", "Quarter"], as_index=False)["TimeKey"]
        .max()
        .sort_values(["Year", "Quarter"])
    )
    base_rev_map = {
        "RELIANCE": 235000, "TCS": 61000, "HDFCBANK": 78000, "ICICIBANK": 56000,
        "SBIN": 112000, "ONGC": 138000, "TATAMOTORS": 105000, "NTPC": 44000,
        "INFY": 39500, "MARUTI": 36500, "M&M": 34000, "HCLTECH": 28000,
        "ITC": 18500, "KOTAKBANK": 24000, "WIPRO": 22500, "HINDUNILVR": 15400,
        "POWERGRID": 11600, "BAJAJ-AUTO": 11800, "NESTLEIND": 5100, "BRITANNIA": 4400,
    }
    qf_rows = []
    for _, q_row in q_end_dates.iterrows():
        for ticker, stock_key in stock_key_map.items():
            base_rev = base_rev_map[ticker]
            growth_factor = 1.0 + 0.03 * (int(q_row["Year"]) - 2023) + np.random.uniform(-0.02, 0.04)
            q_rev = round(base_rev * growth_factor, 2)
            op_margin = round(np.random.uniform(16.5, 28.5), 2)
            net_profit = round(q_rev * (op_margin / 100.0) * 0.72, 2)
            eps = round(net_profit / np.random.uniform(180, 420), 2)
            qf_rows.append(
                {
                    "TimeKey": int(q_row["TimeKey"]),
                    "StockKey": int(stock_key),
                    "ExchangeKey": 1,
                    "FiscalYear": int(q_row["Year"]),
                    "FiscalQuarter": str(q_row["Quarter"]),
                    "QuarterlyRevenue": q_rev,
                    "NetProfit": net_profit,
                    "EPS": eps,
                    "OperatingMargin": op_margin,
                }
            )
    qf_df = pd.DataFrame(qf_rows)
    qf_df.to_sql("Fact_QuarterlyFinancials", conn, if_exists="append", index=False)
    conn.commit()

    # Verify table row counts for Experiment 01
    exp1_tables = [
        "Dim_Time",
        "Dim_Exchange",
        "Dim_Stock",
        "Fact_DailyTrades",
        "Dim_Sector",
        "Dim_Stock_Snowflake",
        "Fact_DailyTrades_Snowflake",
        "Fact_QuarterlyFinancials",
    ]
    count_rows = []
    for tbl in exp1_tables:
        cnt = pd.read_sql_query(f"SELECT COUNT(*) AS cnt FROM {tbl}", conn)["cnt"].iloc[0]
        schema_type = (
            "Conformed Dimension (All Schemas)"
            if tbl in ("Dim_Time", "Dim_Exchange")
            else "Star & Fact Constellation"
            if tbl in ("Dim_Stock", "Fact_DailyTrades")
            else "Snowflake Schema"
            if "Snowflake" in tbl or tbl == "Dim_Sector"
            else "Fact Constellation (Galaxy Schema)"
        )
        count_rows.append({"Table_Name": tbl, "Schema_Role": schema_type, "Row_Count": int(cnt)})
    table_summary_df = pd.DataFrame(count_rows)

    print("\n--- EXPERIMENT 01: POPULATED DATA WAREHOUSE TABLE SUMMARY ---")
    print(table_summary_df.to_string(index=False))

    # Parse and execute the 6 OLAP queries from lab2_olap.sql
    olap_sql_text = Path("lab2_olap.sql").read_text(encoding="utf-8")
    raw_queries = [q.strip() for q in olap_sql_text.split(";") if q.strip()]

    olap_titles = [
        (
            "1. SLICE OPERATION",
            "Isolate trading performance for a single time slice (`Year = 2024 AND MonthName = 'January'`).",
        ),
        (
            "2. DICE OPERATION",
            "Extract a sub-cube across multiple dimensions (`Sector IN ('IT', 'Banking') AND Year = 2024 AND Quarter = 'Q1' AND ExchangeName = 'NSE'`).",
        ),
        (
            "3. ROLL-UP OPERATION",
            "Aggregate daily stock metrics up the time hierarchy (`Day -> Month -> Year`) by Sector (`Year = 2025`).",
        ),
        (
            "4. DRILL-DOWN OPERATION",
            "Navigate from high-level `Energy` Sector summary down to individual Stock Tickers and Quarters using the **Snowflake Schema** (`Dim_Sector -> Dim_Stock_Snowflake -> Fact_DailyTrades_Snowflake`).",
        ),
        (
            "5. PIVOT (CROSS-TABULATION) OPERATION",
            "Rotate dimensional axes to cross-tabulate `Sector` (rows) against `Quarter` (`Q1`–`Q4` columns) for Average Daily Return (%) and Traded Volume (Millions).",
        ),
        (
            "6. DRILL-ACROSS OPERATION",
            "Join two distinct Fact Tables (`Fact_DailyTrades` and `Fact_QuarterlyFinancials`) across conformed dimensions (`Dim_Stock`, `Dim_Time`) in the **Fact Constellation Schema**.",
        ),
    ]

    md_sections = []
    md_sections.append(
        """# Data Warehousing and Mining (DWM) — Experiments 01 & 02
## Experiment 01: Build Data Warehouse / Data Mart & Design Dimensional Data Models (Star, Snowflake, Fact Constellation)
## Experiment 02: Perform OLAP Operations on the Dimensional Model (Slice, Dice, Roll-Up, Drill-Down, Pivot, Drill-Across)

**Dataset:** `stock_market_data.csv` (`14,800` daily trading records across `20` NSE stocks and `5` sectors)  
**SQL Scripts:** [`lab1_schema.sql`](lab1_schema.sql) | [`lab2_olap.sql`](lab2_olap.sql) | **Execution Engine:** [`run_labs_1_2.py`](run_labs_1_2.py)

---

## Part 1 — Experiment 01: Dimensional Data Warehouse Modeling

### 1.1 Star Schema Architecture
A central denormalized fact table (`Fact_DailyTrades`) surrounded by single-join dimension tables (`Dim_Time`, `Dim_Stock`, `Dim_Exchange`).

```mermaid
erDiagram
    Dim_Time ||--o{ Fact_DailyTrades : "TimeKey"
    Dim_Stock ||--o{ Fact_DailyTrades : "StockKey"
    Dim_Exchange ||--o{ Fact_DailyTrades : "ExchangeKey"

    Dim_Time {
        int TimeKey PK
        text Date
        int Day
        int Month
        text MonthName
        text Quarter
        int Year
        text DayOfWeek
    }
    Dim_Stock {
        int StockKey PK
        text Ticker
        text CompanyName
        text Sector
        text Exchange
    }
    Dim_Exchange {
        int ExchangeKey PK
        text ExchangeName
        text Country
        text Currency
    }
    Fact_DailyTrades {
        int TradeID PK
        int TimeKey FK
        int StockKey FK
        int ExchangeKey FK
        real OpenPrice
        real HighPrice
        real LowPrice
        real ClosePrice
        int Volume
        real DailyReturn
        real Volatility
    }
```

### 1.2 Snowflake Schema Architecture
Normalizes `Dim_Stock` into `Dim_Stock_Snowflake` and `Dim_Sector` (`SectorKey` FK) to eliminate sector redundancy.

### 1.3 Fact Constellation (Galaxy) Schema Architecture
Two independent fact tables (`Fact_DailyTrades` for daily market microstructure and `Fact_QuarterlyFinancials` for corporate earnings fundamentals) sharing conformed dimensions (`Dim_Time`, `Dim_Stock`, `Dim_Exchange`).

### 1.4 Populated Data Warehouse Table Counts
"""
        + df_to_markdown_table(table_summary_df)
        + "\n\n---\n\n## Part 2 — Experiment 02: OLAP Cube Operations & Query Outputs\n"
    )

    for (title, desc), sql_query in zip(olap_titles, raw_queries):
        print(f"\n=== {title} ===")
        res_df = pd.read_sql_query(sql_query, conn)
        print(res_df.to_string(index=False))

        # Strip leading SQL comment lines for clean display in markdown
        sql_lines = [
            line for line in sql_query.splitlines() if not line.strip().startswith("--")
        ]
        clean_sql = "\n".join(sql_lines).strip()

        md_sections.append(
            f"### {title}\n"
            f"**Objective:** {desc}\n\n"
            f"```sql\n{clean_sql};\n```\n\n"
            f"**Query Output ({len(res_df)} rows):**\n\n"
            f"{df_to_markdown_table(res_df)}\n"
        )

    conn.close()

    # Remove temporary sqlite db file after verification so git workspace stays clean
    if db_path.exists():
        db_path.unlink()

    report_path = Path("DWM_Lab_1_2_Stock_Market.md")
    report_path.write_text("\n".join(md_sections), encoding="utf-8")
    print(f"\nSuccessfully exported lab report to {report_path}")


if __name__ == "__main__":
    run_experiments_1_and_2()
