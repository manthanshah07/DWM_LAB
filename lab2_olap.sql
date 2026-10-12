-- ============================================================================
-- DATA WAREHOUSING AND MINING (DWM) — EXPERIMENT 02
-- Experiment Title: Perform OLAP Operations on the Dimensional Model
--                   (Slice, Dice, Roll-Up, Drill-Down, Pivot, Drill-Across)
-- Dataset: stock_market_data.csv (Loaded into Star & Fact Constellation Schema)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. SLICE OPERATION
--    Isolate trading performance along a single dimensional slice:
--    Year = 2024 AND MonthName = 'January'
-- ----------------------------------------------------------------------------
SELECT
    s.Ticker,
    s.Sector,
    COUNT(f.TradeID)           AS Trading_Days_Jan2024,
    ROUND(AVG(f.OpenPrice), 2) AS Avg_Open_Price,
    ROUND(AVG(f.ClosePrice), 2) AS Avg_Close_Price,
    SUM(f.Volume)              AS Total_Jan2024_Volume,
    ROUND(AVG(f.DailyReturn), 3) AS Avg_Daily_Return_Pct
FROM Fact_DailyTrades f
JOIN Dim_Time t  ON f.TimeKey  = t.TimeKey
JOIN Dim_Stock s ON f.StockKey = s.StockKey
WHERE t.Year = 2024
  AND t.MonthName = 'January'
GROUP BY s.Ticker, s.Sector
ORDER BY Total_Jan2024_Volume DESC
LIMIT 10;

-- ----------------------------------------------------------------------------
-- 2. DICE OPERATION
--    Extract a multi-dimensional sub-cube by filtering across multiple
--    dimensions simultaneously:
--    Sector IN ('IT', 'Banking') AND Year = 2024 AND Quarter = 'Q1'
-- ----------------------------------------------------------------------------
SELECT
    s.Sector,
    s.Ticker,
    t.Year,
    t.Quarter,
    COUNT(f.TradeID)             AS Trading_Sessions,
    ROUND(AVG(f.ClosePrice), 2)  AS Avg_Close_Price,
    SUM(f.Volume)                AS SubCube_Total_Volume,
    ROUND(AVG(f.DailyReturn), 3) AS Avg_Daily_Return_Pct,
    ROUND(AVG(f.Volatility), 2)  AS Avg_Intraday_Volatility_Pct
FROM Fact_DailyTrades f
JOIN Dim_Time t     ON f.TimeKey     = t.TimeKey
JOIN Dim_Stock s    ON f.StockKey    = s.StockKey
JOIN Dim_Exchange e ON f.ExchangeKey = e.ExchangeKey
WHERE s.Sector IN ('IT', 'Banking')
  AND t.Year = 2024
  AND t.Quarter = 'Q1'
  AND e.ExchangeName = 'NSE'
GROUP BY s.Sector, s.Ticker, t.Year, t.Quarter
ORDER BY s.Sector, SubCube_Total_Volume DESC;

-- ----------------------------------------------------------------------------
-- 3. ROLL-UP OPERATION
--    Aggregate granular daily trading facts up the concept hierarchy:
--    Day -> Month -> Year & Sector summary
-- ----------------------------------------------------------------------------
SELECT
    t.Year,
    t.Month,
    t.MonthName,
    s.Sector,
    COUNT(f.TradeID)             AS Total_Stock_Days,
    SUM(f.Volume)                AS Monthly_Sector_Volume,
    ROUND(AVG(f.ClosePrice), 2)  AS Avg_Sector_Close_Price,
    ROUND(AVG(f.DailyReturn), 3) AS Avg_Daily_Return_Pct
FROM Fact_DailyTrades f
JOIN Dim_Time t  ON f.TimeKey  = t.TimeKey
JOIN Dim_Stock s ON f.StockKey = s.StockKey
WHERE t.Year = 2025
GROUP BY t.Year, t.Month, t.MonthName, s.Sector
ORDER BY t.Year, t.Month, Monthly_Sector_Volume DESC
LIMIT 15;

-- ----------------------------------------------------------------------------
-- 4. DRILL-DOWN OPERATION
--    Navigate from high-level Sector summary down to individual Stock Tickers
--    and Quarterly breakdown (using Snowflake schema tables Dim_Sector &
--    Dim_Stock_Snowflake)
-- ----------------------------------------------------------------------------
SELECT
    sec.SectorName,
    sec.IndustryCategory,
    st.Ticker,
    t.Year,
    t.Quarter,
    COUNT(fs.TradeID)             AS Sessions,
    ROUND(AVG(fs.ClosePrice), 2)  AS Quarterly_Avg_Close,
    SUM(fs.Volume)                AS Quarterly_Volume,
    ROUND(AVG(fs.Volatility), 2)  AS Avg_Volatility_Pct
FROM Fact_DailyTrades_Snowflake fs
JOIN Dim_Stock_Snowflake st ON fs.StockKey  = st.StockKey
JOIN Dim_Sector sec         ON st.SectorKey = sec.SectorKey
JOIN Dim_Time t             ON fs.TimeKey   = t.TimeKey
WHERE sec.SectorName = 'Energy'
  AND t.Year = 2025
GROUP BY sec.SectorName, sec.IndustryCategory, st.Ticker, t.Year, t.Quarter
ORDER BY st.Ticker, t.Quarter;

-- ----------------------------------------------------------------------------
-- 5. PIVOT (CROSS-TABULATION) OPERATION
--    Rotate dimensional axes to display Sectors along rows and Quarters
--    ('Q1', 'Q2', 'Q3', 'Q4') across columns for Average Daily Return (%)
--    and Total Traded Volume (in Millions)
-- ----------------------------------------------------------------------------
SELECT
    s.Sector,
    ROUND(AVG(CASE WHEN t.Quarter = 'Q1' THEN f.DailyReturn END), 4) AS Q1_Avg_Return_Pct,
    ROUND(AVG(CASE WHEN t.Quarter = 'Q2' THEN f.DailyReturn END), 4) AS Q2_Avg_Return_Pct,
    ROUND(AVG(CASE WHEN t.Quarter = 'Q3' THEN f.DailyReturn END), 4) AS Q3_Avg_Return_Pct,
    ROUND(AVG(CASE WHEN t.Quarter = 'Q4' THEN f.DailyReturn END), 4) AS Q4_Avg_Return_Pct,
    ROUND(SUM(CASE WHEN t.Quarter = 'Q1' THEN f.Volume ELSE 0 END) / 1e6, 2) AS Q1_Vol_Millions,
    ROUND(SUM(CASE WHEN t.Quarter = 'Q2' THEN f.Volume ELSE 0 END) / 1e6, 2) AS Q2_Vol_Millions,
    ROUND(SUM(CASE WHEN t.Quarter = 'Q3' THEN f.Volume ELSE 0 END) / 1e6, 2) AS Q3_Vol_Millions,
    ROUND(SUM(CASE WHEN t.Quarter = 'Q4' THEN f.Volume ELSE 0 END) / 1e6, 2) AS Q4_Vol_Millions
FROM Fact_DailyTrades f
JOIN Dim_Time t  ON f.TimeKey  = t.TimeKey
JOIN Dim_Stock s ON f.StockKey = s.StockKey
GROUP BY s.Sector
ORDER BY s.Sector;

-- ----------------------------------------------------------------------------
-- 6. DRILL-ACROSS OPERATION
--    Query across two distinct Fact Tables (Fact_DailyTrades and
--    Fact_QuarterlyFinancials) linked via conformed dimensions (Dim_Stock,
--    Dim_Time) to compare annual market trading activity against corporate
--    financial fundamentals (Revenue, Net Profit, EPS, Operating Margin)
-- ----------------------------------------------------------------------------
WITH Market_Annual AS (
    SELECT
        f.StockKey,
        t.Year AS TradeYear,
        SUM(f.Volume)                AS Annual_Traded_Volume,
        ROUND(AVG(f.ClosePrice), 2)  AS Annual_Avg_Share_Price,
        ROUND(AVG(f.Volatility), 2)  AS Annual_Intraday_Volatility
    FROM Fact_DailyTrades f
    JOIN Dim_Time t ON f.TimeKey = t.TimeKey
    WHERE t.Year = 2025
    GROUP BY f.StockKey, t.Year
),
Financials_Annual AS (
    SELECT
        qf.StockKey,
        qf.FiscalYear,
        ROUND(SUM(qf.QuarterlyRevenue), 2) AS Annual_Revenue_Cr,
        ROUND(SUM(qf.NetProfit), 2)        AS Annual_NetProfit_Cr,
        ROUND(AVG(qf.EPS), 2)              AS Avg_Quarterly_EPS,
        ROUND(AVG(qf.OperatingMargin), 2)  AS Avg_Operating_Margin_Pct
    FROM Fact_QuarterlyFinancials qf
    WHERE qf.FiscalYear = 2025
    GROUP BY qf.StockKey, qf.FiscalYear
)
SELECT
    s.Ticker,
    s.CompanyName,
    s.Sector,
    m.TradeYear                    AS Year,
    m.Annual_Avg_Share_Price       AS Avg_Price_INR,
    m.Annual_Traded_Volume         AS Total_Volume,
    fin.Annual_Revenue_Cr          AS Revenue_Cr,
    fin.Annual_NetProfit_Cr        AS NetProfit_Cr,
    fin.Avg_Quarterly_EPS          AS Avg_EPS_INR,
    fin.Avg_Operating_Margin_Pct   AS Op_Margin_Pct
FROM Market_Annual m
JOIN Financials_Annual fin ON m.StockKey = fin.StockKey AND m.TradeYear = fin.FiscalYear
JOIN Dim_Stock s           ON m.StockKey = s.StockKey
ORDER BY fin.Annual_NetProfit_Cr DESC
LIMIT 12;
