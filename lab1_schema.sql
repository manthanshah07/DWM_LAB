-- ============================================================================
-- DATA WAREHOUSING AND MINING (DWM) — EXPERIMENT 01
-- Experiment Title: Build Data Warehouse/Data Mart & Design Dimensional Data
--                   Models (Star, Snowflake, Fact Constellation)
-- Dataset: stock_market_data.csv (NSE 20 Large-Cap Stocks across 5 Sectors)
-- Compatible with: PostgreSQL / SQLite
-- ============================================================================

-- Clean up existing tables if re-running
DROP TABLE IF EXISTS Fact_QuarterlyFinancials;
DROP TABLE IF EXISTS Fact_DailyTrades_Snowflake;
DROP TABLE IF EXISTS Fact_DailyTrades;
DROP TABLE IF EXISTS Dim_Stock_Snowflake;
DROP TABLE IF EXISTS Dim_Sector;
DROP TABLE IF EXISTS Dim_Stock;
DROP TABLE IF EXISTS Dim_Exchange;
DROP TABLE IF EXISTS Dim_Time;

-- ============================================================================
-- A. STAR SCHEMA MODEL
-- ============================================================================
-- 1. Dimension Table: Dim_Time
CREATE TABLE Dim_Time (
    TimeKey     INTEGER PRIMARY KEY,  -- YYYYMMDD integer surrogate key
    Date        TEXT NOT NULL UNIQUE, -- YYYY-MM-DD
    Day         INTEGER NOT NULL,
    Month       INTEGER NOT NULL,
    MonthName   TEXT NOT NULL,
    Quarter     TEXT NOT NULL,        -- 'Q1', 'Q2', 'Q3', 'Q4'
    Year        INTEGER NOT NULL,
    DayOfWeek   TEXT NOT NULL
);

-- 2. Dimension Table: Dim_Exchange
CREATE TABLE Dim_Exchange (
    ExchangeKey  INTEGER PRIMARY KEY,
    ExchangeName TEXT NOT NULL UNIQUE,
    Country      TEXT NOT NULL,
    Currency     TEXT NOT NULL
);

-- 3. Dimension Table: Dim_Stock (Denormalized for Star Schema)
CREATE TABLE Dim_Stock (
    StockKey    INTEGER PRIMARY KEY,
    Ticker      TEXT NOT NULL UNIQUE,
    CompanyName TEXT NOT NULL,
    Sector      TEXT NOT NULL,
    Exchange    TEXT NOT NULL
);

-- 4. Central Fact Table: Fact_DailyTrades (Star Schema)
CREATE TABLE Fact_DailyTrades (
    TradeID     INTEGER PRIMARY KEY AUTOINCREMENT,
    TimeKey     INTEGER NOT NULL,
    StockKey    INTEGER NOT NULL,
    ExchangeKey INTEGER NOT NULL,
    OpenPrice   REAL NOT NULL,
    HighPrice   REAL NOT NULL,
    LowPrice    REAL NOT NULL,
    ClosePrice  REAL NOT NULL,
    Volume      INTEGER NOT NULL,
    DailyReturn REAL NOT NULL,
    Volatility  REAL NOT NULL,
    FOREIGN KEY (TimeKey)     REFERENCES Dim_Time(TimeKey),
    FOREIGN KEY (StockKey)    REFERENCES Dim_Stock(StockKey),
    FOREIGN KEY (ExchangeKey) REFERENCES Dim_Exchange(ExchangeKey)
);

-- ============================================================================
-- B. SNOWFLAKE SCHEMA MODEL
--    Normalizes Dim_Stock by decomposing Sector hierarchy into Dim_Sector
-- ============================================================================
-- 1. Normalized Sub-Dimension Table: Dim_Sector
CREATE TABLE Dim_Sector (
    SectorKey        INTEGER PRIMARY KEY,
    SectorName       TEXT NOT NULL UNIQUE,
    IndustryCategory TEXT NOT NULL
);

-- 2. Normalized Dimension Table: Dim_Stock_Snowflake
CREATE TABLE Dim_Stock_Snowflake (
    StockKey    INTEGER PRIMARY KEY,
    Ticker      TEXT NOT NULL UNIQUE,
    CompanyName TEXT NOT NULL,
    SectorKey   INTEGER NOT NULL,
    ExchangeKey INTEGER NOT NULL,
    FOREIGN KEY (SectorKey)   REFERENCES Dim_Sector(SectorKey),
    FOREIGN KEY (ExchangeKey) REFERENCES Dim_Exchange(ExchangeKey)
);

-- 3. Snowflake Fact Table: Fact_DailyTrades_Snowflake
CREATE TABLE Fact_DailyTrades_Snowflake (
    TradeID     INTEGER PRIMARY KEY AUTOINCREMENT,
    TimeKey     INTEGER NOT NULL,
    StockKey    INTEGER NOT NULL,
    ExchangeKey INTEGER NOT NULL,
    OpenPrice   REAL NOT NULL,
    HighPrice   REAL NOT NULL,
    LowPrice    REAL NOT NULL,
    ClosePrice  REAL NOT NULL,
    Volume      INTEGER NOT NULL,
    DailyReturn REAL NOT NULL,
    Volatility  REAL NOT NULL,
    FOREIGN KEY (TimeKey)     REFERENCES Dim_Time(TimeKey),
    FOREIGN KEY (StockKey)    REFERENCES Dim_Stock_Snowflake(StockKey),
    FOREIGN KEY (ExchangeKey) REFERENCES Dim_Exchange(ExchangeKey)
);

-- ============================================================================
-- C. FACT CONSTELLATION SCHEMA (GALAXY SCHEMA)
--    Multiple Fact Tables (Fact_DailyTrades & Fact_QuarterlyFinancials)
--    sharing conformed dimensions (Dim_Time, Dim_Stock, Dim_Exchange)
-- ============================================================================
CREATE TABLE Fact_QuarterlyFinancials (
    FinancialID      INTEGER PRIMARY KEY AUTOINCREMENT,
    TimeKey          INTEGER NOT NULL, -- Quarter-end TimeKey from Dim_Time
    StockKey         INTEGER NOT NULL,
    ExchangeKey      INTEGER NOT NULL,
    FiscalYear       INTEGER NOT NULL,
    FiscalQuarter    TEXT NOT NULL,
    QuarterlyRevenue REAL NOT NULL,    -- In INR Crores
    NetProfit        REAL NOT NULL,    -- In INR Crores
    EPS              REAL NOT NULL,    -- Earnings Per Share (INR)
    OperatingMargin  REAL NOT NULL,    -- Operating Margin (%)
    FOREIGN KEY (TimeKey)     REFERENCES Dim_Time(TimeKey),
    FOREIGN KEY (StockKey)    REFERENCES Dim_Stock(StockKey),
    FOREIGN KEY (ExchangeKey) REFERENCES Dim_Exchange(ExchangeKey)
);
