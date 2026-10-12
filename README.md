# Stock Market Analysis System - DWM Lab & Mini Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dwmlab-hrhimxnpqpgyhkvdgg3hne.streamlit.app/)

**Course:** Data Warehousing & Mining (`PCIT10T` / `PCIT10P`) — Semester V  
**Dataset:** `stock_market_data.csv` (3 Years of Daily NSE Trading Data across 20 Major Indian Stocks & 5 Sectors)

---

## 🌐 Live Demo

The **Stock Market Analysis System** is hosted and accessible online for grading and interactive review:

👉 **[Launch Live Streamlit Dashboard: https://dwmlab-hrhimxnpqpgyhkvdgg3hne.streamlit.app/](https://dwmlab-hrhimxnpqpgyhkvdgg3hne.streamlit.app/)**

---

## 1. Project Overview

The **Stock Market Analysis System** is a complete end-to-end **Data Warehousing and Mining (DWM)** laboratory suite and interactive decision-support dashboard built on 3 years of real-world National Stock Exchange (NSE) daily trading data (`14,800` records across `740` trading sessions). It models a multi-dimensional equity data warehouse and implements OLAP cube operations, exploratory visual analytics, supervised price regression, directional return classification, unsupervised asset segmentation, and cross-stock co-movement association rule mining.

### Tracked Universe (20 Nifty 50 Constituents across 5 Sectors)
- **IT:** `TCS`, `INFY`, `WIPRO`, `HCLTECH`
- **Banking:** `HDFCBANK`, `ICICIBANK`, `SBIN`, `KOTAKBANK`
- **Auto:** `TATAMOTORS`, `MARUTI`, `M&M`, `BAJAJ-AUTO`
- **Energy:** `RELIANCE`, `ONGC`, `NTPC`, `POWERGRID`
- **FMCG:** `ITC`, `HINDUNILVR`, `NESTLEIND`, `BRITANNIA`

---

## 2. Lab-by-Lab Curriculum Mapping (Experiments 01 – 10)

| Exp No. | Experiment Title | Implementation Artifact | Key Methodology & Highlights |
| :---: | :--- | :--- | :--- |
| **Exp 01** | **Data Warehouse Star, Snowflake & Fact Constellation Schema** | `lab1_schema.sql`, `run_labs_1_2.py`, `DWM_Lab_1_2_Stock_Market.md` | Full SQL DDL for **Star Schema** (`Fact_DailyTrades`, `Dim_Time`, `Dim_Stock`, `Dim_Exchange`), **Snowflake Schema** (`Dim_Sector`, `Dim_Stock_Snowflake`), and **Fact Constellation / Galaxy Schema** (`Fact_DailyTrades` + `Fact_QuarterlyFinancials`). |
| **Exp 02** | **Multidimensional OLAP Operations** | `lab2_olap.sql`, `run_labs_1_2.py`, `DWM_Lab_1_2_Stock_Market.md`, `app.py` | SQL & Streamlit execution of all 6 OLAP operations: **Slice**, **Dice**, **Roll-Up**, **Drill-Down**, **Pivot (Cross-Tabulation)**, and **Drill-Across** (spanning two fact tables). |
| **Exp 03** | **Data Preprocessing, Exploration & Visualization** | `DWM_Lab_3_Stock_Market.ipynb` (`.html`) | Zero-null verification, temporal feature extraction, `Intraday_Volatility` engineering, sector liquidity bar charts, return/volatility boxplots, and 7-feature correlation heatmap. |
| **Exp 04** | **Simple & Multiple Linear Regression** | `DWM_Lab_4_Stock_Market.ipynb` (`.html`) | Predicts `Close_Price` using `Open_Price` (Simple LR) vs. `['Open_Price', 'High_Price', 'Low_Price', 'Volume']` (Multiple LR) on an 80:20 train-test split. |
| **Exp 05** | **Decision Tree Classification & Mutual Information** | `DWM_Lab_5_Stock_Market.ipynb` (`.html`) | Discretizes `Daily_Return` into balanced tertiles (`Low`, `Medium`, `High`), trains an Entropy-based `DecisionTreeClassifier(max_depth=5)`, and computes `mutual_info_classif`. |
| **Exp 06** | **Gaussian Naive Bayes Classification** | `DWM_Lab_6_Stock_Market.ipynb` (`.html`) | Implements `GaussianNB()` on a 70:30 stratified split to estimate class-conditional likelihoods and posterior probabilities $P(C \mid \mathbf{X})$. |
| **Exp 07** | **K-Means Clustering for Stock Segmentation** | `DWM_Lab_7_Stock_Market.ipynb` (`.html`) | Aggregates 14,800 rows into 20 stock risk-return-liquidity profiles, applies `np.log1p` + `StandardScaler`, determines optimal $K=3$ via Elbow WCSS, and profiles asset personas. |
| **Exp 08** | **Agglomerative Hierarchical Clustering** | `DWM_Lab_8_Stock_Market.ipynb` (`.html`) | Generates dendrograms and compares Silhouette scores across **Single**, **Complete**, **Average**, and **Ward** linkages (`n_clusters=3`). |
| **Exp 09** | **Apriori Association Rule Mining** | `DWM_Lab_9_Stock_Market.ipynb` (`.html`) | Transforms the $740 \times 20$ daily return matrix into a binary bullish market basket (`Daily_Return > 0`) to mine frequent itemsets and high-lift co-movement rules. |
| **Exp 10** | **Integrated Streamlit Mini Project Dashboard** | `app.py` | Production-ready 6-tab Streamlit web application unifying OLAP, EDA, Regression, Classification, Clustering, and Apriori Rule Mining. |

---

## 3. Data Warehouse Schemas & SQL OLAP Operations (Experiments 01 & 02)

- **Dimensional Schemas (`lab1_schema.sql`):**
  1. **Star Schema:** `Fact_DailyTrades` (`14,800` rows) joined directly to `Dim_Time` (`740` rows), `Dim_Stock` (`20` rows), and `Dim_Exchange` (`1` row).
  2. **Snowflake Schema:** Normalizes `Dim_Stock` into `Dim_Stock_Snowflake` (`20` rows) and `Dim_Sector` (`5` rows) linked to `Fact_DailyTrades_Snowflake` (`14,800` rows).
  3. **Fact Constellation (Galaxy) Schema:** Pairs `Fact_DailyTrades` (`14,800` rows) with `Fact_QuarterlyFinancials` (`260` quarterly corporate earnings records) across shared conformed dimensions (`Dim_Time`, `Dim_Stock`, `Dim_Exchange`).
- **OLAP Queries (`lab2_olap.sql` & `DWM_Lab_1_2_Stock_Market.md`):**
  - **Slice:** Isolates January 2024 stock performance (`WHERE t.Year = 2024 AND t.MonthName = 'January'`).
  - **Dice:** Extracts multi-dimensional sub-cube (`Sector IN ('IT', 'Banking') AND Year = 2024 AND Quarter = 'Q1' AND ExchangeName = 'NSE'`).
  - **Roll-Up:** Aggregates daily trading volume and returns up the `Day -> Month -> Year` hierarchy by Sector.
  - **Drill-Down:** Navigates down the Snowflake hierarchy (`Dim_Sector -> Dim_Stock_Snowflake -> Fact_DailyTrades_Snowflake`) to inspect quarterly ticker performance in the `Energy` sector.
  - **Pivot:** Rotates `Sector` (rows) against `Q1`, `Q2`, `Q3`, `Q4` (columns) using conditional SQL `CASE` aggregation.
  - **Drill-Across:** Joins `Fact_DailyTrades` and `Fact_QuarterlyFinancials` via CTEs on shared `StockKey` and `Year` to compare annual traded volume against corporate Net Profit and EPS.

---

## 4. Key Empirical Performance Metrics

- **Linear Regression (Exp 04):**
  - **Simple Linear Regression (`Open_Price` $\rightarrow$ `Close_Price`):** $R^2 = 0.999706$, $\text{MSE} = 2960.9407$ ($\text{RMSE} = ₹54.41$)
  - **Multiple Linear Regression (`Open, High, Low, Volume` $\rightarrow$ `Close_Price`):** $R^2 = 0.999951$, $\text{MSE} = 496.1953$ ($\text{RMSE} = ₹22.28$) — achieving an **$83.24\%$ reduction in MSE**.
- **Classification & Attribute Relevance (Exp 05 & Exp 06):**
  - **Decision Tree (`criterion='entropy'`, `max_depth=5`):** **$48.38\%$ Test Accuracy** (`0.6284` F1-score on `Medium` regime vs. $33.33\%$ 3-class random baseline).
  - **Feature Importance & Mutual Information:** `Intraday_Volatility` accounts for **$95.09\%$** of Decision Tree Information Gain and achieves the highest Mutual Information score ($\text{MI} = 0.137970$).
  - **Gaussian Naive Bayes (70:30 split):** **$35.61\%$ Overall Accuracy** with **$87.86\%$ Recall** on the `Medium` (Neutral) return class.
- **Unsupervised Asset Clustering (Exp 07 & Exp 08):**
  - **K-Means ($K=3$):** WCSS drops from $100.00$ ($K=1$) to $34.2836$ ($K=3$) with a **Silhouette Score of $0.4042$**, segmenting stocks into:
    1. *Blue-Chip Banking & Core Liquidity Leaders (`7` stocks):* `HDFCBANK`, `ICICIBANK`, `INFY`, `ITC`, `KOTAKBANK`, `RELIANCE`, `SBIN`
    2. *High-Denomination Defensive & Quality (`7` stocks):* `BAJAJ-AUTO`, `BRITANNIA`, `HCLTECH`, `HINDUNILVR`, `MARUTI`, `NESTLEIND`, `TCS`
    3. *High-Volatility Cyclicals & Energy PSUs (`6` stocks):* `M&M`, `NTPC`, `ONGC`, `POWERGRID`, `TATAMOTORS`, `WIPRO`
  - **Hierarchical Clustering Linkages ($K=3$):** **Average Linkage** Silhouette = **$0.4042$**, **Ward Linkage** = **$0.3922$**, **Complete Linkage** = **$0.3922$**, **Single Linkage** = **$-0.1060$** (chaining effect).
- **Apriori Co-Movement Mining (Exp 09):**
  - **Frequent Itemsets (`min_support = 0.20`):** **`229` frequent itemsets** (`20` 1-itemsets, `190` 2-itemsets, `18` 3-itemsets, `1` 4-itemset `{HCLTECH, INFY, TCS, WIPRO}`).
  - **Association Rules (`min_confidence = 0.50`):** **`404` total rules** and **`237` strong rules** ($\text{Confidence} \ge 0.55$, $\text{Lift} > 1.10$).
  - **Top Co-Movement Rules:** `INFY, WIPRO → HCLTECH, TCS` ($\text{Support} = 0.2081$, $\text{Confidence} = 0.7404$, $\text{Lift} = 2.4030$) and `INFY, TCS, WIPRO → HCLTECH` ($\text{Confidence} = 0.9006$, $\text{Lift} = 1.9095$).

---

## 5. Setup & Execution Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/manthanshah07/DWM_LAB.git
cd DWM_LAB
```

### 2. Create Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Execute Data Warehouse DDL & OLAP Queries (Experiments 01 & 02)
```bash
python run_labs_1_2.py
```

### 4. (Optional) Re-generate the 3-Year NSE Stock Dataset
```bash
python generate_data.py
```

### 5. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app.py
```

---

## 6. Repository Structure

```text
DWM_LAB/
├── app.py                          # Exp 10: Unified 6-Tab Interactive Streamlit Dashboard
├── generate_data.py                # Automated yfinance NSE Data Ingestion Pipeline
├── stock_market_data.csv           # 3-Year Daily Trading Dataset (14,800 rows x 10 columns)
├── lab1_schema.sql                 # Exp 01: SQL DDL for Star, Snowflake & Fact Constellation Schemas
├── lab2_olap.sql                   # Exp 02: SQL OLAP Queries (Slice, Dice, Roll-Up, Drill-Down, Pivot, Drill-Across)
├── run_labs_1_2.py                 # Automated Relational Warehouse Loader & OLAP Query Runner
├── DWM_Lab_1_2_Stock_Market.md     # Exp 01 & 02: Executed Schema & OLAP Output Report
├── DWM_Lab_3_Stock_Market.ipynb    # Exp 03: Data Preprocessing, EDA & Visualization (.ipynb & .html)
├── DWM_Lab_4_Stock_Market.ipynb    # Exp 04: Simple & Multiple Linear Regression (.ipynb & .html)
├── DWM_Lab_5_Stock_Market.ipynb    # Exp 05: Decision Tree Classification & Mutual Info (.ipynb & .html)
├── DWM_Lab_6_Stock_Market.ipynb    # Exp 06: Gaussian Naive Bayes Classification (.ipynb & .html)
├── DWM_Lab_7_Stock_Market.ipynb    # Exp 07: K-Means Asset Clustering (.ipynb & .html)
├── DWM_Lab_8_Stock_Market.ipynb    # Exp 08: Agglomerative Hierarchical Clustering (.ipynb & .html)
├── DWM_Lab_9_Stock_Market.ipynb    # Exp 09: Apriori Association Rule Mining (.ipynb & .html)
├── requirements.txt                # Python Package Dependencies
├── .gitignore                      # Git Ignore Rules
└── README.md                       # Project Documentation
```
