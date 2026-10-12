import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
)
from sklearn.feature_selection import mutual_info_classif
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------------------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Market Analysis System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(28, 131, 225, 0.06);
        border: 1px solid rgba(28, 131, 225, 0.18);
        padding: 14px 18px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# 2. Cached Data Loader & Pre-Calculations
# ---------------------------------------------------------------------------
@st.cache_data
def load_and_prepare_data(csv_path: str = "stock_market_data.csv"):
    df = pd.read_csv(csv_path)
    df["Date_dt"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date_dt"].dt.year
    df["Quarter"] = "Q" + df["Date_dt"].dt.quarter.astype(str)
    df["Year_Quarter"] = df["Year"].astype(str) + "-" + df["Quarter"]
    df["Month"] = df["Date_dt"].dt.month
    df["MonthName"] = df["Date_dt"].dt.month_name()
    df["YearMonth"] = df["Date_dt"].dt.to_period("M").astype(str)
    df["DayOfWeek"] = df["Date_dt"].dt.day_name()

    # Derived Intraday Volatility (%)
    df["Intraday_Volatility"] = (
        ((df["High_Price"] - df["Low_Price"]) / df["Open_Price"]) * 100
    ).round(2)

    # Quantile tertiles for Return_Category (Low, Medium, High)
    q1 = df["Daily_Return"].quantile(1 / 3)
    q2 = df["Daily_Return"].quantile(2 / 3)
    df["Return_Category"] = pd.cut(
        df["Daily_Return"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    # Stock-level summary for clustering (Labs 7 & 8)
    stock_df = (
        df.groupby(["Ticker", "Sector"], as_index=False)
        .agg(
            Avg_Daily_Return=("Daily_Return", "mean"),
            Annualized_Volatility=("Daily_Return", lambda x: x.std() * np.sqrt(252)),
            Avg_Volume=("Volume", "mean"),
            Avg_Intraday_Volatility=("Intraday_Volatility", "mean"),
            Total_Traded_Volume=("Volume", "sum"),
            Avg_Close_Price=("Close_Price", "mean"),
        )
        .sort_values("Ticker")
        .reset_index(drop=True)
    )
    stock_df["Avg_Daily_Return"] = stock_df["Avg_Daily_Return"].round(4)
    stock_df["Annualized_Volatility"] = stock_df["Annualized_Volatility"].round(2)
    stock_df["Avg_Volume"] = stock_df["Avg_Volume"].round(0).astype(int)
    stock_df["Avg_Intraday_Volatility"] = stock_df["Avg_Intraday_Volatility"].round(2)
    stock_df["Avg_Close_Price"] = stock_df["Avg_Close_Price"].round(2)
    stock_df["Avg_Volume_Log"] = np.log1p(stock_df["Avg_Volume"])
    stock_df["Total_Traded_Volume_Log"] = np.log1p(stock_df["Total_Traded_Volume"])

    # Binary Market Basket for Apriori (Lab 9)
    df["Bullish"] = (df["Daily_Return"] > 0).astype(int)
    basket_bool = (
        df.pivot_table(index="Date", columns="Ticker", values="Bullish", fill_value=0)
        .astype(bool)
    )

    return df, stock_df, basket_bool, float(q1), float(q2)


@st.cache_resource
def train_predictive_models(df: pd.DataFrame):
    # 1. Simple & Multiple Linear Regression (Lab 4)
    X_s = df[["Open_Price"]]
    y_reg = df["Close_Price"]
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_s, y_reg, test_size=0.2, random_state=42
    )
    slr = LinearRegression().fit(X_train_s, y_train_s)
    y_pred_s = slr.predict(X_test_s)
    r2_s = r2_score(y_test_s, y_pred_s)
    mse_s = mean_squared_error(y_test_s, y_pred_s)

    reg_features = ["Open_Price", "High_Price", "Low_Price", "Volume"]
    X_m = df[reg_features]
    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_m, y_reg, test_size=0.2, random_state=42
    )
    mlr = LinearRegression().fit(X_train_m, y_train_m)
    y_pred_m = mlr.predict(X_test_m)
    r2_m = r2_score(y_test_m, y_pred_m)
    mse_m = mean_squared_error(y_test_m, y_pred_m)

    # 2. Decision Tree (Lab 5 - 80:20 split) & Gaussian Naive Bayes (Lab 6 - 70:30 split)
    clf_features = ["Open_Price", "Volume", "Intraday_Volatility", "Month"]
    X_clf = df[clf_features]
    y_clf = df["Return_Category"]

    X_tr_dt, X_te_dt, y_tr_dt, y_te_dt = train_test_split(
        X_clf, y_clf, test_size=0.20, stratify=y_clf, random_state=42
    )
    dt_model = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42)
    dt_model.fit(X_tr_dt, y_tr_dt)
    y_pr_dt = dt_model.predict(X_te_dt)

    X_tr_nb, X_te_nb, y_tr_nb, y_te_nb = train_test_split(
        X_clf, y_clf, test_size=0.30, stratify=y_clf, random_state=42
    )
    nb_model = GaussianNB()
    nb_model.fit(X_tr_nb, y_tr_nb)
    y_pr_nb = nb_model.predict(X_te_nb)

    clf_metrics = pd.DataFrame(
        {
            "Model": [
                "Decision Tree (Entropy, max_depth=5)",
                "Gaussian Naive Bayes (GaussianNB)",
            ],
            "Train:Test Split": ["80:20 Stratified", "70:30 Stratified"],
            "Accuracy (%)": [
                round(accuracy_score(y_te_dt, y_pr_dt) * 100, 2),
                round(accuracy_score(y_te_nb, y_pr_nb) * 100, 2),
            ],
            "Weighted Precision": [
                round(precision_score(y_te_dt, y_pr_dt, average="weighted", zero_division=0), 4),
                round(precision_score(y_te_nb, y_pr_nb, average="weighted", zero_division=0), 4),
            ],
            "Weighted Recall": [
                round(recall_score(y_te_dt, y_pr_dt, average="weighted", zero_division=0), 4),
                round(recall_score(y_te_nb, y_pr_nb, average="weighted", zero_division=0), 4),
            ],
            "Weighted F1-Score": [
                round(f1_score(y_te_dt, y_pr_dt, average="weighted", zero_division=0), 4),
                round(f1_score(y_te_nb, y_pr_nb, average="weighted", zero_division=0), 4),
            ],
        }
    )

    mi_scores = mutual_info_classif(X_clf, y_clf, random_state=42)
    feature_eval_df = pd.DataFrame(
        {
            "Feature": clf_features,
            "DT_Feature_Importance": np.round(dt_model.feature_importances_, 4),
            "Mutual_Information": np.round(mi_scores, 4),
        }
    ).sort_values("DT_Feature_Importance", ascending=False)

    return {
        "slr": slr,
        "mlr": mlr,
        "r2_s": r2_s,
        "mse_s": mse_s,
        "r2_m": r2_m,
        "mse_m": mse_m,
        "dt_model": dt_model,
        "nb_model": nb_model,
        "clf_metrics": clf_metrics,
        "feature_eval_df": feature_eval_df,
    }


@st.cache_data
def compute_clustering_artifacts(stock_df: pd.DataFrame):
    cluster_cols = [
        "Avg_Daily_Return",
        "Annualized_Volatility",
        "Avg_Volume_Log",
        "Avg_Intraday_Volatility",
        "Total_Traded_Volume_Log",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(stock_df[cluster_cols])

    # Elbow curve (K = 1..10)
    elbow_rows = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        elbow_rows.append({"K": k, "WCSS": round(km.inertia_, 4)})
    elbow_df = pd.DataFrame(elbow_rows)

    # K=3 K-Means
    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    clustered_df = stock_df.copy()
    clustered_df["KMeans_Cluster"] = km3.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, clustered_df["KMeans_Cluster"])

    persona_map = {}
    for c_id, grp in clustered_df.groupby("KMeans_Cluster"):
        if grp["Annualized_Volatility"].mean() < 19.5:
            persona_map[c_id] = "Blue-Chip Banking & Core Liquidity Leaders"
        elif grp["Avg_Volume"].mean() < 5_000_000:
            persona_map[c_id] = "FMCG / Auto / IT Stability (High Denomination)"
        else:
            persona_map[c_id] = "High Volatility Cyclicals & Energy PSUs"

    clustered_df["Persona"] = clustered_df["KMeans_Cluster"].map(persona_map)

    # Hierarchical Linkages (K=3)
    linkage_rows = []
    for method in ["average", "ward", "complete", "single"]:
        agg = AgglomerativeClustering(n_clusters=3, linkage=method, metric="euclidean")
        labels = agg.fit_predict(X_scaled)
        clustered_df[f"Hier_{method.capitalize()}"] = labels
        sil = silhouette_score(X_scaled, labels)
        counts = pd.Series(labels).value_counts().reindex([0, 1, 2], fill_value=0)
        linkage_rows.append(
            {
                "Linkage Method": method.capitalize(),
                "Silhouette Score": round(sil, 4),
                "Cluster Sizes (0 / 1 / 2)": f"{counts[0]} / {counts[1]} / {counts[2]}",
            }
        )
    linkage_df = pd.DataFrame(linkage_rows).sort_values(
        "Silhouette Score", ascending=False
    )

    return clustered_df, X_scaled, elbow_df, km_sil, linkage_df


# Load datasets & models
df, stock_df, basket_bool, q1_val, q2_val = load_and_prepare_data()
models_bundle = train_predictive_models(df)
clustered_df, X_scaled_clust, elbow_df, km_sil, linkage_df = compute_clustering_artifacts(
    stock_df
)

# ---------------------------------------------------------------------------
# 3. Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("📈 DWM Stock System")
st.sidebar.caption("Experiment 10 — Unified DWM Mini Project")

nav_selection = st.sidebar.radio(
    "Select Module:",
    [
        "📊 Overview & Data Warehouse (OLAP)",
        "📈 Exploratory Data Analysis (EDA)",
        "🔮 Predictive Modeling (Regression & Classification)",
        "🎯 Asset Clustering (K-Means & Hierarchical)",
        "🔗 Association Rule Mining (Apriori Co-Movement)",
        "📑 Project Summary & Lab Mapping",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Dataset:** `stock_market_data.csv`\n\n"
    "- **Records:** 14,800 daily rows\n"
    "- **Trading Days:** 740 (3 Years)\n"
    "- **Stocks:** 20 NSE Large-Caps\n"
    "- **Sectors:** IT, Banking, Auto, Energy, FMCG"
)

# ===========================================================================
# TAB 1: Overview & Data Warehouse (OLAP)
# ===========================================================================
if nav_selection == "📊 Overview & Data Warehouse (OLAP)":
    st.title("📊 Stock Market Data Warehouse & Interactive OLAP Engine")
    st.markdown(
        "Multidimensional **Star-Schema Fact & Dimension** analytical workspace supporting interactive "
        "**Roll-Up**, **Drill-Down**, **Slice**, **Dice**, and **Pivot** operations across NSE trading data."
    )

    # Top KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trading Records", f"{len(df):,}", "740 Trading Days")
    c2.metric("Tracked NSE Stocks", f"{df['Ticker'].nunique()}", "4 per Sector")
    c3.metric("Market Sectors", f"{df['Sector'].nunique()}", "IT, Bank, Auto, Energy, FMCG")
    c4.metric(
        "Total Traded Volume",
        f"{df['Volume'].sum() / 1e9:.2f}B",
        f"Avg Return: {df['Daily_Return'].mean():.3f}%",
    )

    st.markdown("---")
    st.subheader("🧊 Interactive OLAP Cube Operations")

    olap_col1, olap_col2, olap_col3 = st.columns([1.1, 1.3, 1.6])

    with olap_col1:
        st.markdown("**1. Slice Operation (Single Dimension Filter)**")
        available_years = ["All Years"] + sorted(df["Year"].unique().astype(str).tolist())
        slice_year = st.selectbox("Slice by Year:", available_years, index=0)

    with olap_col2:
        st.markdown("**2. Dice Operation (Multi-Dimension Sub-Cube)**")
        all_sectors = sorted(df["Sector"].unique().tolist())
        dice_sectors = st.multiselect(
            "Select Sectors:", options=all_sectors, default=all_sectors
        )
        all_exchanges = sorted(df["Exchange"].unique().tolist())
        dice_exchanges = st.multiselect(
            "Select Exchange:", options=all_exchanges, default=all_exchanges
        )

    with olap_col3:
        st.markdown("**3. Roll-Up & Drill-Down Granularity**")
        rollup_granularity = st.radio(
            "Roll-Up Time Hierarchy:",
            ["Yearly Summary", "Quarterly Summary", "Monthly Summary"],
            horizontal=True,
        )
        drilldown_sector = st.selectbox(
            "Drill-Down into Sector (Stock Breakdown):",
            options=all_sectors,
            index=0,
        )

    # Apply Slice & Dice filters
    sub_cube = df.copy()
    if slice_year != "All Years":
        sub_cube = sub_cube[sub_cube["Year"] == int(slice_year)]
    if dice_sectors:
        sub_cube = sub_cube[sub_cube["Sector"].isin(dice_sectors)]
    if dice_exchanges:
        sub_cube = sub_cube[sub_cube["Exchange"].isin(dice_exchanges)]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown(f"#### 🔼 Roll-Up Aggregation ({rollup_granularity})")
        if rollup_granularity == "Yearly Summary":
            time_col = "Year"
        elif rollup_granularity == "Quarterly Summary":
            time_col = "Year_Quarter"
        else:
            time_col = "YearMonth"

        rollup_df = (
            sub_cube.groupby([time_col, "Sector"], as_index=False)
            .agg(
                Total_Volume=("Volume", "sum"),
                Avg_Close_Price=("Close_Price", "mean"),
                Avg_Daily_Return=("Daily_Return", "mean"),
                Avg_Intraday_Volatility=("Intraday_Volatility", "mean"),
            )
            .sort_values([time_col, "Sector"])
        )
        rollup_df["Avg_Close_Price"] = rollup_df["Avg_Close_Price"].round(2)
        rollup_df["Avg_Daily_Return"] = rollup_df["Avg_Daily_Return"].round(3)
        rollup_df["Avg_Intraday_Volatility"] = rollup_df["Avg_Intraday_Volatility"].round(2)

        fig_rollup = px.bar(
            rollup_df,
            x=time_col,
            y="Total_Volume",
            color="Sector",
            barmode="group",
            title=f"Roll-Up: Traded Volume by {time_col} & Sector",
            height=370,
        )
        st.plotly_chart(fig_rollup, use_container_width=True)
        st.dataframe(rollup_df, use_container_width=True, height=220)

    with col_right:
        st.markdown(f"#### 🔽 Drill-Down: `{drilldown_sector}` Sector Constituents")
        drill_df = (
            df[df["Sector"] == drilldown_sector]
            .groupby(["Ticker", "Year"], as_index=False)
            .agg(
                Trading_Days=("Date", "count"),
                Avg_Close_Price=("Close_Price", "mean"),
                Total_Volume=("Volume", "sum"),
                Avg_Daily_Return=("Daily_Return", "mean"),
                Avg_Intraday_Volatility=("Intraday_Volatility", "mean"),
            )
            .sort_values(["Ticker", "Year"])
        )
        drill_df["Avg_Close_Price"] = drill_df["Avg_Close_Price"].round(2)
        drill_df["Avg_Daily_Return"] = drill_df["Avg_Daily_Return"].round(3)
        drill_df["Avg_Intraday_Volatility"] = drill_df["Avg_Intraday_Volatility"].round(2)

        fig_drill = px.line(
            df[df["Sector"] == drilldown_sector],
            x="Date_dt",
            y="Close_Price",
            color="Ticker",
            title=f"Drill-Down: Daily Closing Price Trajectory in {drilldown_sector}",
            height=370,
        )
        st.plotly_chart(fig_drill, use_container_width=True)
        st.dataframe(drill_df, use_container_width=True, height=220)

    st.markdown("#### 🔄 OLAP Pivot Table: Average Daily Return (%) by Sector across Quarters")
    pivot_tbl = pd.pivot_table(
        sub_cube,
        values="Daily_Return",
        index="Sector",
        columns="Year_Quarter",
        aggfunc="mean",
        margins=True,
        margins_name="Overall Mean",
    ).round(3)
    st.dataframe(pivot_tbl, use_container_width=True)

# ===========================================================================
# TAB 2: Exploratory Data Analysis (EDA)
# ===========================================================================
elif nav_selection == "📈 Exploratory Data Analysis (EDA)":
    st.title("📈 Exploratory Data Analysis & Market Visualization (Experiment 03)")
    st.markdown(
        "Multi-dimensional statistical distribution, sector liquidity, temporal trend, and correlation analysis."
    )

    r1c1, r1c2 = st.columns(2)

    with r1c1:
        sec_vol = (
            df.groupby("Sector", as_index=False)["Volume"]
            .sum()
            .sort_values("Volume", ascending=True)
        )
        fig_sec_vol = px.bar(
            sec_vol,
            x="Volume",
            y="Sector",
            color="Sector",
            orientation="h",
            title="Total 3-Year Trading Volume Contribution by Sector",
            height=400,
        )
        st.plotly_chart(fig_sec_vol, use_container_width=True)

    with r1c2:
        top10_tickers = (
            df.groupby(["Ticker", "Sector"], as_index=False)["Volume"]
            .sum()
            .sort_values("Volume", ascending=False)
            .head(10)
            .sort_values("Volume", ascending=True)
        )
        fig_top10 = px.bar(
            top10_tickers,
            x="Volume",
            y="Ticker",
            color="Sector",
            orientation="h",
            title="Top 10 Most Actively Traded NSE Stocks by Volume",
            height=400,
        )
        st.plotly_chart(fig_top10, use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_box_ret = px.box(
            df,
            x="Sector",
            y="Daily_Return",
            color="Sector",
            title="Sector-wise Daily Return (%) Distribution & Fat-Tail Outliers",
            height=420,
        )
        fig_box_ret.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_box_ret, use_container_width=True)

    with r2c2:
        fig_box_vol = px.box(
            df,
            x="Sector",
            y="Intraday_Volatility",
            color="Sector",
            title="Sector-wise Intraday Volatility (%) Distribution",
            height=420,
        )
        st.plotly_chart(fig_box_vol, use_container_width=True)

    st.markdown("---")
    r3c1, r3c2 = st.columns([1.2, 1.0])

    with r3c1:
        alpha_val = st.slider(
            "Adjust Scatter Plot Alpha (Opacity):",
            min_value=0.10,
            max_value=1.00,
            value=0.45,
            step=0.05,
        )
        sampled_eda = df.sample(n=3000, random_state=42)
        fig_scat = px.scatter(
            sampled_eda,
            x="Volume",
            y="Daily_Return",
            color="Sector",
            hover_data=["Ticker", "Date", "Close_Price"],
            log_x=True,
            opacity=alpha_val,
            title="Trading Volume (Log Scale) vs Daily Return (%)",
            height=440,
        )
        fig_scat.add_hline(y=0, line_dash="dash", line_color="black")
        st.plotly_chart(fig_scat, use_container_width=True)

    with r3c2:
        num_cols = [
            "Open_Price",
            "High_Price",
            "Low_Price",
            "Close_Price",
            "Volume",
            "Daily_Return",
            "Intraday_Volatility",
        ]
        corr_mat = df[num_cols].corr().round(2)
        fig_corr = px.imshow(
            corr_mat,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            title="Correlation Matrix Heatmap (7 Numerical Features)",
            height=460,
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ===========================================================================
# TAB 3: Predictive Modeling (Regression & Classification)
# ===========================================================================
elif nav_selection == "🔮 Predictive Modeling (Regression & Classification)":
    st.title("🔮 Supervised Predictive Modeling (Experiments 04, 05 & 06)")

    subtab_reg, subtab_clf = st.tabs(
        [
            "📉 Sub-Tab A: Price Prediction (Simple & Multiple Linear Regression)",
            "🧭 Sub-Tab B: Trend Classification (Decision Tree vs. Naive Bayes)",
        ]
    )

    # ---- SUB-TAB A: REGRESSION ----
    with subtab_reg:
        st.subheader("Simple & Multiple Linear Regression — Predicting `Close_Price` (Experiment 04)")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Simple LR R² Score", f"{models_bundle['r2_s']:.6f}", "Feature: Open_Price")
        m2.metric("Simple LR MSE", f"{models_bundle['mse_s']:.2f}", "RMSE: ₹54.41")
        m3.metric(
            "Multiple LR R² Score",
            f"{models_bundle['r2_m']:.6f}",
            "Features: Open, High, Low, Vol",
        )
        m4.metric(
            "Multiple LR MSE",
            f"{models_bundle['mse_m']:.2f}",
            "-83.24% MSE Reduction",
            delta_color="normal",
        )

        st.markdown("#### 🎛️ Interactive `Close_Price` Predictor (Multiple Linear Regression)")
        in_c1, in_c2, in_c3, in_c4 = st.columns(4)
        with in_c1:
            inp_open = st.number_input("Open Price (₹):", min_value=50.0, max_value=20000.0, value=2500.0, step=10.0)
        with in_c2:
            inp_high = st.number_input("High Price (₹):", min_value=50.0, max_value=20000.0, value=2540.0, step=10.0)
        with in_c3:
            inp_low = st.number_input("Low Price (₹):", min_value=50.0, max_value=20000.0, value=2490.0, step=10.0)
        with in_c4:
            inp_vol = st.number_input("Trading Volume:", min_value=1000, max_value=200000000, value=1500000, step=50000)

        reg_input_df = pd.DataFrame(
            {
                "Open_Price": [inp_open],
                "High_Price": [inp_high],
                "Low_Price": [inp_low],
                "Volume": [inp_vol],
            }
        )
        pred_close_val = float(models_bundle["mlr"].predict(reg_input_df)[0])
        intraday_diff = pred_close_val - inp_open
        intraday_pct = (intraday_diff / inp_open) * 100

        res_c1, res_c2 = st.columns([1, 2])
        with res_c1:
            st.metric(
                "Predicted Close Price (₹)",
                f"₹{pred_close_val:,.2f}",
                f"{intraday_diff:+.2f} ₹ ({intraday_pct:+.2f}% vs Open)",
            )
        with res_c2:
            coef_df = pd.DataFrame(
                {
                    "Feature": ["Intercept (b0)", "Open_Price", "High_Price", "Low_Price", "Volume"],
                    "Coefficient": [
                        round(models_bundle["mlr"].intercept_, 6),
                        *[round(c, 6) for c in models_bundle["mlr"].coef_],
                    ],
                }
            )
            st.dataframe(coef_df, use_container_width=True, height=210)

    # ---- SUB-TAB B: CLASSIFICATION ----
    with subtab_clf:
        st.subheader("Return Category Classification — Decision Tree (Lab 5) vs. Gaussian Naive Bayes (Lab 6)")
        st.caption(
            f"Balanced Quantile Tertiles on `Daily_Return`: **Low (Bearish)** <= {q1_val:.2f}% | "
            f"**Medium (Neutral)** {q1_val:.2f}% to {q2_val:.2f}% | **High (Bullish)** > {q2_val:.2f}%"
        )

        st.markdown("#### 📊 Model Performance Comparison Table")
        st.dataframe(models_bundle["clf_metrics"], use_container_width=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            fig_fi = px.bar(
                models_bundle["feature_eval_df"],
                x="DT_Feature_Importance",
                y="Feature",
                orientation="h",
                color="Feature",
                title="Decision Tree Feature Importance (Entropy Information Gain)",
                height=340,
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        with fc2:
            fig_mi = px.bar(
                models_bundle["feature_eval_df"].sort_values("Mutual_Information", ascending=False),
                x="Mutual_Information",
                y="Feature",
                orientation="h",
                color="Feature",
                title="Mutual Information Attribute Relevance Score (nats)",
                height=340,
            )
            st.plotly_chart(fig_mi, use_container_width=True)

        st.markdown("#### 🧪 Live Session Trend Classifier (Low / Medium / High)")
        tc1, tc2, tc3, tc4 = st.columns(4)
        with tc1:
            clf_open = st.number_input("Session Open Price (₹):", min_value=50.0, max_value=20000.0, value=2600.0, step=25.0)
        with tc2:
            clf_vol = st.number_input("Session Volume:", min_value=1000, max_value=200000000, value=2000000, step=100000)
        with tc3:
            clf_ivol = st.slider("Intraday Volatility (%):", min_value=0.1, max_value=12.0, value=2.5, step=0.1)
        with tc4:
            clf_month = st.slider("Trading Month (1-12):", min_value=1, max_value=12, value=10, step=1)

        clf_sample = pd.DataFrame(
            {
                "Open_Price": [clf_open],
                "Volume": [clf_vol],
                "Intraday_Volatility": [clf_ivol],
                "Month": [clf_month],
            }
        )

        dt_pred = models_bundle["dt_model"].predict(clf_sample)[0]
        dt_proba = models_bundle["dt_model"].predict_proba(clf_sample)[0]
        nb_pred = models_bundle["nb_model"].predict(clf_sample)[0]
        nb_proba = models_bundle["nb_model"].predict_proba(clf_sample)[0]

        pc1, pc2 = st.columns(2)
        with pc1:
            st.success(f"**Decision Tree Prediction:** `{dt_pred}`")
            dt_prob_df = pd.DataFrame(
                [np.round(dt_proba, 4)], columns=models_bundle["dt_model"].classes_
            )[["Low", "Medium", "High"]]
            st.caption("Decision Tree Posterior Probabilities:")
            st.dataframe(dt_prob_df, use_container_width=True)

        with pc2:
            st.info(f"**Gaussian Naive Bayes Prediction:** `{nb_pred}`")
            nb_prob_df = pd.DataFrame(
                [np.round(nb_proba, 4)], columns=models_bundle["nb_model"].classes_
            )[["Low", "Medium", "High"]]
            st.caption("Gaussian Naive Bayes Posterior Probabilities:")
            st.dataframe(nb_prob_df, use_container_width=True)

# ===========================================================================
# TAB 4: Asset Clustering (K-Means & Hierarchical)
# ===========================================================================
elif nav_selection == "🎯 Asset Clustering (K-Means & Hierarchical)":
    st.title("🎯 Unsupervised Asset Clustering — K-Means (Lab 7) & Hierarchical (Lab 8)")
    st.markdown(
        "Stock-level risk-return-liquidity profiling across all 20 NSE stocks using standardized 5D features."
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("Optimal K (Elbow Method)", "K = 3 Clusters", "WCSS drops 100.0 -> 34.28")
    k2.metric("K-Means Silhouette Score", f"{km_sil:.4f}", "Balanced 7 / 7 / 6 Partition")
    k3.metric("Best Hierarchical Linkage", "Average (0.4042)", "Ward & Complete: 0.3922")

    st.markdown("#### 📋 20-Stock Risk, Return, Liquidity & Cluster Profile Table")
    display_cols = [
        "Ticker",
        "Sector",
        "Avg_Daily_Return",
        "Annualized_Volatility",
        "Avg_Intraday_Volatility",
        "Avg_Volume",
        "KMeans_Cluster",
        "Hier_Ward",
        "Persona",
    ]
    st.dataframe(clustered_df[display_cols], use_container_width=True, height=340)

    c_left, c_right = st.columns([1.35, 1.0])

    with c_left:
        fig_km = px.scatter(
            clustered_df,
            x="Annualized_Volatility",
            y="Avg_Daily_Return",
            color="Persona",
            symbol="Sector",
            text="Ticker",
            size="Avg_Volume_Log",
            hover_data=["Ticker", "Sector", "Avg_Volume", "Avg_Intraday_Volatility"],
            title="K-Means (K=3) Asset Segmentation: Annualized Volatility vs Avg Daily Return",
            height=480,
        )
        fig_km.update_traces(textposition="top center")
        fig_km.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_km, use_container_width=True)

    with c_right:
        fig_elbow = px.line(
            elbow_df,
            x="K",
            y="WCSS",
            markers=True,
            title="Elbow Method Curve (K = 1 to 10)",
            height=240,
        )
        fig_elbow.add_vline(x=3, line_dash="dash", line_color="red")
        st.plotly_chart(fig_elbow, use_container_width=True)

        st.markdown("**Hierarchical Linkage Comparison (Experiment 08)**")
        st.dataframe(linkage_df, use_container_width=True, height=190)

    st.markdown("#### 🌳 Interactive Hierarchical Clustering Dendrogram")
    sel_linkage = st.radio(
        "Select Linkage Criterion:",
        ["ward", "average", "complete", "single"],
        horizontal=True,
    )
    Z_mat = linkage(X_scaled_clust, method=sel_linkage, metric="euclidean")
    fig_dendro, ax_d = plt.subplots(figsize=(12, 4.2))
    dendrogram(
        Z_mat,
        labels=clustered_df["Ticker"].values,
        leaf_rotation=90,
        leaf_font_size=10,
        ax=ax_d,
    )
    ax_d.set_title(f"Agglomerative Hierarchical Dendrogram — {sel_linkage.capitalize()} Linkage")
    ax_d.set_xlabel("NSE Stock Ticker")
    ax_d.set_ylabel("Euclidean Linkage Distance")
    plt.tight_layout()
    st.pyplot(fig_dendro)

# ===========================================================================
# TAB 5: Association Rule Mining (Apriori Co-Movement)
# ===========================================================================
elif nav_selection == "🔗 Association Rule Mining (Apriori Co-Movement)":
    st.title("🔗 Apriori Association Rule Mining — Stock Co-Movement (Experiment 09)")
    st.markdown(
        "Each of the **740 trading days** is modeled as a market basket where stocks achieving a positive "
        "daily gain (`Daily_Return > 0`) are treated as co-purchased items."
    )

    sl_c1, sl_c2, sl_c3 = st.columns(3)
    with sl_c1:
        min_sup = st.slider(
            "Minimum Support Threshold:",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.01,
        )
    with sl_c2:
        min_conf = st.slider(
            "Minimum Confidence Threshold:",
            min_value=0.30,
            max_value=0.85,
            value=0.50,
            step=0.05,
        )
    with sl_c3:
        min_lift = st.slider(
            "Minimum Lift Filter:",
            min_value=1.00,
            max_value=2.00,
            value=1.10,
            step=0.05,
        )

    freq_items = apriori(basket_bool, min_support=min_sup, use_colnames=True)
    freq_items["itemset_size"] = freq_items["itemsets"].apply(len)

    if freq_items.empty or (freq_items["itemset_size"] >= 2).sum() == 0:
        st.warning("No multi-stock frequent itemsets found at this support threshold. Try lowering Minimum Support.")
    else:
        rules_df = association_rules(
            freq_items[["support", "itemsets"]],
            metric="confidence",
            min_threshold=min_conf,
        )
        rules_df = rules_df[rules_df["lift"] >= min_lift].copy()

        if rules_df.empty:
            st.warning("No association rules matched the Confidence/Lift filters. Try lowering the thresholds.")
        else:
            rules_df["Antecedent"] = rules_df["antecedents"].apply(
                lambda s: ", ".join(sorted(list(s)))
            )
            rules_df["Consequent"] = rules_df["consequents"].apply(
                lambda s: ", ".join(sorted(list(s)))
            )
            rules_df["Rule"] = rules_df["Antecedent"] + " → " + rules_df["Consequent"]
            rules_df = rules_df.sort_values(["lift", "confidence"], ascending=False).reset_index(
                drop=True
            )

            a1, a2, a3 = st.columns(3)
            a1.metric("Frequent Itemsets Found", f"{len(freq_items):,}", f"min_support = {min_sup:.2f}")
            a2.metric("Filtered Association Rules", f"{len(rules_df):,}", f"min_conf = {min_conf:.2f}")
            a3.metric(
                "Highest Lift Rule",
                f"{rules_df['lift'].iloc[0]:.2f}x",
                rules_df["Rule"].iloc[0],
            )

            st.markdown("#### 📋 Top Discovered Co-Movement Association Rules (Ranked by Lift)")
            st.dataframe(
                rules_df[["Antecedent", "Consequent", "support", "confidence", "lift"]]
                .round(4)
                .head(30),
                use_container_width=True,
                height=320,
            )

            ch_c1, ch_c2 = st.columns(2)
            top_n = rules_df.head(15).iloc[::-1]

            with ch_c1:
                fig_lift = px.bar(
                    top_n,
                    x="lift",
                    y="Rule",
                    color="confidence",
                    orientation="h",
                    title="Top 15 Co-Movement Association Rules by Lift",
                    color_continuous_scale="Magma",
                    height=460,
                )
                st.plotly_chart(fig_lift, use_container_width=True)

            with ch_c2:
                top_conf = (
                    rules_df.sort_values(["confidence", "lift"], ascending=False)
                    .head(15)
                    .iloc[::-1]
                )
                fig_conf = px.bar(
                    top_conf,
                    x="confidence",
                    y="Rule",
                    color="lift",
                    orientation="h",
                    title="Top 15 Co-Movement Association Rules by Confidence",
                    color_continuous_scale="Viridis",
                    height=460,
                )
                st.plotly_chart(fig_conf, use_container_width=True)

# ===========================================================================
# TAB 6: Project Summary & Lab Mapping
# ===========================================================================
elif nav_selection == "📑 Project Summary & Lab Mapping":
    st.title("📑 DWM Curriculum Mapping & Executive Summary (Experiments 01 – 10)")
    st.markdown(
        "This unified **Stock Market Analysis System** integrates the complete **Data Warehousing and Mining (DWM)** "
        "curriculum across a single, real-world 3-year NSE dataset (`14,800` records, `20` stocks, `5` sectors)."
    )

    st.markdown(
        """
| Experiment | Lab Title | Implementation in Stock Market Analysis System | Key Deliverable / Metric |
| :---: | :--- | :--- | :--- |
| **Exp 01** | **Data Warehouse Design & Schema Modeling** | Star Schema modeling with `Fact_Daily_Trading` (`Open`, `High`, `Low`, `Close`, `Volume`, `Daily_Return`) linked to `Dim_Stock` (`Ticker`, `Sector`, `Exchange`) and `Dim_Time` (`Date`, `Year`, `Quarter`, `Month`, `DayOfWeek`). | Unified schema in `stock_market_data.csv` (`14,800 × 10`). |
| **Exp 02** | **OLAP Operations (Roll-Up, Drill-Down, Slice, Dice, Pivot)** | Implemented interactively in **Tab 1 (Overview & Data Warehouse)** with multi-granularity aggregation and quarterly sector cross-tabulation. | Live Roll-Up, Drill-Down, Slice, Dice & Pivot Engine. |
| **Exp 03** | **Data Preprocessing, Exploration & Visualization** | Implemented in `DWM_Lab_3_Stock_Market.ipynb` & **Tab 2 (EDA)**: zero-null validation, `Intraday_Volatility` engineering, sector boxplots, and 7-feature correlation matrix. | 100% data completeness; multi-sector visual analytics. |
| **Exp 04** | **Simple & Multiple Linear Regression** | Implemented in `DWM_Lab_4_Stock_Market.ipynb` & **Tab 3A**: predicting `Close_Price` from `Open_Price` vs. full intraday bounds (`Open`, `High`, `Low`, `Volume`). | $R^2 = 0.999951$, $\\text{MSE} = 496.20$ (**83.24% MSE reduction**). |
| **Exp 05** | **Decision Tree Classification & Mutual Information** | Implemented in `DWM_Lab_5_Stock_Market.ipynb` & **Tab 3B**: entropy-based `DecisionTreeClassifier(max_depth=5)` on tertile `Return_Category` (`Low`, `Medium`, `High`). | **48.38% Accuracy**; `Intraday_Volatility` holds **95.09%** importance ($\\text{MI} = 0.1380$). |
| **Exp 06** | **Gaussian Naive Bayes Classification** | Implemented in `DWM_Lab_6_Stock_Market.ipynb` & **Tab 3B**: probabilistic 70:30 classification with posterior probability estimation $P(C \\mid X)$. | **35.61% Overall Accuracy**; **87.86% Recall** on Neutral (`Medium`) regime. |
| **Exp 07** | **K-Means Clustering for Asset Segmentation** | Implemented in `DWM_Lab_7_Stock_Market.ipynb` & **Tab 4**: log-volume + Z-score standardization across 20 stocks, Elbow WCSS curve, and $K=3$ persona profiling. | Optimal **$K = 3$**, **Silhouette Score = 0.4042**. |
| **Exp 08** | **Agglomerative Hierarchical Clustering** | Implemented in `DWM_Lab_8_Stock_Market.ipynb` & **Tab 4**: dendrogram construction and evaluation across Single, Complete, Average, and Ward linkages. | **Average Linkage Silhouette = 0.4042**, **Ward = 0.3922**, Single = $-0.1060$. |
| **Exp 09** | **Apriori Association Rule Mining** | Implemented in `DWM_Lab_9_Stock_Market.ipynb` & **Tab 5**: $740 \\times 20$ bullish market basket matrix mining synchronized intra-sector rallies. | **229 Frequent Itemsets**, **404 Rules**, max Lift = **2.40x** (`INFY, WIPRO → HCLTECH, TCS`). |
| **Exp 10** | **DWM Mini Project — Interactive Analytical Dashboard** | Implemented in `app.py`: end-to-end Streamlit decision-support system unifying OLAP, EDA, Regression, Classification, Clustering, and Apriori. | Production-ready interactive web application. |
        """
    )
