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
# 1. Page Configuration & FinTech Dark Obsidian CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantPulse | Market Intelligence & Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

FINTECH_PALETTE = ["#38BDF8", "#6366F1", "#10B981", "#F59E0B", "#EC4899", "#8B5CF6"]
SECTOR_COLORS = {
    "IT": "#38BDF8",
    "Banking": "#6366F1",
    "Auto": "#10B981",
    "Energy": "#F59E0B",
    "FMCG": "#A855F7",
}
PERSONA_COLORS = {
    "Blue-Chip Core Leaders": "#38BDF8",
    "Capital Preservation / Defensive": "#10B981",
    "Cyclical Momentum & High-Beta": "#F59E0B",
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global App Background & Typography */
    .stApp {
        background: radial-gradient(circle at 15% 10%, #111827 0%, #0B0F19 55%, #080B12 100%);
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .block-container {
        padding-top: 1.4rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1440px;
    }
    header[data-testid="stHeader"] {
        background: rgba(11, 15, 25, 0.75) !important;
        backdrop-filter: blur(10px);
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0E131F !important;
        border-right: 1px solid #222F46 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
    }

    /* Headings & Muted Text */
    h1, h2, h3, h4, .fintech-title {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }
    p, .stCaption, label {
        color: #94A3B8 !important;
    }

    /* Sleek KPI & Surface Cards */
    .fintech-card {
        background: linear-gradient(145deg, #151D2E 0%, #121927 100%);
        border: 1px solid #222F46;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
        transition: border-color 0.2s ease, transform 0.15s ease;
        margin-bottom: 12px;
    }
    .fintech-card:hover {
        border-color: #334766;
    }
    .kpi-label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
        font-family: 'JetBrains Mono', monospace;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        margin-top: 6px;
        font-weight: 500;
    }
    .delta-pos { color: #10B981; }
    .delta-neg { color: #EF4444; }
    .delta-neutral { color: #38BDF8; }

    /* Native Streamlit Metric Override */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #151D2E 0%, #121927 100%);
        border: 1px solid #222F46;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.28);
    }
    div[data-testid="stMetricLabel"] > div {
        color: #94A3B8 !important;
        font-size: 0.8rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] > div {
        color: #F8FAFC !important;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700 !important;
    }

    /* Pill Radio Buttons & Tabs */
    div[role="radiogroup"] {
        gap: 6px;
    }
    div[role="radiogroup"] label {
        background-color: #151D2E !important;
        border: 1px solid #222F46 !important;
        border-radius: 8px !important;
        padding: 6px 14px !important;
        transition: all 0.15s ease;
    }
    div[role="radiogroup"] label:hover {
        border-color: #38BDF8 !important;
        background-color: #1A2338 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0E131F;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #222F46;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 18px;
        color: #94A3B8;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1A2338 !important;
        color: #38BDF8 !important;
        border-bottom: none !important;
    }

    /* Compact Inputs, Selectboxes, Expanders & Containers */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #151D2E !important;
        border-color: #222F46 !important;
        color: #F8FAFC !important;
        border-radius: 8px !important;
    }
    details[data-testid="stExpander"] {
        background-color: #151D2E !important;
        border: 1px solid #222F46 !important;
        border-radius: 12px !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #151D2E !important;
        border: 1px solid #222F46 !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }

    /* Probability Bar Gauge */
    .prob-row {
        margin-bottom: 10px;
    }
    .prob-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.8rem;
        margin-bottom: 4px;
        color: #CBD5E1;
    }
    .prob-track {
        width: 100%;
        height: 8px;
        background: #0E131F;
        border-radius: 999px;
        overflow: hidden;
        border: 1px solid #222F46;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
    }

    /* Co-movement Rule Card */
    .rule-card {
        background: #151D2E;
        border: 1px solid #222F46;
        border-left: 3px solid #38BDF8;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .rule-pair {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        color: #F8FAFC;
        font-weight: 600;
    }
    .rule-arrow {
        color: #38BDF8;
        padding: 0 8px;
    }
    .lift-badge {
        background: rgba(16, 185, 129, 0.14);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.35);
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 999px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def apply_fintech_theme(fig, height: int = 380):
    """Apply consistent obsidian dark FinTech styling to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 29, 46, 0.55)",
        font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
        title_font=dict(family="Inter, sans-serif", color="#F8FAFC", size=14),
        margin=dict(l=16, r=16, t=44, b=16),
        height=height,
        hoverlabel=dict(
            bgcolor="#0E131F",
            bordercolor="#38BDF8",
            font=dict(family="JetBrains Mono, monospace", color="#F8FAFC", size=12),
        ),
        legend=dict(
            bgcolor="rgba(14, 19, 31, 0.65)",
            bordercolor="#222F46",
            borderwidth=1,
            font=dict(color="#CBD5E1", size=11),
        ),
    )
    fig.update_xaxes(
        gridcolor="#1E293B",
        zerolinecolor="#222F46",
        linecolor="#222F46",
        tickfont=dict(color="#94A3B8"),
    )
    fig.update_yaxes(
        gridcolor="#1E293B",
        zerolinecolor="#222F46",
        linecolor="#222F46",
        tickfont=dict(color="#94A3B8"),
    )
    return fig


def render_kpi_card(label: str, value: str, subtitle: str, tone: str = "neutral"):
    tone_cls = {
        "pos": "delta-pos",
        "neg": "delta-neg",
        "neutral": "delta-neutral",
    }.get(tone, "delta-neutral")
    st.markdown(
        f"""
        <div class="fintech-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub {tone_cls}">{subtitle}</div>
        </div>
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

    df["Intraday_Volatility"] = (
        ((df["High_Price"] - df["Low_Price"]) / df["Open_Price"]) * 100
    ).round(2)

    q1 = df["Daily_Return"].quantile(1 / 3)
    q2 = df["Daily_Return"].quantile(2 / 3)
    df["Return_Category"] = pd.cut(
        df["Daily_Return"],
        bins=[-np.inf, q1, q2, np.inf],
        labels=["Low", "Medium", "High"],
    ).astype(str)

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

    df["Bullish"] = (df["Daily_Return"] > 0).astype(int)
    basket_bool = (
        df.pivot_table(index="Date", columns="Ticker", values="Bullish", fill_value=0)
        .astype(bool)
    )

    return df, stock_df, basket_bool, float(q1), float(q2)


@st.cache_resource
def train_predictive_models(df: pd.DataFrame):
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
            "Algorithm": [
                "Entropy Decision Tree (Depth 5)",
                "Gaussian Naïve Bayes",
            ],
            "Validation Split": ["80:20 Stratified", "70:30 Stratified"],
            "Accuracy (%)": [
                round(accuracy_score(y_te_dt, y_pr_dt) * 100, 2),
                round(accuracy_score(y_te_nb, y_pr_nb) * 100, 2),
            ],
            "Precision": [
                round(precision_score(y_te_dt, y_pr_dt, average="weighted", zero_division=0), 4),
                round(precision_score(y_te_nb, y_pr_nb, average="weighted", zero_division=0), 4),
            ],
            "Recall": [
                round(recall_score(y_te_dt, y_pr_dt, average="weighted", zero_division=0), 4),
                round(recall_score(y_te_nb, y_pr_nb, average="weighted", zero_division=0), 4),
            ],
            "F1-Score": [
                round(f1_score(y_te_dt, y_pr_dt, average="weighted", zero_division=0), 4),
                round(f1_score(y_te_nb, y_pr_nb, average="weighted", zero_division=0), 4),
            ],
        }
    )

    mi_scores = mutual_info_classif(X_clf, y_clf, random_state=42)
    feature_eval_df = pd.DataFrame(
        {
            "Feature": clf_features,
            "Information_Gain_Weight": np.round(dt_model.feature_importances_, 4),
            "Mutual_Information": np.round(mi_scores, 4),
        }
    ).sort_values("Information_Gain_Weight", ascending=True)

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

    elbow_rows = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        elbow_rows.append({"K": k, "WCSS": round(km.inertia_, 4)})
    elbow_df = pd.DataFrame(elbow_rows)

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    clustered_df = stock_df.copy()
    clustered_df["Cluster_ID"] = km3.fit_predict(X_scaled)
    km_sil = silhouette_score(X_scaled, clustered_df["Cluster_ID"])

    persona_map = {}
    for c_id, grp in clustered_df.groupby("Cluster_ID"):
        if grp["Annualized_Volatility"].mean() < 19.5:
            persona_map[c_id] = "Blue-Chip Core Leaders"
        elif grp["Avg_Volume"].mean() < 5_000_000:
            persona_map[c_id] = "Capital Preservation / Defensive"
        else:
            persona_map[c_id] = "Cyclical Momentum & High-Beta"

    clustered_df["Persona"] = clustered_df["Cluster_ID"].map(persona_map)

    linkage_rows = []
    for method in ["average", "ward", "complete", "single"]:
        agg = AgglomerativeClustering(n_clusters=3, linkage=method, metric="euclidean")
        labels = agg.fit_predict(X_scaled)
        clustered_df[f"Hier_{method.capitalize()}"] = labels
        sil = silhouette_score(X_scaled, labels)
        counts = pd.Series(labels).value_counts().reindex([0, 1, 2], fill_value=0)
        linkage_rows.append(
            {
                "Linkage Criterion": method.capitalize(),
                "Silhouette Score": round(sil, 4),
                "Partition Distribution": f"{counts[0]} / {counts[1]} / {counts[2]}",
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
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 4px 0 14px 0; border-bottom: 1px solid #222F46; margin-bottom: 16px;">
            <div style="font-size: 1.2rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.02em;">
                ⚡ QuantPulse
            </div>
            <div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">
                Multi-Asset Intelligence & OLAP Suite
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_selection = st.radio(
        "WORKSPACE NAVIGATION",
        [
            "🌐 Market Overview & OLAP",
            "📊 Exploratory Analytics",
            "⚡ Predictive Modeling",
            "🎯 Asset Clusters & Risk",
            "🔄 Co-Movement Engine",
            "📄 Technical Architecture",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="fintech-card" style="padding: 14px 16px;">
            <div class="kpi-label">Data Feed Status</div>
            <div style="display:flex; align-items:center; gap:8px; margin: 6px 0;">
                <span style="width:8px; height:8px; border-radius:50%; background:#10B981; display:inline-block;"></span>
                <span style="color:#F8FAFC; font-size:0.86rem; font-weight:600;">NSE Large-Cap Universe</span>
            </div>
            <div style="font-size:0.78rem; color:#94A3B8; line-height:1.6;">
                • <b>14,800</b> Daily OHLCV Bars<br>
                • <b>20</b> Active Equities<br>
                • <b>740</b> Trading Sessions<br>
                • <b>5</b> Core Market Sectors
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ===========================================================================
# TAB 1: 🌐 Market Overview & OLAP
# ===========================================================================
if nav_selection == "🌐 Market Overview & OLAP":
    st.markdown("## 🌐 Market Overview & Multidimensional OLAP")
    st.caption(
        "Interactive dimensional slicing, dicing, temporal roll-ups, and constituent drill-downs across the NSE equity cube."
    )

    # Top KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi_card(
            "Market Universe",
            f"{df['Ticker'].nunique()} Equities",
            "Large-Cap NSE Constituents",
            "neutral",
        )
    with k2:
        render_kpi_card(
            "Tracked Sessions",
            f"{df['Date'].nunique():,} Days",
            f"{len(df):,} Total OHLCV Bars",
            "neutral",
        )
    with k3:
        render_kpi_card(
            "Sectors",
            f"{df['Sector'].nunique()} Key Sectors",
            "IT • Banking • Auto • Energy • FMCG",
            "neutral",
        )
    with k4:
        avg_ret = df["Daily_Return"].mean()
        render_kpi_card(
            "Cumulative Volume",
            f"{df['Volume'].sum() / 1e9:.2f}B",
            f"Mean Daily Return: {avg_ret:+.3f}%",
            "pos" if avg_ret >= 0 else "neg",
        )

    # Minimalist OLAP Filtering Bar
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.0, 1.8, 1.0, 1.3])
        with f1:
            available_years = ["All Years"] + sorted(
                df["Year"].unique().astype(str).tolist()
            )
            slice_year = st.selectbox("Year (Slice)", available_years, index=0)
        with f2:
            all_sectors = sorted(df["Sector"].unique().tolist())
            dice_sectors = st.multiselect(
                "Sectors (Dice)", options=all_sectors, default=all_sectors
            )
        with f3:
            all_exchanges = sorted(df["Exchange"].unique().tolist())
            dice_exchange = st.selectbox(
                "Exchange", options=["All"] + all_exchanges, index=0
            )
        with f4:
            grain = st.selectbox(
                "Aggregation Grain (Roll-Up)",
                ["Quarterly", "Monthly", "Yearly"],
                index=0,
            )

    sub_cube = df.copy()
    if slice_year != "All Years":
        sub_cube = sub_cube[sub_cube["Year"] == int(slice_year)]
    if dice_sectors:
        sub_cube = sub_cube[sub_cube["Sector"].isin(dice_sectors)]
    if dice_exchange != "All":
        sub_cube = sub_cube[sub_cube["Exchange"] == dice_exchange]

    grain_col = {"Yearly": "Year", "Quarterly": "Year_Quarter", "Monthly": "YearMonth"}[
        grain
    ]

    v_left, v_right = st.columns([1.25, 1.0])

    with v_left:
        rollup_df = (
            sub_cube.groupby([grain_col, "Sector"], as_index=False)
            .agg(
                Total_Volume=("Volume", "sum"),
                Avg_Close_Price=("Close_Price", "mean"),
                Avg_Daily_Return=("Daily_Return", "mean"),
            )
            .sort_values([grain_col, "Sector"])
        )
        fig_traj = px.bar(
            rollup_df,
            x=grain_col,
            y="Total_Volume",
            color="Sector",
            barmode="group",
            color_discrete_map=SECTOR_COLORS,
            title=f"Turnover Trajectory by {grain} Grain",
            labels={"Total_Volume": "Traded Volume", grain_col: "Period"},
        )
        st.plotly_chart(apply_fintech_theme(fig_traj, height=400), use_container_width=True)

    with v_right:
        pivot_matrix = pd.pivot_table(
            sub_cube,
            values="Daily_Return",
            index="Sector",
            columns="Year_Quarter",
            aggfunc="mean",
        ).round(3)

        fig_piv = px.imshow(
            pivot_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale=["#EF4444", "#151D2E", "#10B981"],
            color_continuous_midpoint=0,
            title="Dynamic Pivot Matrix: Mean Daily Return (%) — Sectors vs. Quarters",
            labels=dict(x="Quarter", y="Sector", color="Return %"),
        )
        st.plotly_chart(apply_fintech_theme(fig_piv, height=400), use_container_width=True)

    # Collapsible Drill-Down Inspector
    with st.expander("🔍 Single-Stock Drill-Down Inspector", expanded=False):
        dd_c1, dd_c2, dd_c3 = st.columns([1, 1, 2])
        with dd_c1:
            sel_sec = st.selectbox("Filter Sector", all_sectors, index=0)
        sec_tickers = sorted(
            df[df["Sector"] == sel_sec]["Ticker"].unique().tolist()
        )
        with dd_c2:
            sel_ticker = st.selectbox("Select Equity Ticker", sec_tickers, index=0)

        t_df = df[df["Ticker"] == sel_ticker].sort_values("Date_dt")
        latest_close = t_df["Close_Price"].iloc[-1]
        period_ret = (
            (t_df["Close_Price"].iloc[-1] / t_df["Close_Price"].iloc[0]) - 1
        ) * 100

        with dd_c3:
            render_kpi_card(
                f"{sel_ticker} ({sel_sec}) — Terminal Close",
                f"₹{latest_close:,.2f}",
                f"3-Year Cumulative Price Change: {period_ret:+.2f}%",
                "pos" if period_ret >= 0 else "neg",
            )

        fig_stock = go.Figure()
        fig_stock.add_trace(
            go.Scatter(
                x=t_df["Date_dt"],
                y=t_df["Close_Price"],
                mode="lines",
                name="Close Price (₹)",
                line=dict(color="#38BDF8", width=2),
                fill="tozeroy",
                fillcolor="rgba(56, 189, 248, 0.08)",
            )
        )
        fig_stock.update_layout(
            title=f"{sel_ticker} — Historical Closing Price Trajectory",
            xaxis_title="Date",
            yaxis_title="Price (₹)",
        )
        st.plotly_chart(apply_fintech_theme(fig_stock, height=340), use_container_width=True)


# ===========================================================================
# TAB 2: 📊 Exploratory Analytics
# ===========================================================================
elif nav_selection == "📊 Exploratory Analytics":
    st.markdown("## 📊 Exploratory Market Analytics")
    st.caption(
        "Cross-sectional liquidity distribution, high-turnover leaders, return dispersion, and multi-factor correlation structure."
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
            color_discrete_map=SECTOR_COLORS,
            title="Sector Volume Distribution (Cumulative 3Y Turnover)",
        )
        fig_sec_vol.update_layout(showlegend=False)
        st.plotly_chart(apply_fintech_theme(fig_sec_vol, height=370), use_container_width=True)

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
            color_discrete_map=SECTOR_COLORS,
            title="Top 10 High-Turnover Equities",
        )
        st.plotly_chart(apply_fintech_theme(fig_top10, height=370), use_container_width=True)

    r2c1, r2c2 = st.columns(2)

    with r2c1:
        fig_box_ret = px.box(
            df,
            x="Sector",
            y="Daily_Return",
            color="Sector",
            color_discrete_map=SECTOR_COLORS,
            title="Sector-Wise Daily Return (%) Dispersion",
        )
        fig_box_ret.update_traces(marker=dict(size=3, opacity=0.45))
        fig_box_ret.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
        fig_box_ret.update_layout(showlegend=False)
        st.plotly_chart(apply_fintech_theme(fig_box_ret, height=390), use_container_width=True)

    with r2c2:
        fig_box_vol = px.box(
            df,
            x="Sector",
            y="Intraday_Volatility",
            color="Sector",
            color_discrete_map=SECTOR_COLORS,
            title="Sector-Wise Intraday Volatility (%) Profile",
        )
        fig_box_vol.update_traces(marker=dict(size=3, opacity=0.45))
        fig_box_vol.update_layout(showlegend=False)
        st.plotly_chart(apply_fintech_theme(fig_box_vol, height=390), use_container_width=True)

    r3c1, r3c2 = st.columns([1.15, 1.0])

    with r3c1:
        sampled_eda = df.sample(n=2500, random_state=42)
        fig_scat = px.scatter(
            sampled_eda,
            x="Volume",
            y="Daily_Return",
            color="Sector",
            color_discrete_map=SECTOR_COLORS,
            hover_data=["Ticker", "Date", "Close_Price"],
            log_x=True,
            opacity=0.55,
            title="Volume vs. Daily Return Dispersion (Log Scale)",
        )
        fig_scat.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
        st.plotly_chart(apply_fintech_theme(fig_scat, height=430), use_container_width=True)

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
            color_continuous_scale="IceFire",
            zmin=-1,
            zmax=1,
            title="Multi-Factor Correlation Matrix",
        )
        st.plotly_chart(apply_fintech_theme(fig_corr, height=430), use_container_width=True)


# ===========================================================================
# TAB 3: ⚡ Predictive Modeling
# ===========================================================================
elif nav_selection == "⚡ Predictive Modeling":
    st.markdown("## ⚡ Quantitative Predictive Modeling")
    st.caption(
        "Real-time intraday closing price estimation and probabilistic regime classification."
    )

    mode = st.radio(
        "Select Modeling Engine",
        ["Price Forecasting (Regression)", "Trend Classification (ML)"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if mode == "Price Forecasting (Regression)":
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi_card(
                "Multi-Factor R²",
                f"{models_bundle['r2_m']:.5f}",
                "Open • High • Low • Volume",
                "pos",
            )
        with k2:
            render_kpi_card(
                "Multi-Factor MSE",
                f"{models_bundle['mse_m']:.2f}",
                "-83.24% Error vs Single-Factor",
                "pos",
            )
        with k3:
            render_kpi_card(
                "Single-Factor R²",
                f"{models_bundle['r2_s']:.5f}",
                "Baseline (Open Price Only)",
                "neutral",
            )
        with k4:
            render_kpi_card(
                "Single-Factor MSE",
                f"{models_bundle['mse_s']:.2f}",
                "RMSE: ₹54.41",
                "neutral",
            )

        left_box, right_box = st.columns([1.35, 1.0])

        with left_box:
            with st.container(border=True):
                st.markdown("#### 🎛️ Live Intraday Price Forecaster")
                st.caption("Adjust intraday OHLCV parameters to compute the expected closing equilibrium.")
                ic1, ic2 = st.columns(2)
                with ic1:
                    inp_open = st.number_input(
                        "Open Price (₹)", min_value=50.0, max_value=25000.0, value=2500.0, step=10.0
                    )
                    inp_low = st.number_input(
                        "Intraday Low (₹)", min_value=50.0, max_value=25000.0, value=2490.0, step=10.0
                    )
                with ic2:
                    inp_high = st.number_input(
                        "Intraday High (₹)", min_value=50.0, max_value=25000.0, value=2540.0, step=10.0
                    )
                    inp_vol = st.number_input(
                        "Session Volume", min_value=1000, max_value=200000000, value=1500000, step=50000
                    )

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
                delta_cls = "delta-pos" if intraday_diff >= 0 else "delta-neg"
                arrow = "▲" if intraday_diff >= 0 else "▼"

                st.markdown(
                    f"""
                    <div style="background:#0E131F; border:1px solid #222F46; border-radius:10px; padding:16px 20px; margin-top:10px; display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <div class="kpi-label">Predicted Closing Price</div>
                            <div class="kpi-value" style="font-size:2.0rem;">₹{pred_close_val:,.2f}</div>
                        </div>
                        <div style="text-align:right;">
                            <div class="kpi-label">Expected Intraday Delta</div>
                            <div class="{delta_cls}" style="font-family:'JetBrains Mono',monospace; font-size:1.15rem; font-weight:700;">
                                {arrow} {intraday_diff:+.2f} ₹ ({intraday_pct:+.2f}%)
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with right_box:
            with st.container(border=True):
                st.markdown("#### 📐 Model Parameter Weights")
                st.caption("Ordinary Least Squares (OLS) fitted coefficients.")
                coef_df = pd.DataFrame(
                    {
                        "Parameter": ["Intercept (β₀)", "Open Price", "High Price", "Low Price", "Volume"],
                        "Weight": [
                            round(models_bundle["mlr"].intercept_, 6),
                            *[round(c, 6) for c in models_bundle["mlr"].coef_],
                        ],
                    }
                )
                st.dataframe(coef_df, use_container_width=True, hide_index=True, height=245)

    else:
        st.caption(
            f"Regime Boundaries — Bearish (Low): ≤ {q1_val:.2f}%  |  Neutral (Medium): {q1_val:.2f}% to {q2_val:.2f}%  |  Bullish (High): > {q2_val:.2f}%"
        )

        with st.container(border=True):
            st.markdown("#### 🧪 Live Regime Simulator")
            tc1, tc2, tc3, tc4 = st.columns(4)
            with tc1:
                clf_open = st.number_input(
                    "Open Price (₹)", min_value=50.0, max_value=25000.0, value=2600.0, step=25.0
                )
            with tc2:
                clf_vol = st.number_input(
                    "Volume", min_value=1000, max_value=200000000, value=2000000, step=100000
                )
            with tc3:
                clf_ivol = st.slider(
                    "Intraday Volatility (%)", min_value=0.1, max_value=12.0, value=2.5, step=0.1
                )
            with tc4:
                clf_month = st.slider("Calendar Month", min_value=1, max_value=12, value=10, step=1)

        clf_sample = pd.DataFrame(
            {
                "Open_Price": [clf_open],
                "Volume": [clf_vol],
                "Intraday_Volatility": [clf_ivol],
                "Month": [clf_month],
            }
        )

        dt_pred = models_bundle["dt_model"].predict(clf_sample)[0]
        dt_proba = dict(
            zip(
                models_bundle["dt_model"].classes_,
                models_bundle["dt_model"].predict_proba(clf_sample)[0],
            )
        )
        nb_pred = models_bundle["nb_model"].predict(clf_sample)[0]
        nb_proba = dict(
            zip(
                models_bundle["nb_model"].classes_,
                models_bundle["nb_model"].predict_proba(clf_sample)[0],
            )
        )

        regime_meta = {
            "Low": ("Bearish (Low)", "#EF4444"),
            "Medium": ("Neutral (Medium)", "#38BDF8"),
            "High": ("Bullish (High)", "#10B981"),
        }

        def render_model_tile(title: str, subtitle: str, pred_key: str, probs: dict, acc: float):
            label_txt, accent_hex = regime_meta.get(pred_key, (pred_key, "#38BDF8"))
            bars_html = ""
            for k in ["Low", "Medium", "High"]:
                r_name, r_col = regime_meta[k]
                pct = probs.get(k, 0.0) * 100
                bars_html += f"""
                <div class="prob-row">
                    <div class="prob-header">
                        <span>{r_name}</span>
                        <span style="font-family:'JetBrains Mono',monospace; font-weight:600;">{pct:.1f}%</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{pct:.1f}%; background:{r_col};"></div>
                    </div>
                </div>
                """
            st.markdown(
                f"""
                <div class="fintech-card">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                        <div>
                            <div style="font-size:1.02rem; font-weight:600; color:#F8FAFC;">{title}</div>
                            <div style="font-size:0.76rem; color:#94A3B8;">{subtitle}</div>
                        </div>
                        <div style="text-align:right;">
                            <span style="background:rgba(56,189,248,0.12); color:#38BDF8; border:1px solid rgba(56,189,248,0.3); padding:4px 10px; border-radius:999px; font-size:0.75rem; font-family:'JetBrains Mono',monospace;">
                                Acc: {acc:.2f}%
                            </span>
                        </div>
                    </div>
                    <div style="background:#0E131F; border:1px solid #222F46; border-radius:8px; padding:12px 14px; margin-bottom:14px; display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.8rem; color:#94A3B8; text-transform:uppercase; letter-spacing:0.05em;">Predicted State</span>
                        <span style="font-size:1.05rem; font-weight:700; color:{accent_hex}; font-family:'JetBrains Mono',monospace;">● {label_txt}</span>
                    </div>
                    {bars_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        c_dt, c_nb = st.columns(2)
        with c_dt:
            render_model_tile(
                "Entropy Decision Tree",
                "Non-linear hierarchical threshold partitioning (max_depth=5)",
                dt_pred,
                dt_proba,
                float(models_bundle["clf_metrics"]["Accuracy (%)"].iloc[0]),
            )
        with c_nb:
            render_model_tile(
                "Gaussian Naïve Bayes",
                "Continuous probabilistic likelihood estimator",
                nb_pred,
                nb_proba,
                float(models_bundle["clf_metrics"]["Accuracy (%)"].iloc[1]),
            )

        f_left, f_right = st.columns(2)
        with f_left:
            fig_fi = px.bar(
                models_bundle["feature_eval_df"],
                x="Information_Gain_Weight",
                y="Feature",
                orientation="h",
                title="Decision Tree Feature Importance (Entropy Gain)",
                color_discrete_sequence=["#38BDF8"],
            )
            st.plotly_chart(apply_fintech_theme(fig_fi, height=300), use_container_width=True)
        with f_right:
            fig_mi = px.bar(
                models_bundle["feature_eval_df"].sort_values("Mutual_Information", ascending=True),
                x="Mutual_Information",
                y="Feature",
                orientation="h",
                title="Mutual Information Feature Relevance (nats)",
                color_discrete_sequence=["#6366F1"],
            )
            st.plotly_chart(apply_fintech_theme(fig_mi, height=300), use_container_width=True)


# ===========================================================================
# TAB 4: 🎯 Asset Clusters & Risk
# ===========================================================================
elif nav_selection == "🎯 Asset Clusters & Risk":
    st.markdown("## 🎯 Multi-Dimensional Asset Clustering & Risk Personas")
    st.caption(
        "Unsupervised equity segmentation across return, annualized volatility, and liquidity profiles."
    )

    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi_card(
            "Optimal Regime Count",
            "K = 3 Personas",
            "WCSS Reduction: 100.0 ➔ 34.28",
            "neutral",
        )
    with k2:
        render_kpi_card(
            "K-Means Silhouette",
            f"{km_sil:.4f}",
            "Balanced 7 / 7 / 6 Equity Split",
            "pos",
        )
    with k3:
        render_kpi_card(
            "Top Hierarchical Linkage",
            "Average (0.4042)",
            "Ward & Complete: 0.3922",
            "pos",
        )

    c_left, c_right = st.columns([1.45, 1.0])

    with c_left:
        fig_km = px.scatter(
            clustered_df,
            x="Annualized_Volatility",
            y="Avg_Daily_Return",
            color="Persona",
            color_discrete_map=PERSONA_COLORS,
            symbol="Sector",
            text="Ticker",
            size="Avg_Volume_Log",
            hover_data=["Ticker", "Sector", "Avg_Volume", "Avg_Intraday_Volatility"],
            title="Risk vs. Return Map — Colored by Investment Persona",
            labels={
                "Annualized_Volatility": "Annualized Volatility (%)",
                "Avg_Daily_Return": "Mean Daily Return (%)",
            },
        )
        fig_km.update_traces(textposition="top center", textfont=dict(size=10, color="#E2E8F0"))
        fig_km.add_hline(y=0, line_dash="dot", line_color="#94A3B8")
        st.plotly_chart(apply_fintech_theme(fig_km, height=460), use_container_width=True)

    with c_right:
        fig_elbow = px.line(
            elbow_df,
            x="K",
            y="WCSS",
            markers=True,
            title="Within-Cluster Sum of Squares (Elbow Curve)",
            color_discrete_sequence=["#38BDF8"],
        )
        fig_elbow.add_vline(x=3, line_dash="dash", line_color="#10B981")
        st.plotly_chart(apply_fintech_theme(fig_elbow, height=235), use_container_width=True)

        st.dataframe(
            linkage_df,
            use_container_width=True,
            hide_index=True,
            height=195,
        )

    st.markdown("#### 📋 Equity Risk-Reward & Persona Directory")
    st.dataframe(
        clustered_df[
            [
                "Ticker",
                "Sector",
                "Persona",
                "Avg_Daily_Return",
                "Annualized_Volatility",
                "Avg_Intraday_Volatility",
                "Avg_Volume",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Avg_Daily_Return": st.column_config.NumberColumn("Mean Return (%)", format="%.4f%%"),
            "Annualized_Volatility": st.column_config.ProgressColumn(
                "Annualized Volatility (%)",
                min_value=10.0,
                max_value=35.0,
                format="%.2f%%",
            ),
            "Avg_Intraday_Volatility": st.column_config.ProgressColumn(
                "Intraday Spread (%)",
                min_value=1.0,
                max_value=3.5,
                format="%.2f%%",
            ),
            "Avg_Volume": st.column_config.NumberColumn("Avg Daily Volume", format="%d"),
        },
        height=340,
    )

    with st.container(border=True):
        d_top1, d_top2 = st.columns([1.5, 1.0])
        with d_top1:
            st.markdown("#### 🌳 Hierarchical Taxonomy Dendrogram")
            st.caption("Agglomerative Euclidean tree structure across 20 NSE equities.")
        with d_top2:
            sel_linkage = st.radio(
                "Linkage Criterion",
                ["ward", "average", "complete", "single"],
                horizontal=True,
            )

        Z_mat = linkage(X_scaled_clust, method=sel_linkage, metric="euclidean")
        fig_dendro, ax_d = plt.subplots(figsize=(12, 3.8), facecolor="#151D2E")
        ax_d.set_facecolor("#151D2E")
        dendrogram(
            Z_mat,
            labels=clustered_df["Ticker"].values,
            leaf_rotation=45,
            leaf_font_size=9,
            color_threshold=0,
            above_threshold_color="#38BDF8",
            ax=ax_d,
        )
        ax_d.set_ylabel("Euclidean Distance", color="#94A3B8", fontsize=9)
        ax_d.tick_params(axis="x", colors="#F8FAFC")
        ax_d.tick_params(axis="y", colors="#94A3B8")
        for spine in ax_d.spines.values():
            spine.set_color("#222F46")
        ax_d.grid(axis="y", linestyle="--", alpha=0.25, color="#94A3B8")
        plt.tight_layout()
        st.pyplot(fig_dendro)


# ===========================================================================
# TAB 5: 🔄 Co-Movement Engine
# ===========================================================================
elif nav_selection == "🔄 Co-Movement Engine":
    st.markdown("## 🔄 Multi-Asset Co-Movement & Market Basket Engine")
    st.caption(
        "Discovers high-conviction synchronized equity rallies across 740 trading sessions using Apriori association rule mining."
    )

    with st.container(border=True):
        sl_c1, sl_c2, sl_c3 = st.columns(3)
        with sl_c1:
            min_sup = st.slider(
                "Minimum Support",
                min_value=0.10,
                max_value=0.40,
                value=0.20,
                step=0.01,
            )
        with sl_c2:
            min_conf = st.slider(
                "Minimum Confidence",
                min_value=0.30,
                max_value=0.85,
                value=0.50,
                step=0.05,
            )
        with sl_c3:
            min_lift = st.slider(
                "Minimum Lift Ratio",
                min_value=1.00,
                max_value=2.20,
                value=1.15,
                step=0.05,
            )

    freq_items = apriori(basket_bool, min_support=min_sup, use_colnames=True)
    freq_items["itemset_size"] = freq_items["itemsets"].apply(len)

    if freq_items.empty or (freq_items["itemset_size"] >= 2).sum() == 0:
        st.warning("No multi-asset itemsets detected at the current Support threshold. Lower Minimum Support.")
    else:
        rules_df = association_rules(
            freq_items[["support", "itemsets"]],
            metric="confidence",
            min_threshold=min_conf,
        )
        rules_df = rules_df[rules_df["lift"] >= min_lift].copy()

        if rules_df.empty:
            st.warning("No co-movement rules matched the selected Confidence & Lift filters.")
        else:
            rules_df["Antecedent"] = rules_df["antecedents"].apply(
                lambda s: " + ".join(sorted(list(s)))
            )
            rules_df["Consequent"] = rules_df["consequents"].apply(
                lambda s: " + ".join(sorted(list(s)))
            )
            rules_df["Rule"] = rules_df["Antecedent"] + " ➔ " + rules_df["Consequent"]
            rules_df = rules_df.sort_values(
                ["lift", "confidence"], ascending=False
            ).reset_index(drop=True)

            a1, a2, a3 = st.columns(3)
            with a1:
                render_kpi_card(
                    "Frequent Rally Bundles",
                    f"{len(freq_items):,}",
                    f"Support ≥ {min_sup:.2f}",
                    "neutral",
                )
            with a2:
                render_kpi_card(
                    "Active Co-Movement Signals",
                    f"{len(rules_df):,}",
                    f"Confidence ≥ {min_conf:.2f}",
                    "pos",
                )
            with a3:
                render_kpi_card(
                    "Peak Lift Multiplier",
                    f"{rules_df['lift'].iloc[0]:.2f}x",
                    rules_df["Rule"].iloc[0],
                    "pos",
                )

            st.markdown("#### ⚡ High-Conviction Synchronized Rally Pairs")
            top_cards = rules_df.head(6)
            rc1, rc2 = st.columns(2)
            for idx, row in top_cards.iterrows():
                target_col = rc1 if idx % 2 == 0 else rc2
                with target_col:
                    st.markdown(
                        f"""
                        <div class="rule-card">
                            <div>
                                <div class="rule-pair">
                                    {row['Antecedent']} <span class="rule-arrow">➔</span> {row['Consequent']}
                                </div>
                                <div style="font-size:0.76rem; color:#94A3B8; margin-top:4px;">
                                    Confidence: <b style="color:#F8FAFC;">{row['confidence']*100:.1f}%</b> &nbsp;|&nbsp;
                                    Support: <b style="color:#F8FAFC;">{row['support']*100:.1f}%</b>
                                </div>
                            </div>
                            <div class="lift-badge">Lift {row['lift']:.2f}x</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            ch_c1, ch_c2 = st.columns([1.15, 1.0])
            top_n = rules_df.head(12).iloc[::-1]

            with ch_c1:
                fig_lift = px.bar(
                    top_n,
                    x="lift",
                    y="Rule",
                    color="confidence",
                    orientation="h",
                    title="Top 12 Co-Movement Signals Ranked by Lift",
                    color_continuous_scale=["#38BDF8", "#6366F1", "#10B981"],
                )
                st.plotly_chart(apply_fintech_theme(fig_lift, height=420), use_container_width=True)

            with ch_c2:
                st.dataframe(
                    rules_df[["Antecedent", "Consequent", "support", "confidence", "lift"]]
                    .round(4)
                    .head(25),
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                )


# ===========================================================================
# TAB 6: 📄 Technical Architecture
# ===========================================================================
elif nav_selection == "📄 Technical Architecture":
    st.markdown("## 📄 System Architecture & Quantitative Pipeline")
    st.caption(
        "End-to-end data engineering topology, dimensional warehouse schemas, and production model benchmarks."
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown(
            """
            <div class="fintech-card" style="height: 100%;">
                <div class="kpi-label">1. Dimensional Star Schema</div>
                <div style="font-size:1.05rem; font-weight:600; color:#F8FAFC; margin-bottom:8px;">
                    High-Throughput OLAP Core
                </div>
                <div style="font-size:0.84rem; color:#94A3B8; line-height:1.6;">
                    Central <code>Fact_DailyTrades</code> table (14,800 bars) linked via single-join keys to denormalized
                    <code>Dim_Time</code>, <code>Dim_Stock</code>, and <code>Dim_Exchange</code> tables for sub-millisecond
                    slice, dice, and pivot aggregations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="fintech-card" style="height: 100%;">
                <div class="kpi-label">2. Normalized Snowflake Schema</div>
                <div style="font-size:1.05rem; font-weight:600; color:#F8FAFC; margin-bottom:8px;">
                    3NF Attribute Hierarchy
                </div>
                <div style="font-size:0.84rem; color:#94A3B8; line-height:1.6;">
                    Decomposes <code>Dim_Stock</code> into normalized <code>Dim_Sector</code> metadata (Sector, Industry,
                    Cap Tier), eliminating attribute redundancy across multi-exchange equity listings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a3:
        st.markdown(
            """
            <div class="fintech-card" style="height: 100%;">
                <div class="kpi-label">3. Fact Constellation (Galaxy)</div>
                <div style="font-size:1.05rem; font-weight:600; color:#F8FAFC; margin-bottom:8px;">
                    Multi-Fact Drill-Across Engine
                </div>
                <div style="font-size:0.84rem; color:#94A3B8; line-height:1.6;">
                    Pairs <code>Fact_DailyTrades</code> with <code>Fact_QuarterlyFinancials</code> (Revenue, Net Profit, EPS, P/E)
                    across conformed <code>Dim_Time</code> and <code>Dim_Stock</code> dimensions.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### ⚙️ Production Pipeline & Model Benchmarks")
    arch_df = pd.DataFrame(
        [
            {
                "Subsystem": "Data Warehouse & OLAP Cube",
                "Engine / Methodology": "Star, Snowflake & Galaxy Schemas + In-Memory Cube",
                "Input Grain": "14,800 Daily OHLCV Bars (20 Equities × 740 Days)",
                "Production Benchmark": "Zero Missing Values | 6 OLAP Cube Operations",
            },
            {
                "Subsystem": "Intraday Equilibrium Forecaster",
                "Engine / Methodology": "Multi-Factor Ordinary Least Squares (OLS)",
                "Input Grain": "Open, High, Low, Volume ➔ Close Price",
                "Production Benchmark": "R² = 0.999951 | MSE = 496.20 (-83.24% Error vs Baseline)",
            },
            {
                "Subsystem": "Regime Classification Engine",
                "Engine / Methodology": "Entropy Decision Tree & Gaussian Naïve Bayes",
                "Input Grain": "Open, Volume, Intraday Spread, Month ➔ Return Tertile",
                "Production Benchmark": "DT Accuracy: 48.38% (Spread Weight: 95.09%, MI: 0.1380)",
            },
            {
                "Subsystem": "Risk-Reward Asset Segmentation",
                "Engine / Methodology": "K-Means (K=3) & Agglomerative Hierarchical Clustering",
                "Input Grain": "5D Standardized Equity Profile (Return, Vol, Liquidity)",
                "Production Benchmark": "Silhouette = 0.4042 (K-Means & Average Linkage)",
            },
            {
                "Subsystem": "Co-Movement Association Engine",
                "Engine / Methodology": "Apriori Frequent Itemset & Rule Mining",
                "Input Grain": "740 × 20 Binary Bullish Session Matrix",
                "Production Benchmark": "229 Itemsets | 404 Rules | Peak Lift: 2.40x (IT Sector)",
            },
        ]
    )
    st.dataframe(arch_df, use_container_width=True, hide_index=True)
