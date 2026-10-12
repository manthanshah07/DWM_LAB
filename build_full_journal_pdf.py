#!/usr/bin/env python3
"""
build_full_journal_pdf.py
=========================
Builds the complete DWM Lab Journal PDF for Manthan Shah (Roll No: 24101A0078).

Steps:
  1. Generate ER diagram PNGs  (Star, Snowflake, Fact Constellation)
  2. Build lab1_formatted.html and lab2_formatted.html
  3. Convert every experiment HTML → individual PDF (weasyprint)
  4. Merge all PDFs → DWM_Lab_Journal_Manthan_Shah.pdf  (pypdf)
  5. Print total page count
"""

import os
import sys
import sqlite3
import textwrap
import math
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "stock_market_data.csv")
OUT_PDF    = os.path.join(BASE_DIR, "DWM_Lab_Journal_Manthan_Shah.pdf")

STAR_PNG   = os.path.join(BASE_DIR, "schema_star.png")
SNOW_PNG   = os.path.join(BASE_DIR, "schema_snowflake.png")
CONST_PNG  = os.path.join(BASE_DIR, "schema_constellation.png")

LAB1_HTML  = os.path.join(BASE_DIR, "lab1_formatted.html")
LAB2_HTML  = os.path.join(BASE_DIR, "lab2_formatted.html")

# Ordered list of HTMLs → PDFs
HTML_ORDER = [
    ("lab1_formatted.html",          "exp01.pdf"),
    ("lab2_formatted.html",          "exp02.pdf"),
    ("DWM_Lab_3_Stock_Market.html",  "exp03.pdf"),
    ("DWM_Lab_4_Stock_Market.html",  "exp04.pdf"),
    ("DWM_Lab_5_Stock_Market.html",  "exp05.pdf"),
    ("DWM_Lab_6_Stock_Market.html",  "exp06.pdf"),
    ("DWM_Lab_7_Stock_Market.html",  "exp07.pdf"),
    ("DWM_Lab_8_Stock_Market.html",  "exp08.pdf"),
    ("DWM_Lab_9_Stock_Market.html",  "exp09.pdf"),
]

# ---------------------------------------------------------------------------
# Student / Institute metadata
# ---------------------------------------------------------------------------
META = dict(
    institute   = "Vidyalankar Institute of Technology (VIT)",
    address     = "Wadala (W), Mumbai – 400 037",
    naac        = "Accredited A+ by NAAC",
    dept        = "Department of Information Technology",
    subject     = "Data Warehousing and Mining (DWM – PCIT10T / PCIT10P)",
    sem         = "T.E. Semester V  |  TEINFT - A  (Odd Sem 2026-2027)",
    batch       = "Batch - 3",
    prof        = "Dr. Varsha Bhosale",
    s1_name     = "Manthan Shah",
    s1_roll     = "24101A0078",
    s2_name     = "Ishan Gurav",
    s2_roll     = "24101A0077",
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_HEADER   = "#1a237e"   # dark indigo
C_FACT     = "#b71c1c"   # crimson
C_DIM      = "#1565c0"   # royal blue
C_DIM2     = "#2e7d32"   # green (snowflake extra level)
C_LIGHT    = "#e3f2fd"
C_WHITE    = "#ffffff"
C_GRAY     = "#f5f5f5"

# ===========================================================================
# STEP 1 – ER Diagram generation
# ===========================================================================

def _draw_entity(ax, cx, cy, title, attrs, color, width=2.6, row_h=0.38):
    """Draw a UML-style entity box centered at (cx, cy)."""
    n    = len(attrs)
    h    = row_h * (n + 1)
    x0   = cx - width / 2
    y0   = cy - h / 2

    # Shadow
    shadow = FancyBboxPatch((x0 + 0.04, y0 - 0.04), width, h,
                             boxstyle="round,pad=0.05", linewidth=0,
                             facecolor="#bdbdbd", zorder=1)
    ax.add_patch(shadow)

    # Body
    body = FancyBboxPatch((x0, y0), width, h,
                           boxstyle="round,pad=0.05", linewidth=1.5,
                           edgecolor=color, facecolor=C_WHITE, zorder=2)
    ax.add_patch(body)

    # Header bar
    hbar = FancyBboxPatch((x0, y0 + h - row_h), width, row_h,
                           boxstyle="round,pad=0.05", linewidth=0,
                           facecolor=color, zorder=3)
    ax.add_patch(hbar)

    # Title
    ax.text(cx, y0 + h - row_h / 2, title,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=C_WHITE, zorder=4)

    # Attributes
    for i, attr in enumerate(attrs):
        ty = y0 + h - row_h * (i + 1) - row_h / 2
        is_pk = attr.startswith("🔑")
        is_fk = attr.startswith("🔗")
        style = "bold" if is_pk else "normal"
        ax.text(x0 + 0.15, ty, attr, ha="left", va="center",
                fontsize=6.5, fontstyle="italic" if is_fk else "normal",
                fontweight=style, color="#212121", zorder=4)
        if i < n - 1:
            ax.plot([x0, x0 + width], [ty - row_h / 2, ty - row_h / 2],
                    color="#e0e0e0", linewidth=0.5, zorder=3)

    return {"cx": cx, "cy": cy, "x0": x0, "y0": y0, "w": width, "h": h}


def _draw_line(ax, e1, e2, label="", color="#555555"):
    """Draw a connector between two entity boxes."""
    ax.annotate("", xy=(e2["cx"], e2["y0"] + e2["h"] / 2),
                xytext=(e1["cx"], e1["y0"] + e1["h"] / 2),
                arrowprops=dict(arrowstyle="-", color=color, lw=1.2,
                                connectionstyle="arc3,rad=0.0"))
    if label:
        mx = (e1["cx"] + e2["cx"]) / 2
        my = (e1["cy"] + e2["cy"]) / 2
        ax.text(mx, my + 0.12, label, ha="center", fontsize=6,
                color=color, backgroundcolor=C_WHITE)


def generate_star_schema():
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14); ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_facecolor(C_GRAY)
    fig.patch.set_facecolor(C_GRAY)
    ax.set_title("Star Schema — DWM Stock Market Data Warehouse",
                 fontsize=13, fontweight="bold", color=C_HEADER, pad=12)

    fact = _draw_entity(ax, 7, 4.5, "Fact_DailyTrades  (FACT)", [
        "🔑 Trade_ID  (PK)",
        "🔗 Time_Key  (FK)",
        "🔗 Stock_Key (FK)",
        "🔗 Exchange_Key (FK)",
        "Open_Price  REAL",
        "High_Price  REAL",
        "Low_Price   REAL",
        "Close_Price REAL",
        "Volume      INTEGER",
        "Daily_Return REAL",
        "Intraday_Volatility REAL",
    ], C_FACT, width=3.0)

    dim_time = _draw_entity(ax, 2.2, 7.5, "Dim_Time  (DIMENSION)", [
        "🔑 Time_Key (PK)",
        "Date        DATE",
        "Day         INTEGER",
        "Month       INTEGER",
        "Quarter     INTEGER",
        "Year        INTEGER",
        "Day_of_Week TEXT",
        "Is_Weekend  BOOLEAN",
    ], C_DIM)

    dim_stock = _draw_entity(ax, 7, 1.2, "Dim_Stock  (DIMENSION)", [
        "🔑 Stock_Key (PK)",
        "Ticker      TEXT",
        "Company     TEXT",
        "Sector      TEXT",
        "Industry    TEXT",
        "Market_Cap  TEXT",
    ], C_DIM)

    dim_exch = _draw_entity(ax, 11.8, 7.5, "Dim_Exchange  (DIMENSION)", [
        "🔑 Exchange_Key (PK)",
        "Exchange    TEXT",
        "Country     TEXT",
        "Currency    TEXT",
        "Time_Zone   TEXT",
    ], C_DIM)

    # Connectors
    for dim in [dim_time, dim_stock, dim_exch]:
        ax.plot([fact["cx"], dim["cx"]], [fact["cy"], dim["cy"]],
                color="#9e9e9e", lw=1.2, zorder=1)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=C_FACT, label="Fact Table"),
        mpatches.Patch(facecolor=C_DIM, label="Dimension Table"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig(STAR_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  schema_star.png saved")


def generate_snowflake_schema():
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(C_GRAY)
    fig.patch.set_facecolor(C_GRAY)
    ax.set_title("Snowflake Schema — DWM Stock Market Data Warehouse",
                 fontsize=13, fontweight="bold", color=C_HEADER, pad=12)

    fact = _draw_entity(ax, 7, 5, "Fact_DailyTrades_Snowflake  (FACT)", [
        "🔑 Trade_ID   (PK)",
        "🔗 Time_Key   (FK)",
        "🔗 Stock_Key  (FK)",
        "🔗 Exchange_Key (FK)",
        "Open_Price   REAL",
        "Close_Price  REAL",
        "Volume       INTEGER",
        "Daily_Return REAL",
        "Intraday_Volatility REAL",
    ], C_FACT, width=3.2)

    dim_time = _draw_entity(ax, 2, 7.5, "Dim_Time  (DIMENSION)", [
        "🔑 Time_Key (PK)",
        "Date   DATE",
        "Day    INTEGER",
        "Month  INTEGER",
        "Quarter INTEGER",
        "Year   INTEGER",
    ], C_DIM)

    dim_stock_snow = _draw_entity(ax, 7, 8.5, "Dim_Stock_Snowflake  (DIMENSION)", [
        "🔑 Stock_Key  (PK)",
        "🔗 Sector_Key (FK)",
        "Ticker   TEXT",
        "Company  TEXT",
    ], C_DIM)

    dim_sector = _draw_entity(ax, 11, 8.5, "Dim_Sector  (SUB-DIMENSION)", [
        "🔑 Sector_Key (PK)",
        "Sector_Name  TEXT",
        "Industry     TEXT",
        "Market_Cap   TEXT",
    ], C_DIM2)

    dim_exch = _draw_entity(ax, 7, 1.5, "Dim_Exchange  (DIMENSION)", [
        "🔑 Exchange_Key (PK)",
        "Exchange  TEXT",
        "Country   TEXT",
        "Currency  TEXT",
    ], C_DIM)

    for dim in [dim_time, dim_stock_snow, dim_exch]:
        ax.plot([fact["cx"], dim["cx"]], [fact["cy"], dim["cy"]],
                color="#9e9e9e", lw=1.2, zorder=1)
    ax.plot([dim_stock_snow["cx"], dim_sector["cx"]],
            [dim_stock_snow["cy"], dim_sector["cy"]],
            color="#a5d6a7", lw=1.2, zorder=1, linestyle="--")

    legend_elements = [
        mpatches.Patch(facecolor=C_FACT,  label="Fact Table"),
        mpatches.Patch(facecolor=C_DIM,   label="Dimension Table"),
        mpatches.Patch(facecolor=C_DIM2,  label="Sub-Dimension Table"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(SNOW_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  schema_snowflake.png saved")


def generate_constellation_schema():
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 15); ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_facecolor(C_GRAY)
    fig.patch.set_facecolor(C_GRAY)
    ax.set_title("Fact Constellation (Galaxy) Schema — DWM Stock Market Data Warehouse",
                 fontsize=12, fontweight="bold", color=C_HEADER, pad=12)

    fact1 = _draw_entity(ax, 4, 5.5, "Fact_DailyTrades  (FACT)", [
        "🔑 Trade_ID  (PK)",
        "🔗 Time_Key  (FK)",
        "🔗 Stock_Key (FK)",
        "🔗 Exchange_Key (FK)",
        "Open_Price  REAL",
        "Close_Price REAL",
        "Volume      INTEGER",
        "Daily_Return REAL",
    ], C_FACT, width=2.9)

    fact2 = _draw_entity(ax, 11, 5.5, "Fact_QuarterlyFinancials  (FACT)", [
        "🔑 QF_ID     (PK)",
        "🔗 Time_Key  (FK)",
        "🔗 Stock_Key (FK)",
        "Revenue_Cr  REAL",
        "Net_Profit_Cr REAL",
        "EPS         REAL",
        "PE_Ratio    REAL",
    ], "#6a1b9a", width=2.9)

    dim_time = _draw_entity(ax, 7.5, 8.5, "Dim_Time  (CONFORMED)", [
        "🔑 Time_Key  (PK)",
        "Date   DATE",
        "Month  INTEGER",
        "Quarter INTEGER",
        "Year   INTEGER",
    ], C_DIM)

    dim_stock = _draw_entity(ax, 7.5, 2, "Dim_Stock  (CONFORMED)", [
        "🔑 Stock_Key (PK)",
        "Ticker  TEXT",
        "Company TEXT",
        "Sector  TEXT",
    ], C_DIM)

    dim_exch = _draw_entity(ax, 1, 2, "Dim_Exchange  (DIMENSION)", [
        "🔑 Exchange_Key (PK)",
        "Exchange TEXT",
        "Country  TEXT",
        "Currency TEXT",
    ], C_DIM)

    # Fact1 connections
    for dim in [dim_time, dim_stock, dim_exch]:
        ax.plot([fact1["cx"], dim["cx"]], [fact1["cy"], dim["cy"]],
                color="#ef9a9a", lw=1.2, zorder=1)
    # Fact2 connections (shared conformed dims)
    for dim in [dim_time, dim_stock]:
        ax.plot([fact2["cx"], dim["cx"]], [fact2["cy"], dim["cy"]],
                color="#ce93d8", lw=1.2, zorder=1, linestyle="--")

    legend_elements = [
        mpatches.Patch(facecolor=C_FACT,   label="Fact_DailyTrades"),
        mpatches.Patch(facecolor="#6a1b9a", label="Fact_QuarterlyFinancials"),
        mpatches.Patch(facecolor=C_DIM,    label="Conformed Dimension"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(CONST_PNG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅  schema_constellation.png saved")


# ===========================================================================
# STEP 2 – VIT institutional HTML header
# ===========================================================================

VIT_CSS = """
<style>
  * { box-sizing: border-box; }
  body { font-family: 'Times New Roman', serif; margin: 0; padding: 0;
         background: #fff; color: #111; font-size: 12pt; }
  .page-wrapper { padding: 18mm 20mm 15mm 20mm; }

  /* ── Institute header ── */
  .inst-header { text-align: center; border-bottom: 2px solid #1a237e;
                 padding-bottom: 8px; margin-bottom: 12px; }
  .inst-header .inst-name { font-size: 18pt; font-weight: bold;
                             color: #1a237e; letter-spacing: 0.5px; }
  .inst-header .inst-sub  { font-size: 10pt; color: #555; margin-top: 2px; }
  .inst-header .dept-name { font-size: 13pt; font-weight: bold; color: #b71c1c;
                             margin-top: 4px; }

  /* ── Metadata table ── */
  .meta-table { width: 100%; border-collapse: collapse; margin-bottom: 14px; }
  .meta-table td { border: 1px solid #333; padding: 5px 8px; font-size: 10.5pt; }
  .meta-table .label { font-weight: bold; width: 32%; background: #e8eaf6; }

  /* ── Section headings ── */
  h2 { font-size: 13pt; color: #1a237e; border-bottom: 1px solid #1a237e;
       padding-bottom: 3px; margin-top: 18px; }
  h3 { font-size: 11.5pt; color: #b71c1c; margin-top: 14px; }

  /* ── Code blocks ── */
  pre  { background: #f5f5f5; border: 1px solid #ccc; border-left: 4px solid #1a237e;
         padding: 10px 14px; font-family: 'Courier New', monospace; font-size: 9pt;
         overflow-x: auto; white-space: pre-wrap; word-break: break-word; }
  code { font-family: 'Courier New', monospace; font-size: 9pt; }

  /* ── Result tables ── */
  .result-table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 9.5pt; }
  .result-table th { background: #1a237e; color: #fff; padding: 5px 8px; text-align: left; }
  .result-table td { border: 1px solid #bbb; padding: 4px 8px; }
  .result-table tr:nth-child(even) td { background: #f0f4ff; }

  /* ── ER diagram images ── */
  .er-img { display: block; margin: 14px auto; max-width: 100%;
            border: 1px solid #ccc; border-radius: 4px; }

  /* ── Conclusion box ── */
  .conclusion { background: #e8f5e9; border-left: 5px solid #2e7d32;
                padding: 10px 14px; margin-top: 16px; border-radius: 3px; }

  @media print {
    .page-wrapper { padding: 10mm 14mm 10mm 14mm; }
  }
</style>
"""

def _vit_header(exp_no, exp_title, date_str=""):
    m = META
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DWM Lab {exp_no} – {m['s1_name']}</title>
{VIT_CSS}
</head>
<body>
<div class="page-wrapper">

<!-- ── Institute Header ── -->
<div class="inst-header">
  <div class="inst-name">{m['institute']}</div>
  <div class="inst-sub">{m['address']}  |  {m['naac']}</div>
  <div class="dept-name">{m['dept']}</div>
</div>

<!-- ── Metadata Table ── -->
<table class="meta-table">
  <tr>
    <td class="label">Subject</td>
    <td colspan="3">{m['subject']}</td>
  </tr>
  <tr>
    <td class="label">Experiment No.</td>
    <td>{exp_no}</td>
    <td class="label">Semester / Division</td>
    <td>{m['sem']}</td>
  </tr>
  <tr>
    <td class="label">Laboratory Batch</td>
    <td>{m['batch']}</td>
    <td class="label">Date of Performance</td>
    <td>{date_str if date_str else "—"}</td>
  </tr>
  <tr>
    <td class="label">Student 1</td>
    <td>{m['s1_name']}  (Roll No: {m['s1_roll']})</td>
    <td class="label">Student 2</td>
    <td>{m['s2_name']}  (Roll No: {m['s2_roll']})</td>
  </tr>
  <tr>
    <td class="label">Professor In-charge</td>
    <td colspan="3">{m['prof']}</td>
  </tr>
</table>

<h1 style="text-align:center;font-size:14pt;color:#1a237e;margin-bottom:6px;">
  Experiment {exp_no}: {exp_title}
</h1>
"""


def _vit_footer():
    return """
</div><!-- /.page-wrapper -->
</body>
</html>"""


# ===========================================================================
# STEP 2a – Lab 1 HTML: Star / Snowflake / Fact Constellation schemas
# ===========================================================================

LAB1_DDL_STAR = """\
-- ══════════════════════════════════════════════
--  STAR SCHEMA DDL
-- ══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS Dim_Time (
    Time_Key    INTEGER PRIMARY KEY AUTOINCREMENT,
    Date        DATE    NOT NULL UNIQUE,
    Day         INTEGER,
    Month       INTEGER,
    Quarter     INTEGER,
    Year        INTEGER,
    Day_of_Week TEXT,
    Is_Weekend  BOOLEAN
);

CREATE TABLE IF NOT EXISTS Dim_Stock (
    Stock_Key   INTEGER PRIMARY KEY AUTOINCREMENT,
    Ticker      TEXT    NOT NULL UNIQUE,
    Company     TEXT,
    Sector      TEXT,
    Industry    TEXT,
    Market_Cap  TEXT
);

CREATE TABLE IF NOT EXISTS Dim_Exchange (
    Exchange_Key  INTEGER PRIMARY KEY AUTOINCREMENT,
    Exchange      TEXT    NOT NULL UNIQUE,
    Country       TEXT,
    Currency      TEXT,
    Time_Zone     TEXT
);

CREATE TABLE IF NOT EXISTS Fact_DailyTrades (
    Trade_ID            INTEGER PRIMARY KEY AUTOINCREMENT,
    Time_Key            INTEGER REFERENCES Dim_Time(Time_Key),
    Stock_Key           INTEGER REFERENCES Dim_Stock(Stock_Key),
    Exchange_Key        INTEGER REFERENCES Dim_Exchange(Exchange_Key),
    Open_Price          REAL,
    High_Price          REAL,
    Low_Price           REAL,
    Close_Price         REAL,
    Volume              INTEGER,
    Daily_Return        REAL,
    Intraday_Volatility REAL
);
"""

LAB1_DDL_SNOW = """\
-- ══════════════════════════════════════════════
--  SNOWFLAKE SCHEMA DDL
-- ══════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS Dim_Sector (
    Sector_Key  INTEGER PRIMARY KEY AUTOINCREMENT,
    Sector_Name TEXT    NOT NULL UNIQUE,
    Industry    TEXT,
    Market_Cap  TEXT
);

CREATE TABLE IF NOT EXISTS Dim_Stock_Snowflake (
    Stock_Key   INTEGER PRIMARY KEY AUTOINCREMENT,
    Sector_Key  INTEGER REFERENCES Dim_Sector(Sector_Key),
    Ticker      TEXT    NOT NULL UNIQUE,
    Company     TEXT
);

CREATE TABLE IF NOT EXISTS Fact_DailyTrades_Snowflake (
    Trade_ID            INTEGER PRIMARY KEY AUTOINCREMENT,
    Time_Key            INTEGER REFERENCES Dim_Time(Time_Key),
    Stock_Key           INTEGER REFERENCES Dim_Stock_Snowflake(Stock_Key),
    Exchange_Key        INTEGER REFERENCES Dim_Exchange(Exchange_Key),
    Open_Price          REAL,
    Close_Price         REAL,
    Volume              INTEGER,
    Daily_Return        REAL,
    Intraday_Volatility REAL
);
"""

LAB1_DDL_CONST = """\
-- ══════════════════════════════════════════════
--  FACT CONSTELLATION (GALAXY) SCHEMA DDL
-- ══════════════════════════════════════════════

-- Conformed dimensions: Dim_Time, Dim_Stock, Dim_Exchange  (reused from Star)

CREATE TABLE IF NOT EXISTS Fact_QuarterlyFinancials (
    QF_ID         INTEGER PRIMARY KEY AUTOINCREMENT,
    Time_Key      INTEGER REFERENCES Dim_Time(Time_Key),
    Stock_Key     INTEGER REFERENCES Dim_Stock(Stock_Key),
    Revenue_Cr    REAL,
    Net_Profit_Cr REAL,
    EPS           REAL,
    PE_Ratio      REAL
);

-- Fact_DailyTrades already created under Star Schema (shared via conformed dims)
"""

def build_lab1_html():
    star_b64   = _img_b64(STAR_PNG)
    snow_b64   = _img_b64(SNOW_PNG)
    const_b64  = _img_b64(CONST_PNG)

    html = _vit_header("01",
        "Build Data Warehouse / Data Mart &amp; Design Dimensional Data Models")
    html += """
<h2>1. Aim / Objective</h2>
<p>
To design and implement a <strong>Data Warehouse</strong> for the Stock Market dataset using
three dimensional modelling techniques:
<em>Star Schema</em>, <em>Snowflake Schema</em>, and <em>Fact Constellation (Galaxy) Schema</em>.
The warehouse stores daily trade facts (Open, High, Low, Close prices, Volume, Daily Return)
together with conformed time, stock, and exchange dimensions.
</p>

<h2>2. Theory</h2>
<h3>2.1 Data Warehouse</h3>
<p>
A <em>Data Warehouse (DW)</em> is a subject-oriented, integrated, time-variant, and non-volatile
collection of data used to support management decision-making (Inmon, 1992).
It separates analytical workloads from operational databases (OLTP) by storing pre-aggregated,
historical data in a schema optimised for read-heavy OLAP queries.
</p>

<h3>2.2 Star Schema</h3>
<p>
The <strong>Star Schema</strong> is the simplest dimensional model.  A central
<em>Fact table</em> (<code>Fact_DailyTrades</code>) holds quantitative trade metrics
and foreign keys to denormalized <em>Dimension tables</em>
(<code>Dim_Time</code>, <code>Dim_Stock</code>, <code>Dim_Exchange</code>).
Queries involve at most one join per dimension, delivering the fastest analytical performance.
</p>

<h3>2.3 Snowflake Schema</h3>
<p>
The <strong>Snowflake Schema</strong> extends the star by further normalizing dimension tables.
<code>Dim_Stock_Snowflake</code> references a separate <code>Dim_Sector</code> sub-dimension,
eliminating redundant sector/industry storage at the cost of additional joins.
</p>

<h3>2.4 Fact Constellation (Galaxy) Schema</h3>
<p>
The <strong>Fact Constellation</strong> supports multiple fact tables sharing
<em>conformed dimensions</em>.  Here, <code>Fact_DailyTrades</code> and
<code>Fact_QuarterlyFinancials</code> both reference <code>Dim_Time</code> and <code>Dim_Stock</code>,
enabling cross-fact drill-across OLAP queries.
</p>

<h2>3. ER Diagrams</h2>

<h3>3.1 Star Schema</h3>
"""
    html += f'<img class="er-img" src="data:image/png;base64,{star_b64}" alt="Star Schema ER Diagram"/>\n'

    html += "<h3>3.2 Snowflake Schema</h3>\n"
    html += f'<img class="er-img" src="data:image/png;base64,{snow_b64}" alt="Snowflake Schema ER Diagram"/>\n'

    html += "<h3>3.3 Fact Constellation (Galaxy) Schema</h3>\n"
    html += f'<img class="er-img" src="data:image/png;base64,{const_b64}" alt="Fact Constellation ER Diagram"/>\n'

    html += """
<h2>4. DDL Scripts</h2>

<h3>4.1 Star Schema DDL</h3>
<pre>""" + LAB1_DDL_STAR + """</pre>

<h3>4.2 Snowflake Schema DDL</h3>
<pre>""" + LAB1_DDL_SNOW + """</pre>

<h3>4.3 Fact Constellation DDL</h3>
<pre>""" + LAB1_DDL_CONST + """</pre>

<h2>5. Data Loading Summary</h2>
"""
    # Load CSV and show table row counts
    try:
        df = pd.read_csv(CSV_PATH)
        n_rows   = len(df)
        n_dates  = df["Date"].nunique()
        n_stocks = df["Ticker"].nunique()
        n_exch   = df["Exchange"].nunique() if "Exchange" in df.columns else "N/A"
        html += f"""
<table class="result-table">
  <tr><th>Table</th><th>Rows Loaded</th><th>Description</th></tr>
  <tr><td>Dim_Time</td><td>{n_dates:,}</td><td>Unique trading dates</td></tr>
  <tr><td>Dim_Stock</td><td>{n_stocks:,}</td><td>Unique stock tickers</td></tr>
  <tr><td>Dim_Exchange</td><td>{n_exch}</td><td>Unique exchanges</td></tr>
  <tr><td>Fact_DailyTrades</td><td>{n_rows:,}</td><td>Total daily trade records</td></tr>
</table>
"""
    except Exception as e:
        html += f"<p><em>Could not load CSV: {e}</em></p>\n"

    html += """
<div class="conclusion">
  <strong>Conclusion:</strong>
  The Star, Snowflake, and Fact Constellation schemas were successfully designed and implemented
  for the stock market dataset.  The Star Schema offers optimal query performance; the Snowflake
  Schema reduces storage redundancy by normalising sector data; and the Galaxy Schema enables
  multi-fact analytical queries using conformed dimensions shared across daily trades and
  quarterly financials.
</div>
"""
    html += _vit_footer()

    with open(LAB1_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅  lab1_formatted.html saved")


# ===========================================================================
# STEP 2b – Lab 2 HTML: OLAP Queries
# ===========================================================================

OLAP_QUERIES = {
    "Slice": {
        "description": "Extracts a single dimension value — all trades for the NSE exchange.",
        "sql": """SELECT t.Year, s.Ticker, s.Sector,
       ROUND(AVG(f.Close_Price), 2)   AS Avg_Close,
       ROUND(AVG(f.Daily_Return), 4)  AS Avg_Return,
       SUM(f.Volume)                  AS Total_Volume
  FROM Fact_DailyTrades f
  JOIN Dim_Time     t ON f.Time_Key     = t.Time_Key
  JOIN Dim_Stock    s ON f.Stock_Key    = s.Stock_Key
  JOIN Dim_Exchange e ON f.Exchange_Key = e.Exchange_Key
 WHERE e.Exchange = 'NSE'
 GROUP BY t.Year, s.Ticker, s.Sector
 ORDER BY t.Year, s.Ticker
 LIMIT 10;""",
    },
    "Dice": {
        "description": "Applies filters on two+ dimensions — IT sector stocks in 2023 with positive returns.",
        "sql": """SELECT t.Year, t.Quarter, s.Ticker, s.Sector,
       ROUND(AVG(f.Close_Price), 2)   AS Avg_Close,
       ROUND(AVG(f.Daily_Return), 4)  AS Avg_Return,
       SUM(f.Volume)                  AS Total_Volume
  FROM Fact_DailyTrades f
  JOIN Dim_Time  t ON f.Time_Key  = t.Time_Key
  JOIN Dim_Stock s ON f.Stock_Key = s.Stock_Key
 WHERE s.Sector = 'IT'
   AND t.Year   = 2023
   AND f.Daily_Return > 0
 GROUP BY t.Year, t.Quarter, s.Ticker
 ORDER BY t.Quarter, Avg_Return DESC
 LIMIT 10;""",
    },
    "Roll-Up": {
        "description": "Aggregates from Month → Quarter → Year hierarchy.",
        "sql": """SELECT t.Year, t.Quarter,
       ROUND(AVG(f.Close_Price),  2)  AS Avg_Close,
       ROUND(AVG(f.Daily_Return), 4)  AS Avg_Daily_Return,
       SUM(f.Volume)                  AS Total_Volume,
       COUNT(*)                       AS Trade_Days
  FROM Fact_DailyTrades f
  JOIN Dim_Time t ON f.Time_Key = t.Time_Key
 GROUP BY t.Year, t.Quarter
 ORDER BY t.Year, t.Quarter
 LIMIT 12;""",
    },
    "Drill-Down": {
        "description": "Decomposes from Year → Quarter → Month for INFY.",
        "sql": """SELECT t.Year, t.Quarter, t.Month,
       ROUND(AVG(f.Close_Price),  2) AS Avg_Close,
       ROUND(AVG(f.Daily_Return), 4) AS Avg_Return,
       SUM(f.Volume)                 AS Total_Volume
  FROM Fact_DailyTrades f
  JOIN Dim_Time  t ON f.Time_Key  = t.Time_Key
  JOIN Dim_Stock s ON f.Stock_Key = s.Stock_Key
 WHERE s.Ticker = 'INFY'
 GROUP BY t.Year, t.Quarter, t.Month
 ORDER BY t.Year, t.Quarter, t.Month
 LIMIT 12;""",
    },
    "Pivot": {
        "description": "Cross-tabulates average daily return per sector per year.",
        "sql": """SELECT s.Sector,
       ROUND(AVG(CASE WHEN t.Year = 2021 THEN f.Daily_Return END), 4) AS [2021],
       ROUND(AVG(CASE WHEN t.Year = 2022 THEN f.Daily_Return END), 4) AS [2022],
       ROUND(AVG(CASE WHEN t.Year = 2023 THEN f.Daily_Return END), 4) AS [2023],
       ROUND(AVG(CASE WHEN t.Year = 2024 THEN f.Daily_Return END), 4) AS [2024]
  FROM Fact_DailyTrades f
  JOIN Dim_Stock s ON f.Stock_Key = s.Stock_Key
  JOIN Dim_Time  t ON f.Time_Key  = t.Time_Key
 GROUP BY s.Sector
 ORDER BY s.Sector;""",
    },
    "Drill-Across": {
        "description": "Joins two fact tables (DailyTrades + QuarterlyFinancials) via conformed dims.",
        "sql": """SELECT s.Ticker, s.Sector, t.Year, t.Quarter,
       ROUND(AVG(f.Daily_Return), 4) AS Avg_Daily_Return,
       ROUND(AVG(f.Volume),       0) AS Avg_Volume,
       ROUND(AVG(q.EPS),          2) AS Avg_EPS,
       ROUND(AVG(q.PE_Ratio),     2) AS Avg_PE
  FROM Fact_DailyTrades f
  JOIN Dim_Stock s ON f.Stock_Key = s.Stock_Key
  JOIN Dim_Time  t ON f.Time_Key  = t.Time_Key
  JOIN Fact_QuarterlyFinancials q
       ON q.Stock_Key = f.Stock_Key AND q.Time_Key = f.Time_Key
 GROUP BY s.Ticker, t.Year, t.Quarter
 ORDER BY s.Ticker, t.Year, t.Quarter
 LIMIT 10;""",
    },
}


def _run_olap_queries():
    """Re-create the SQLite warehouse in memory and run all OLAP queries."""
    import random, math

    df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
    df["Intraday_Volatility"] = df["High_Price"] - df["Low_Price"]

    con = sqlite3.connect(":memory:")
    cur = con.cursor()

    # ── Dimensions ──────────────────────────────────────────────────────────
    cur.executescript("""
    CREATE TABLE Dim_Time (
        Time_Key INTEGER PRIMARY KEY, Date TEXT, Day INTEGER,
        Month INTEGER, Quarter INTEGER, Year INTEGER,
        Day_of_Week TEXT, Is_Weekend INTEGER);
    CREATE TABLE Dim_Stock (
        Stock_Key INTEGER PRIMARY KEY, Ticker TEXT, Company TEXT,
        Sector TEXT, Industry TEXT, Market_Cap TEXT);
    CREATE TABLE Dim_Exchange (
        Exchange_Key INTEGER PRIMARY KEY, Exchange TEXT,
        Country TEXT, Currency TEXT, Time_Zone TEXT);
    CREATE TABLE Fact_DailyTrades (
        Trade_ID INTEGER PRIMARY KEY, Time_Key INTEGER, Stock_Key INTEGER,
        Exchange_Key INTEGER, Open_Price REAL, High_Price REAL,
        Low_Price REAL, Close_Price REAL, Volume INTEGER,
        Daily_Return REAL, Intraday_Volatility REAL);
    CREATE TABLE Fact_QuarterlyFinancials (
        QF_ID INTEGER PRIMARY KEY, Time_Key INTEGER, Stock_Key INTEGER,
        Revenue_Cr REAL, Net_Profit_Cr REAL, EPS REAL, PE_Ratio REAL);
    """)

    # Dim_Time
    dates = sorted(df["Date"].unique())
    date_key = {}
    for k, d in enumerate(dates, 1):
        pd_d = pd.Timestamp(d)
        cur.execute("INSERT INTO Dim_Time VALUES (?,?,?,?,?,?,?,?)",
            (k, str(pd_d.date()), pd_d.day, pd_d.month,
             (pd_d.month - 1) // 3 + 1, pd_d.year,
             pd_d.day_name(), 1 if pd_d.weekday() >= 5 else 0))
        date_key[d] = k

    # Dim_Stock
    stocks = df[["Ticker","Sector"]].drop_duplicates().reset_index(drop=True)
    stock_key = {}
    for k, row in stocks.iterrows():
        sk = k + 1
        cur.execute("INSERT INTO Dim_Stock VALUES (?,?,?,?,?,?)",
            (sk, row["Ticker"], row["Ticker"] + " Ltd.",
             row["Sector"], row["Sector"] + " Industry", "Large Cap"))
        stock_key[row["Ticker"]] = sk

    # Dim_Exchange
    exchanges = df["Exchange"].unique() if "Exchange" in df.columns else ["NSE"]
    exch_key = {}
    for k, ex in enumerate(exchanges, 1):
        cur.execute("INSERT INTO Dim_Exchange VALUES (?,?,?,?,?)",
            (k, ex, "India", "INR", "Asia/Kolkata"))
        exch_key[ex] = k

    # Fact_DailyTrades
    trade_id = 1
    for _, row in df.iterrows():
        tk = date_key.get(row["Date"], 1)
        sk = stock_key.get(row["Ticker"], 1)
        exch_col = row.get("Exchange", "NSE") if "Exchange" in df.columns else "NSE"
        ek = exch_key.get(exch_col, 1)
        cur.execute(
            "INSERT INTO Fact_DailyTrades VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (trade_id, tk, sk, ek,
             row["Open_Price"], row["High_Price"], row["Low_Price"],
             row["Close_Price"], int(row["Volume"]), row["Daily_Return"],
             row["Intraday_Volatility"]))
        trade_id += 1

    # Fact_QuarterlyFinancials (synthetic, quarterly avg per ticker)
    qf_id = 1
    q_df = df.copy()
    q_df["Quarter"] = pd.to_datetime(q_df["Date"]).dt.to_period("Q")
    for (ticker, q), grp in q_df.groupby(["Ticker", "Quarter"]):
        # Use first date in quarter as time key
        first_date = pd.Timestamp(grp["Date"].min())
        tk = date_key.get(grp["Date"].min(), 1)
        sk = stock_key.get(ticker, 1)
        rev   = round(random.uniform(500, 5000), 2)
        profit= round(rev * random.uniform(0.08, 0.25), 2)
        eps   = round(random.uniform(5, 80), 2)
        pe    = round(random.uniform(10, 50), 2)
        cur.execute("INSERT INTO Fact_QuarterlyFinancials VALUES (?,?,?,?,?,?,?)",
            (qf_id, tk, sk, rev, profit, eps, pe))
        qf_id += 1

    con.commit()

    results = {}
    for op_name, info in OLAP_QUERIES.items():
        try:
            res_df = pd.read_sql_query(info["sql"], con)
            results[op_name] = res_df
        except Exception as e:
            results[op_name] = pd.DataFrame({"Error": [str(e)]})

    con.close()
    return results


def _df_to_html_table(df):
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for _, row in df.iterrows():
        cells = "".join(f"<td>{v}</td>" for v in row)
        rows += f"<tr>{cells}</tr>\n"
    return f"""<table class="result-table">
<thead><tr>{cols}</tr></thead>
<tbody>{rows}</tbody>
</table>"""


def build_lab2_html():
    html = _vit_header("02", "OLAP Operations on the Dimensional Model")
    html += """
<h2>1. Aim / Objective</h2>
<p>
To perform standard OLAP (Online Analytical Processing) operations — <strong>Slice, Dice,
Roll-Up, Drill-Down, Pivot,</strong> and <strong>Drill-Across</strong> — on the Stock Market
Data Warehouse built in Experiment 01, using SQL queries executed against a SQLite in-memory
warehouse populated from <code>stock_market_data.csv</code>.
</p>

<h2>2. Theory</h2>

<h3>2.1 OLAP Concepts</h3>
<p>
OLAP enables multidimensional analysis of business data.  Operations traverse the
<em>data cube</em> defined by the warehouse dimensions (Time, Stock, Exchange).
</p>
<table class="result-table" style="margin-bottom:12px;">
  <thead><tr><th>Operation</th><th>Description</th></tr></thead>
  <tbody>
    <tr><td><strong>Slice</strong></td><td>Selects a single value of one dimension, producing a sub-cube.</td></tr>
    <tr><td><strong>Dice</strong></td><td>Applies range/value filters on two or more dimensions.</td></tr>
    <tr><td><strong>Roll-Up</strong></td><td>Aggregates data up a hierarchy (Month → Quarter → Year).</td></tr>
    <tr><td><strong>Drill-Down</strong></td><td>Decomposes aggregated data to a finer granularity.</td></tr>
    <tr><td><strong>Pivot</strong></td><td>Rotates the data cube to present a cross-tabulated view.</td></tr>
    <tr><td><strong>Drill-Across</strong></td><td>Queries across multiple fact tables sharing conformed dimensions.</td></tr>
  </tbody>
</table>

<h2>3. OLAP Query Results</h2>
"""
    print("    Running OLAP queries against in-memory SQLite warehouse …")
    results = _run_olap_queries()

    for op_name, info in OLAP_QUERIES.items():
        html += f"\n<h3>3.{list(OLAP_QUERIES.keys()).index(op_name)+1}  {op_name} Operation</h3>\n"
        html += f"<p><em>{info['description']}</em></p>\n"
        html += f"<pre>{info['sql']}</pre>\n"
        html += "<p><strong>Result:</strong></p>\n"
        res_df = results.get(op_name, pd.DataFrame())
        if res_df.empty:
            html += "<p><em>No rows returned.</em></p>\n"
        else:
            html += _df_to_html_table(res_df) + "\n"

    html += """
<div class="conclusion">
  <strong>Conclusion:</strong>
  All six OLAP operations were successfully executed on the Stock Market Data Warehouse.
  Slice isolated NSE-listed stocks; Dice filtered IT-sector positive-return days in 2023;
  Roll-Up aggregated trade metrics to the quarterly level; Drill-Down decomposed INFY's
  performance month-by-month; Pivot cross-tabulated sector-wise average returns by year;
  and Drill-Across joined daily trade facts with quarterly financials via conformed dimensions,
  demonstrating the power of a multi-fact constellation schema.
</div>
"""
    html += _vit_footer()

    with open(LAB2_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅  lab2_formatted.html saved")


# ===========================================================================
# Helper – image → base64
# ===========================================================================

def _img_b64(path):
    import base64
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ===========================================================================
# STEP 3 – HTML → PDF  (weasyprint primary, xhtml2pdf fallback)
# ===========================================================================

def _html_to_pdf_weasyprint(html_path, pdf_path):
    from weasyprint import HTML as WP_HTML
    WP_HTML(filename=html_path).write_pdf(pdf_path)


def _html_to_pdf_xhtml2pdf(html_path, pdf_path):
    import re
    from xhtml2pdf import pisa
    with open(html_path, "r", encoding="utf-8") as f:
        html_src = f.read()
    def abs_src(m):
        src = m.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return m.group(0)
        return f'src="{os.path.join(BASE_DIR, src)}"'
    html_src = re.sub(r'src="([^"]+)"', abs_src, html_src)
    with open(pdf_path, "wb") as out_f:
        result = pisa.CreatePDF(html_src, dest=out_f, encoding="utf-8", path=BASE_DIR)
    return not result.err


def convert_html_to_pdfs():
    # Detect primary engine
    try:
        import weasyprint  # noqa: F401
        engine = "weasyprint"
    except Exception:
        engine = "xhtml2pdf"
    print(f"  PDF engine: {engine}")

    pdf_paths = []
    for html_name, pdf_name in HTML_ORDER:
        html_path = os.path.join(BASE_DIR, html_name)
        pdf_path  = os.path.join(BASE_DIR, pdf_name)
        if not os.path.exists(html_path):
            print(f"  ⚠️  MISSING: {html_name} — skipped")
            continue
        print(f"  🔄  {html_name} → {pdf_name} …", end=" ", flush=True)
        success = False
        try:
            if engine == "weasyprint":
                _html_to_pdf_weasyprint(html_path, pdf_path)
            else:
                if not _html_to_pdf_xhtml2pdf(html_path, pdf_path):
                    raise RuntimeError("pisa error")
            success = True
        except Exception as e:
            if engine == "weasyprint":
                print(f"\n     weasyprint err ({type(e).__name__}), retrying with xhtml2pdf …",
                      end=" ", flush=True)
                try:
                    if _html_to_pdf_xhtml2pdf(html_path, pdf_path):
                        success = True
                except Exception as e2:
                    print(f"\n     xhtml2pdf also failed: {e2}")
            else:
                print(f"ERROR — {e}")

        if success and os.path.exists(pdf_path):
            size_kb = os.path.getsize(pdf_path) // 1024
            print(f"done  ({size_kb} KB)")
            pdf_paths.append(pdf_path)
        elif not success:
            print("ERROR — conversion failed")
    return pdf_paths


# ===========================================================================
# STEP 4 – Merge PDFs via pypdf
# ===========================================================================

def merge_pdfs(pdf_paths):
    from pypdf import PdfWriter, PdfReader
    writer = PdfWriter()
    total_pages = 0
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)
        total_pages += len(reader.pages)
        print(f"  📄  {os.path.basename(path)}: {len(reader.pages)} page(s)")

    with open(OUT_PDF, "wb") as f:
        writer.write(f)

    return total_pages


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("\n" + "="*65)
    print("  DWM Lab Journal PDF Builder")
    print("  Student : Manthan Shah  (24101A0078)")
    print("  Subject : Data Warehousing and Mining (DWM)")
    print("="*65 + "\n")

    # ── Step 1 ── ER Diagrams ────────────────────────────────────────────────
    print("STEP 1: Generating ER Diagram PNGs …")
    generate_star_schema()
    generate_snowflake_schema()
    generate_constellation_schema()

    # ── Step 2 ── Lab HTML files ─────────────────────────────────────────────
    print("\nSTEP 2: Building VIT-formatted HTML files …")
    build_lab1_html()
    build_lab2_html()

    # ── Step 3 ── HTML → PDF ─────────────────────────────────────────────────
    print("\nSTEP 3: Converting HTML files to individual PDFs …")
    pdf_paths = convert_html_to_pdfs()

    if not pdf_paths:
        print("\n❌  No PDFs generated. Aborting merge.")
        sys.exit(1)

    # ── Step 4 ── Merge ──────────────────────────────────────────────────────
    print(f"\nSTEP 4: Merging {len(pdf_paths)} PDFs into journal …")
    total_pages = merge_pdfs(pdf_paths)

    # ── Step 5 ── Summary ────────────────────────────────────────────────────
    size_mb = os.path.getsize(OUT_PDF) / (1024 * 1024)
    print("\n" + "="*65)
    print(f"  ✅  Journal PDF created: DWM_Lab_Journal_Manthan_Shah.pdf")
    print(f"  📃  Total pages : {total_pages}")
    print(f"  💾  File size   : {size_mb:.2f} MB")
    print("="*65 + "\n")

    # Clean up individual PDFs
    for path in pdf_paths:
        try:
            os.remove(path)
        except Exception:
            pass


if __name__ == "__main__":
    main()
