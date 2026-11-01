"""Dataset quality checks and an optional Streamlit panel for QuantPulse.

The core audit has no Streamlit dependency, so it can run in CI and from the CLI.
Only the presentation helper imports Streamlit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "Date", "Ticker", "Sector", "Exchange", "Open_Price", "High_Price",
    "Low_Price", "Close_Price", "Volume", "Daily_Return",
}
PRICE_COLUMNS = ["Open_Price", "High_Price", "Low_Price", "Close_Price"]


@dataclass(frozen=True)
class AuditResult:
    checks: pd.DataFrame
    summary: dict[str, Any]
    fingerprint_sha256: str | None

    @property
    def passed(self) -> bool:
        return bool(not self.checks.empty and (self.checks["Status"] != "FAIL").all())

    @property
    def failed_count(self) -> int:
        return int((self.checks["Status"] == "FAIL").sum())

    @property
    def warning_count(self) -> int:
        return int((self.checks["Status"] == "WARN").sum())


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return None
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(rows: list[dict[str, Any]], check: str, status: str,
            observed: Any, details: str, severity: str) -> None:
    rows.append({
        "Check": check,
        "Status": status,
        "Observed": observed,
        "Details": details,
        "Severity": severity,
    })


def audit_stock_data(data: pd.DataFrame, source_path: str | Path | None = None) -> AuditResult:
    """Audit the raw QuantPulse OHLCV panel without mutating the input frame.

    PASS/WARN/FAIL are evidence checks, not a claim that the supplied prices are
    authentic exchange records. Provenance must be documented separately.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    frame = data.copy(deep=False)
    rows: list[dict[str, Any]] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        _record(rows, "Required schema", "FAIL", missing_columns,
                "Required columns are missing; remaining checks cannot be trusted.", "Critical")
        checks = pd.DataFrame(rows)
        return AuditResult(checks, {"rows": int(len(frame)), "columns": int(len(frame.columns))},
                           _file_sha256(source_path))

    _record(rows, "Required schema", "PASS", f"{len(REQUIRED_COLUMNS)} required columns",
            "All required OHLCV and classification columns are present.", "None")

    missing = int(frame[list(REQUIRED_COLUMNS)].isna().sum().sum())
    _record(rows, "Missing required values", "PASS" if missing == 0 else "FAIL", missing,
            "No nulls in required fields." if missing == 0 else "Resolve missing values before modelling or loading facts.",
            "None" if missing == 0 else "High")

    date_values = pd.to_datetime(frame["Date"], errors="coerce")
    invalid_dates = int(date_values.isna().sum())
    _record(rows, "Date parseability", "PASS" if invalid_dates == 0 else "FAIL", invalid_dates,
            "All Date values parse as timestamps." if invalid_dates == 0 else "Invalid dates cannot be placed in the warehouse time dimension.",
            "None" if invalid_dates == 0 else "High")

    numeric = frame[PRICE_COLUMNS + ["Volume", "Daily_Return"]].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    non_finite_count = int((~finite).sum())
    _record(rows, "Finite numeric measures", "PASS" if non_finite_count == 0 else "FAIL",
            non_finite_count,
            "All prices, volume and returns are finite numeric values." if non_finite_count == 0 else "Non-finite or non-numeric measure values were found.",
            "None" if non_finite_count == 0 else "High")

    positive_prices = (numeric[PRICE_COLUMNS] > 0).all(axis=1)
    invalid_price_count = int((~positive_prices).sum())
    _record(rows, "Positive OHLC prices", "PASS" if invalid_price_count == 0 else "FAIL",
            invalid_price_count,
            "All OHLC prices are positive." if invalid_price_count == 0 else "OHLC prices must be strictly positive.",
            "None" if invalid_price_count == 0 else "High")

    high_floor = numeric[["Open_Price", "Low_Price", "Close_Price"]].max(axis=1)
    low_ceiling = numeric[["Open_Price", "High_Price", "Close_Price"]].min(axis=1)
    invalid_ohlc = (numeric["High_Price"] < high_floor) | (numeric["Low_Price"] > low_ceiling)
    invalid_ohlc_count = int(invalid_ohlc.sum())
    _record(rows, "OHLC envelope", "PASS" if invalid_ohlc_count == 0 else "FAIL",
            invalid_ohlc_count,
            "High encloses Open/Close and Low encloses Open/Close." if invalid_ohlc_count == 0 else "At least one bar violates OHLC price ordering.",
            "None" if invalid_ohlc_count == 0 else "High")

    invalid_volume = int((numeric["Volume"] < 0).sum())
    _record(rows, "Non-negative volume", "PASS" if invalid_volume == 0 else "FAIL", invalid_volume,
            "Volume is non-negative." if invalid_volume == 0 else "Negative traded volume is invalid.",
            "None" if invalid_volume == 0 else "High")

    business_key_frame = pd.DataFrame({"_audit_date": date_values, "_audit_ticker": frame["Ticker"]})
    business_key_duplicates = int(business_key_frame.duplicated(["_audit_date", "_audit_ticker"]).sum())
    _record(rows, "Date-ticker uniqueness", "PASS" if business_key_duplicates == 0 else "FAIL",
            business_key_duplicates,
            "One row at most per ticker per date." if business_key_duplicates == 0 else "Duplicate fact grain: Date + Ticker.",
            "None" if business_key_duplicates == 0 else "Critical")

    exact_duplicates = int(frame.duplicated().sum())
    _record(rows, "Exact duplicate rows", "PASS" if exact_duplicates == 0 else "WARN", exact_duplicates,
            "No exact duplicate records." if exact_duplicates == 0 else "Exact duplicate records found; inspect before ETL.",
            "None" if exact_duplicates == 0 else "Medium")

    ticker_sector_counts = frame.groupby("Ticker", dropna=False)["Sector"].nunique(dropna=False)
    multi_sector_tickers = int((ticker_sector_counts > 1).sum())
    _record(rows, "Ticker-to-sector consistency", "PASS" if multi_sector_tickers == 0 else "FAIL",
            multi_sector_tickers,
            "Each ticker maps to exactly one sector." if multi_sector_tickers == 0 else "A ticker maps to multiple sectors; dimension key assignment would be ambiguous.",
            "None" if multi_sector_tickers == 0 else "High")

    row_counts = frame.assign(_audit_date=date_values).groupby("_audit_date", dropna=True).size()
    ticker_counts = frame.assign(_audit_date=date_values).groupby("_audit_date", dropna=True)["Ticker"].nunique()
    expected_tickers = int(frame["Ticker"].nunique())
    balanced_dates = bool(len(row_counts) and (row_counts == expected_tickers).all()
                          and (ticker_counts == expected_tickers).all())
    incomplete_dates = int((~((row_counts == expected_tickers) & (ticker_counts == expected_tickers))).sum())
    _record(rows, "Complete date panel", "PASS" if balanced_dates else "WARN",
            f"{len(row_counts)} dates; {incomplete_dates} incomplete",
            f"Each date contains exactly {expected_tickers} unique tickers." if balanced_dates else "Some dates do not contain one row for every ticker; assess whether missing sessions are expected.",
            "None" if balanced_dates else "Medium")

    ticker_session_counts = frame.assign(_audit_date=date_values).groupby("Ticker")["_audit_date"].nunique()
    equal_ticker_history = bool(len(ticker_session_counts) and ticker_session_counts.nunique() == 1)
    _record(rows, "Ticker history balance", "PASS" if equal_ticker_history else "WARN",
            f"{int(ticker_session_counts.min())}–{int(ticker_session_counts.max())} sessions" if len(ticker_session_counts) else "No valid ticker history",
            "All tickers have the same number of dates." if equal_ticker_history else "Ticker histories have different lengths; check listing dates, suspensions and missing rows.",
            "None" if equal_ticker_history else "Medium")

    expected_intraday = ((numeric["Close_Price"] - numeric["Open_Price"]) / numeric["Open_Price"] * 100).round(2)
    return_matches = int((expected_intraday == numeric["Daily_Return"].round(2)).sum())
    return_match_ratio = return_matches / max(len(frame), 1)
    _record(rows, "Daily_Return definition", "PASS" if return_matches == len(frame) else "WARN",
            f"{return_matches:,}/{len(frame):,} match ({return_match_ratio:.2%})",
            "Daily_Return equals rounded Open-to-Close percentage return." if return_matches == len(frame) else "Daily_Return does not consistently match the expected intraday Open-to-Close formula; document its actual definition.",
            "None" if return_matches == len(frame) else "Medium")

    exchanges = sorted(frame["Exchange"].dropna().astype(str).unique().tolist())
    _record(rows, "Exchange coverage", "PASS" if len(exchanges) >= 1 else "FAIL",
            exchanges, "Observed exchanges are listed; one exchange is sufficient for the current scope." if len(exchanges) else "No exchange labels found.",
            "None" if exchanges else "High")

    provenance_candidates = [c for c in ["Source", "Source_URL", "Provider", "Retrieved_At", "Adjustment_Basis"] if c in frame.columns]
    provenance_columns = [c for c in provenance_candidates if frame[c].notna().astype(bool).any()]
    provenance_documented = bool(provenance_columns)
    _record(rows, "Source provenance metadata", "PASS" if provenance_documented else "WARN",
            provenance_columns if provenance_documented else "No populated provenance columns / sidecar supplied",
            "At least one provenance field contains values; verify that the provider and adjustment basis are fully documented." if provenance_documented else "The CSV alone does not establish where prices came from, retrieval time, or whether prices are adjusted. Add a source citation and extraction metadata to project documentation.",
            "None" if provenance_documented else "Medium")

    sha = _file_sha256(source_path)
    if source_path is not None:
        _record(rows, "Source file fingerprint", "PASS" if sha else "WARN",
                sha if sha else "File unavailable",
                "SHA-256 computed from the source file bytes." if sha else "No readable source file at the supplied path.",
                "None" if sha else "Low")

    checks = pd.DataFrame(rows)
    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "dates": int(date_values.nunique()),
        "tickers": int(frame["Ticker"].nunique()),
        "sectors": int(frame["Sector"].nunique()),
        "exchanges": exchanges,
        "date_min": str(date_values.min().date()) if date_values.notna().any() else None,
        "date_max": str(date_values.max().date()) if date_values.notna().any() else None,
        "daily_return_formula_match_pct": round(return_match_ratio * 100, 4),
        "business_key_duplicates": business_key_duplicates,
        "missing_required_values": missing,
        "complete_date_panel": balanced_dates,
    }
    return AuditResult(checks=checks, summary=summary, fingerprint_sha256=sha)


def render_streamlit_quality_panel(data: pd.DataFrame, source_path: str | Path | None = None) -> AuditResult:
    """Render a compact, read-only quality panel inside a Streamlit tab."""
    import streamlit as st

    result = audit_stock_data(data, source_path=source_path)
    summary = result.summary
    st.subheader("Dataset Quality & Provenance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{summary.get('rows', 0):,}")
    c2.metric("Trading Dates", f"{summary.get('dates', 0):,}")
    c3.metric("Equities", f"{summary.get('tickers', 0):,}")
    c4.metric("Failed Checks", str(result.failed_count))

    if result.failed_count:
        st.error(f"{result.failed_count} critical data-quality check(s) failed.")
    elif result.warning_count:
        st.warning(f"No hard failures; {result.warning_count} documentation or completeness warning(s) remain.")
    else:
        st.success("All configured data-quality checks passed.")

    checks_display = result.checks.copy()
    checks_display["Observed"] = checks_display["Observed"].astype(str)
    st.dataframe(checks_display, use_container_width=True, hide_index=True)
    if result.fingerprint_sha256:
        with st.expander("Source fingerprint and dataset scope"):
            st.caption("Fingerprint identifies file bytes; it does not prove source authenticity.")
            st.code(result.fingerprint_sha256, language="text")
            st.json(summary)
    return result
