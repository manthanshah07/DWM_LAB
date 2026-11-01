"""Date-clustered moving-block bootstrap for paired classifier comparisons.

Rows are first collapsed to one paired accuracy difference per trading date,
which keeps all same-date stocks together. Consecutive dates are then resampled
in moving blocks, retaining short-range serial dependence across dates.
"""
from __future__ import annotations

from math import ceil
from typing import Iterable

import numpy as np
import pandas as pd


def _moving_block_indices(n_dates: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n_dates indices by concatenating contiguous, non-circular blocks."""
    if n_dates < 2:
        raise ValueError("At least two distinct dates are required")
    if block_length < 1 or block_length > n_dates:
        raise ValueError("block_length must be between 1 and n_dates")
    n_blocks = ceil(n_dates / block_length)
    starts = rng.integers(0, n_dates - block_length + 1, size=n_blocks)
    indices = np.concatenate([np.arange(start, start + block_length) for start in starts])
    return indices[:n_dates]


def paired_moving_block_bootstrap(
    y_true: Iterable,
    y_pred_reference: Iterable,
    y_pred_candidate: Iterable,
    dates: Iterable,
    block_lengths: tuple[int, ...] = (3, 5, 10),
    n_resamples: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare two classifiers using a paired, date-clustered moving-block bootstrap.

    Returns one row per block length with the observed accuracy difference in
    percentage points, percentile 95% CI and a two-sided null-centred bootstrap
    p-value. Positive differences favor the candidate model.

    Important assumptions: the ordered date-level difference series is locally
    stationary over the evaluation window, and the chosen block lengths capture
    meaningful short-range dependence. The output is a sensitivity analysis,
    not a universal significance certificate.
    """
    y = np.asarray(list(y_true), dtype=object)
    pred_ref = np.asarray(list(y_pred_reference), dtype=object)
    pred_cand = np.asarray(list(y_pred_candidate), dtype=object)
    date_values = pd.to_datetime(pd.Series(list(dates)), errors="coerce")

    n = len(y)
    if n == 0 or len(pred_ref) != n or len(pred_cand) != n or len(date_values) != n:
        raise ValueError("y_true, both prediction arrays, and dates must have the same non-zero length")
    if date_values.isna().any():
        raise ValueError("dates contains missing or invalid values")
    if n_resamples < 100:
        raise ValueError("n_resamples must be at least 100")
    if not block_lengths or any(not isinstance(x, (int, np.integer)) for x in block_lengths):
        raise ValueError("block_lengths must be a non-empty tuple of integers")

    row_frame = pd.DataFrame({
        "date": date_values,
        "reference_correct": (pred_ref == y).astype(float),
        "candidate_correct": (pred_cand == y).astype(float),
    })
    row_frame["delta"] = row_frame["candidate_correct"] - row_frame["reference_correct"]
    daily = row_frame.groupby("date", sort=True, as_index=False).agg(
        delta=("delta", "mean"),
        stock_rows=("delta", "size"),
    )
    daily_delta = daily["delta"].to_numpy(dtype=float)
    n_dates = len(daily_delta)
    observed = float(daily_delta.mean())

    if n_dates < 20:
        raise ValueError("At least 20 trading dates are required for block-bootstrap inference")

    results: list[dict] = []
    for block_length in block_lengths:
        if block_length < 1 or block_length > n_dates:
            raise ValueError(f"block length {block_length} is invalid for {n_dates} dates")
        rng = np.random.default_rng(random_state + int(block_length))
        boot_means = np.empty(n_resamples, dtype=float)
        null_centered = daily_delta - observed

        for b in range(n_resamples):
            sample_idx = _moving_block_indices(n_dates, int(block_length), rng)
            boot_means[b] = daily_delta[sample_idx].mean()

        null_means = np.empty(n_resamples, dtype=float)
        for b in range(n_resamples):
            sample_idx = _moving_block_indices(n_dates, int(block_length), rng)
            null_means[b] = null_centered[sample_idx].mean()

        p_value = (1 + int(np.count_nonzero(np.abs(null_means) >= abs(observed)))) / (n_resamples + 1)
        ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975])
        results.append({
            "Block Length (trading days)": int(block_length),
            "Trading-Date Clusters": int(n_dates),
            "Rows": int(n),
            "Reference Accuracy (%)": float(row_frame["reference_correct"].mean() * 100),
            "Candidate Accuracy (%)": float(row_frame["candidate_correct"].mean() * 100),
            "Accuracy Difference (pp)": observed * 100,
            "95% CI Lower (pp)": float(ci_low * 100),
            "95% CI Upper (pp)": float(ci_high * 100),
            "Two-Sided Bootstrap p-value": float(p_value),
            "Bootstrap Replicates": int(n_resamples),
        })

    return pd.DataFrame(results)


def render_streamlit_bootstrap_result(result: pd.DataFrame) -> None:
    """Display an already-computed bootstrap sensitivity table in Streamlit."""
    import streamlit as st

    st.subheader("Paired Moving-Block Bootstrap")
    st.caption(
        "Stocks are paired within each date; contiguous trading-date blocks are resampled "
        "to retain short-range temporal dependence. Compare the 3-, 5- and 10-session results."
    )
    st.dataframe(result.round(4), width="stretch", hide_index=True)
    if not result.empty:
        crosses_zero = (result["95% CI Lower (pp)"] <= 0) & (result["95% CI Upper (pp)"] >= 0)
        if crosses_zero.all():
            st.info("All reported block-length confidence intervals include zero; evidence of a positive accuracy gain is inconclusive.")
        elif (~crosses_zero).all() and (result["95% CI Lower (pp)"] > 0).all():
            st.success("All reported block-length intervals are above zero. Interpret alongside the modelling and bootstrap assumptions.")
        else:
            st.warning("Inference is sensitive to block length. Report this sensitivity rather than selecting only the most favorable result.")
