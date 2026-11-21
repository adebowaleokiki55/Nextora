"""Leak-free feature construction for 1-4 week ahead case forecasts.

A row is (location, origin week t, horizon h): features use only data up to
week t; the target is confirmed cases in week t + h.

Seasonal quantities for a target week T (alert threshold, historical
weekly mean, seasonal naive) only use weeks T - 52k (+/- 2), k >= 1. For
h <= 4 all of those are at or before T - 50 < t, so they never peek ahead.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from backend.app.core.surveillance.metrics import CASES, HUMIDITY, RAINFALL, TEMP_MEAN

MAX_YEARS_BACK = 15
THRESHOLD_WINDOW = 2  # +/- weeks around the same week in earlier years
THRESHOLD_SD = 2.0  # classic "mean + 2 SD" malaria epidemic threshold
MIN_HISTORY_VALUES = 8  # need ~2 years before a threshold is trusted

RAIN_LAGS = 10

FEATURES = [
    "lc_0", "lc_1", "lc_2", "lc_3",
    "lc_mean_4", "lc_mean_8", "lc_mean_13",
    "lc_level",
    "seasonal_naive_log", "hist_mean_log", "threshold_log",
    *[f"rain_l{k}" for k in range(RAIN_LAGS)],
    "rain_sum_4", "rain_sum_8",
    "temp_mean_4", "rh_mean_4",
    "woy_sin", "woy_cos",
]

FEATURE_GROUPS = {
    "Recent case levels": ["lc_0", "lc_1", "lc_2", "lc_3", "lc_mean_4", "lc_mean_8", "lc_mean_13", "lc_level"],
    "Seasonal pattern": ["seasonal_naive_log", "hist_mean_log", "threshold_log", "woy_sin", "woy_cos"],
    "Rainfall": [f"rain_l{k}" for k in range(RAIN_LAGS)] + ["rain_sum_4", "rain_sum_8"],
    "Temperature and humidity": ["temp_mean_4", "rh_mean_4"],
}


def regular_weekly(wide: pd.DataFrame, extra_weeks: int = 0) -> pd.DataFrame:
    """Reindex to a gap-free weekly index so shifts mean 'weeks'.

    `extra_weeks` empty weeks are appended after the last week, so future
    target weeks still get their seasonal references (threshold, means).
    """
    end = pd.Timestamp(wide.index.max()) + pd.Timedelta(weeks=extra_weeks)
    index = pd.date_range(wide.index.min(), end, freq="7D")
    frame = wide.copy()
    frame.index = pd.to_datetime(frame.index)
    return frame.reindex(index)


def _seasonal_stack(cases: pd.Series, window: int) -> np.ndarray:
    """values[i, j]: cases at week i - 52k + offset for each (k, offset)."""
    columns = []
    for k in range(1, MAX_YEARS_BACK + 1):
        for offset in range(-window, window + 1):
            columns.append(cases.shift(52 * k - offset).to_numpy())
    return np.column_stack(columns)


def seasonal_reference(cases: pd.Series) -> pd.DataFrame:
    """Per target week: alert threshold, historical weekly mean, seasonal naive."""
    stack = _seasonal_stack(cases, THRESHOLD_WINDOW)
    counts = np.sum(~np.isnan(stack), axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows early on
        mean = np.nanmean(stack, axis=1)
        sd = np.nanstd(stack, axis=1, ddof=1)
        same_week = _seasonal_stack(cases, 0)
        hist_mean = np.nanmean(same_week, axis=1)

    threshold = mean + THRESHOLD_SD * sd
    threshold[counts < MIN_HISTORY_VALUES] = np.nan
    return pd.DataFrame(
        {
            "threshold": threshold,
            "hist_mean": hist_mean,
            "seasonal_naive": cases.shift(52).to_numpy(),
        },
        index=cases.index,
    )


def location_rows(wide: pd.DataFrame, location_id: int, horizon: int) -> pd.DataFrame:
    """Feature rows for one location and horizon (target may be NaN at the end)."""
    frame = regular_weekly(wide, extra_weeks=horizon)
    cases = frame[CASES] if CASES in frame else pd.Series(np.nan, index=frame.index)
    log_cases = np.log1p(cases)
    rain = frame.get(RAINFALL, pd.Series(np.nan, index=frame.index))
    temp = frame.get(TEMP_MEAN, pd.Series(np.nan, index=frame.index))
    rh = frame.get(HUMIDITY, pd.Series(np.nan, index=frame.index))

    reference = seasonal_reference(cases)
    target_week = frame.index + pd.Timedelta(weeks=horizon)
    woy = target_week.isocalendar().week.to_numpy().astype(float)

    rows = pd.DataFrame(index=frame.index)
    for lag in range(4):
        rows[f"lc_{lag}"] = log_cases.shift(lag)
    for span in (4, 8, 13):
        rows[f"lc_mean_{span}"] = log_cases.rolling(span, min_periods=span).mean()
    rows["lc_level"] = log_cases.expanding(min_periods=26).mean()

    # Seasonal references belong to the target week: shift them back by h.
    rows["seasonal_naive_log"] = np.log1p(reference["seasonal_naive"].shift(-horizon))
    rows["hist_mean_log"] = np.log1p(reference["hist_mean"].shift(-horizon))
    rows["threshold_log"] = np.log1p(reference["threshold"].shift(-horizon))

    for lag in range(RAIN_LAGS):
        rows[f"rain_l{lag}"] = rain.shift(lag)
    rows["rain_sum_4"] = rain.rolling(4, min_periods=4).sum()
    rows["rain_sum_8"] = rain.rolling(8, min_periods=8).sum()
    rows["temp_mean_4"] = temp.rolling(4, min_periods=4).mean()
    rows["rh_mean_4"] = rh.rolling(4, min_periods=4).mean()
    rows["woy_sin"] = np.sin(2 * np.pi * woy / 52.18)
    rows["woy_cos"] = np.cos(2 * np.pi * woy / 52.18)

    # Targets and evaluation references.
    rows["target_cases"] = cases.shift(-horizon)
    rows["target_log"] = np.log1p(rows["target_cases"])
    rows["threshold"] = reference["threshold"].shift(-horizon)
    rows["baseline_persistence"] = cases
    rows["baseline_seasonal_naive"] = reference["seasonal_naive"].shift(-horizon)
    rows["baseline_hist_mean"] = reference["hist_mean"].shift(-horizon)

    rows["location_id"] = location_id
    rows["horizon"] = horizon
    rows["origin_week"] = frame.index
    rows["target_week"] = target_week
    rows = rows[cases.notna().to_numpy()]  # an origin needs this week's count
    return rows.reset_index(drop=True)


def build_rows(wides: dict[int, pd.DataFrame], horizon: int) -> pd.DataFrame:
    frames = [location_rows(wide, location_id, horizon) for location_id, wide in wides.items() if not wide.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
