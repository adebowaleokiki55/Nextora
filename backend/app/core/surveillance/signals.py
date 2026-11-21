"""Surveillance signals for one location: deterministic comparisons of the
latest weeks against the same time of year in earlier years.

These describe current conditions. Forecasts and risk scores come from the
prediction engine (Module 2), not from here.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backend.app.core.surveillance.metrics import CASES, LAB_POSITIVE, RAINFALL, TEMP_MEAN

STATUS_ORDER = {"normal": 0, "elevated": 1, "high": 2}

# Mean temperature range in which P. falciparum transmission is supported.
TRANSMISSION_TEMP_RANGE = (18.0, 34.0)
TRANSMISSION_TEMP_OPTIMUM = (23.0, 29.0)


def seasonal_values(series: pd.Series, target: date, window_weeks: int = 2) -> list[float]:
    """Values from previous years within +/- window_weeks of the same week.

    `series` is indexed by week_start (Monday dates). 52-week steps keep
    Monday alignment.
    """
    values: list[float] = []
    if series.empty:
        return values
    first = min(series.index)
    years_back = 1
    while True:
        centre = target - timedelta(weeks=52 * years_back)
        if centre < first - timedelta(weeks=window_weeks):
            break
        for offset in range(-window_weeks, window_weeks + 1):
            week = centre + timedelta(weeks=offset)
            value = series.get(week)
            if value is not None and not pd.isna(value):
                values.append(float(value))
        years_back += 1
    return values


def _pct_change(value: float, reference: float) -> float | None:
    if reference is None or reference <= 0 or np.isnan(reference):
        return None
    return round((value / reference - 1) * 100, 1)


def anomaly(value: float, history: list[float]) -> tuple[float | None, float | None, float | None]:
    """(baseline mean, % change, z-score) of `value` against seasonal history."""
    if len(history) < 3:
        return None, None, None
    mean = float(np.mean(history))
    std = float(np.std(history, ddof=1))
    pct = _pct_change(value, mean)
    z = round((value - mean) / std, 2) if std > 0 else None
    return mean, pct, z


def anomaly_status(
    pct: float | None,
    z: float | None,
    excess: float | None = None,
    min_excess: tuple[float, float] = (0.0, 0.0),
) -> str:
    """All conditions must hold: unusual for this series (z), a material
    relative change (pct) and, where given, a material absolute excess
    (so tiny dry-season totals cannot look alarming)."""
    if pct is None or z is None:
        return "normal"
    if excess is None:
        excess = float("inf")
    if z >= 2.5 and pct >= 40 and excess >= min_excess[1]:
        return "high"
    if z >= 1.5 and pct >= 20 and excess >= min_excess[0]:
        return "elevated"
    return "normal"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "no comparison available"
    direction = "above" if pct >= 0 else "below"
    return f"{abs(pct):.0f}% {direction}"


def compute_signals(
    weekly: pd.DataFrame,
    origins: dict[str, str] | None = None,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Signals for one location.

    `weekly` is indexed by week_start with one column per metric.
    `origins` maps metric -> REAL | SYNTHETIC | MIXED for the badges.
    """
    origins = origins or {}
    weekly = weekly.sort_index()
    cases = weekly[CASES].dropna() if CASES in weekly else pd.Series(dtype=float)

    if as_of is None:
        if cases.empty:
            return {"as_of_week": None, "level": "normal", "signals": []}
        as_of = cases.index.max()

    signals: list[dict[str, Any]] = []

    # 1. Latest week's cases vs the same time in earlier years.
    cases2 = cases.rolling(2, min_periods=2).mean()
    if as_of in cases2.index and not pd.isna(cases2[as_of]):
        latest = float(cases2[as_of])
        history = seasonal_values(cases2.dropna(), as_of)
        baseline, pct, z = anomaly(latest, history)
        years = len(history) // 5 or 1
        signals.append(
            {
                "key": "cases_vs_seasonal_baseline",
                "label": "Cases vs usual for this time of year",
                "value": round(latest, 1),
                "unit": "cases/week (2-week average)",
                "reference": round(baseline, 1) if baseline else None,
                "change_pct": pct,
                "z_score": z,
                "status": anomaly_status(pct, z),
                "explanation": (
                    f"{latest:.0f} confirmed cases a week over the 2 weeks to {as_of.isoformat()}, "
                    f"{_fmt_pct(pct)} the usual {baseline:.0f} for this time of year "
                    f"({years} earlier year(s))."
                    if baseline
                    else f"{latest:.0f} cases a week; not enough history for a seasonal comparison."
                ),
                "data_origin": origins.get(CASES, "SYNTHETIC"),
            }
        )

    # 2. Short-term trend: last 4 weeks vs the 4 before.
    sum4 = cases.rolling(4, min_periods=4).sum()
    growth = (sum4 / sum4.shift(4)).replace([np.inf, -np.inf], np.nan).dropna()
    if as_of in growth.index:
        last4, prev4 = float(sum4[as_of]), float(sum4.shift(4)[as_of])
        observed = float(growth[as_of])
        history = seasonal_values(growth, as_of, window_weeks=1)
        usual, excess_pct, z = anomaly(observed, history)
        actual_pct = round((observed - 1) * 100, 1)
        signals.append(
            {
                "key": "cases_trend_4wk",
                "label": "4-week case trend vs usual seasonal trend",
                "value": last4,
                "unit": "cases",
                "reference": prev4,
                "change_pct": actual_pct,
                "usual_change_pct": round((usual - 1) * 100, 1) if usual else None,
                "z_score": z,
                # Excess cases beyond what the usual seasonal change would give.
                "status": anomaly_status(
                    excess_pct, z, excess=last4 - prev4 * usual if usual else None, min_excess=(15, 30)
                ),
                "explanation": (
                    f"{last4:.0f} cases in the last 4 weeks vs {prev4:.0f} in the 4 weeks before "
                    f"({actual_pct:+.0f}%). At this time of year the change is usually "
                    f"{(usual - 1) * 100:+.0f}%."
                    if usual
                    else f"{last4:.0f} cases in the last 4 weeks vs {prev4:.0f} before ({actual_pct:+.0f}%)."
                ),
                "data_origin": origins.get(CASES, "SYNTHETIC"),
            }
        )

    # 3. Rainfall over the last 8 weeks vs the same window in earlier years.
    if RAINFALL in weekly:
        rain = weekly[RAINFALL]
        rain8 = rain.rolling(8, min_periods=8).sum()
        if as_of in rain8.index and not pd.isna(rain8[as_of]):
            value = float(rain8[as_of])
            history = seasonal_values(rain8.dropna(), as_of, window_weeks=1)
            baseline, pct, z = anomaly(value, history)
            signals.append(
                {
                    "key": "rainfall_8wk_anomaly",
                    "label": "Rainfall, last 8 weeks",
                    "value": round(value, 1),
                    "unit": "mm",
                    "reference": round(baseline, 1) if baseline else None,
                    "change_pct": pct,
                    "z_score": z,
                    "status": anomaly_status(
                        pct, z, excess=value - baseline if baseline else None, min_excess=(40, 75)
                    ),
                    "explanation": (
                        f"{value:.0f} mm in the last 8 weeks, {_fmt_pct(pct)} the "
                        f"{baseline:.0f} mm usual for this window ({len(history) // 3 or 1} earlier year(s)). "
                        "Above-normal rain creates breeding sites; case rises typically follow by 3-8 weeks."
                        if baseline
                        else f"{value:.0f} mm in the last 8 weeks."
                    ),
                    "data_origin": origins.get(RAINFALL, "SYNTHETIC"),
                }
            )

    # 4. Temperature suitability over the last 4 weeks.
    if TEMP_MEAN in weekly:
        temps = weekly[TEMP_MEAN][weekly.index <= as_of].tail(4).dropna()
        if len(temps):
            mean_temp = float(temps.mean())
            low, high = TRANSMISSION_TEMP_OPTIMUM
            in_optimum = low <= mean_temp <= high
            supported = TRANSMISSION_TEMP_RANGE[0] <= mean_temp <= TRANSMISSION_TEMP_RANGE[1]
            signals.append(
                {
                    "key": "temperature_suitability",
                    "label": "Temperature suitability, last 4 weeks",
                    "value": round(mean_temp, 1),
                    "unit": "°C",
                    "reference": None,
                    "change_pct": None,
                    # Context only: Lagos is almost always in the favourable
                    # band, so this never raises the overall level by itself.
                    "status": "normal",
                    "favourable": in_optimum,
                    "explanation": (
                        f"Mean {mean_temp:.1f} °C: "
                        + (
                            f"inside the {low:.0f}-{high:.0f} °C band most favourable to transmission."
                            if in_optimum
                            else "transmission possible but outside the most favourable band."
                            if supported
                            else "outside the range that usually supports transmission."
                        )
                    ),
                    "data_origin": origins.get(TEMP_MEAN, "SYNTHETIC"),
                }
            )

    # 5. Verified laboratory positives (Module 5), last 4 weeks.
    if LAB_POSITIVE in weekly:
        start = as_of - timedelta(weeks=3)
        lab = weekly[LAB_POSITIVE][(weekly.index >= start) & (weekly.index <= as_of)].fillna(0)
        positives = int(lab.sum())
        signals.append(
            {
                "key": "verified_lab_positives_4wk",
                "label": "Verified lab-positive smears, last 4 weeks",
                "value": positives,
                "unit": "smears",
                "reference": None,
                "change_pct": None,
                "status": "high" if positives >= 3 else "elevated" if positives >= 1 else "normal",
                "explanation": (
                    f"{positives} blood smear(s) verified positive by a qualified reviewer. "
                    "Only reviewer-verified results are counted."
                ),
                "data_origin": origins.get(LAB_POSITIVE, "SYNTHETIC"),
            }
        )

    level = max((s["status"] for s in signals), key=STATUS_ORDER.get, default="normal")
    return {"as_of_week": as_of.isoformat(), "level": level, "signals": signals}
