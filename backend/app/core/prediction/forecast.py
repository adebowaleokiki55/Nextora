"""Live forecasts, risk scores and early-warning alerts.

Risk score: the highest estimated probability, over the next 1-4 weeks,
that weekly cases reach the alert threshold (x100). Probabilities come from
the model's out-of-sample validation errors, so they are estimates, not
guarantees; they inherit the limits of the (synthetic) training data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.app.core.prediction.features import FEATURE_GROUPS, FEATURES, location_rows
from backend.app.core.prediction.model import ForecastBundle, exceedance_probability, interval
from backend.app.core.prediction.training_data import load_training_wides
from backend.app.models import Alert, AlertSeverity, AlertStatus, DataOrigin, Forecast, Location

ALERT_TYPE = "RISK_FORECAST"

RISK_LEVELS = (  # (minimum score, level)
    (80, "VERY_HIGH"),
    (50, "HIGH"),
    (20, "MODERATE"),
    (0, "LOW"),
)


def risk_level(score: float | None) -> str:
    if score is None:
        return "UNKNOWN"
    return next(level for minimum, level in RISK_LEVELS if score >= minimum)


def _drivers(contributions: np.ndarray) -> dict[str, float]:
    """Sum feature contributions (log scale) into readable groups."""
    index = {name: i for i, name in enumerate(FEATURES)}
    return {
        group: round(float(sum(contributions[index[f]] for f in features)), 3)
        for group, features in FEATURE_GROUPS.items()
    }


def forecast_location(bundle: ForecastBundle, wide: pd.DataFrame, location_id: int) -> list[dict[str, Any]]:
    """Forecasts for 1-4 weeks after the latest week with data."""
    results = []
    for horizon, model in sorted(bundle.models.items()):
        rows = location_rows(wide, location_id, horizon)
        if rows.empty:
            continue
        latest = rows.tail(1).reset_index(drop=True)
        pred_log = model.predict_log(latest)
        low, high = interval(pred_log, model.residuals)
        threshold = latest["threshold"].to_numpy(dtype=float)
        probability = exceedance_probability(pred_log, model.residuals, threshold)
        results.append(
            {
                "horizon": horizon,
                "origin_week": pd.Timestamp(latest["origin_week"].iloc[0]).date(),
                "target_week": pd.Timestamp(latest["target_week"].iloc[0]).date(),
                "predicted_cases": round(float(np.expm1(pred_log[0])), 1),
                "lower_cases": round(float(low[0]), 1),
                "upper_cases": round(float(high[0]), 1),
                "alert_threshold": None if np.isnan(threshold[0]) else round(float(threshold[0]), 1),
                "exceedance_probability": None if np.isnan(probability[0]) else round(float(probability[0]), 3),
                "alert_cutoff": model.alert_cutoff,
                "drivers": _drivers(model.contributions(latest)[0]),
            }
        )
    return results


def summarise(forecasts: list[dict[str, Any]]) -> dict[str, Any]:
    """Risk score, level and whether an early warning is due."""
    probabilities = [f["exceedance_probability"] for f in forecasts if f["exceedance_probability"] is not None]
    score = round(max(probabilities) * 100, 1) if probabilities else None
    warning = [
        f for f in forecasts
        if f["exceedance_probability"] is not None and f["exceedance_probability"] >= f["alert_cutoff"]
    ]
    return {
        "risk_score": score,
        "risk_level": risk_level(score),
        "early_warning": bool(warning),
        "first_warning_week": min((f["target_week"] for f in warning), default=None),
    }


def run_forecasts(session: Session, bundle: ForecastBundle) -> dict[str, Any]:
    """Forecast every location, store results, and raise or resolve alerts."""
    wides, case_origin = load_training_wides(session)
    origin = DataOrigin(case_origin) if case_origin in ("REAL", "SYNTHETIC") else DataOrigin.SYNTHETIC
    names = {loc.id: loc.name for loc in session.scalars(select(Location))}
    summaries = {}

    for location_id, wide in wides.items():
        forecasts = forecast_location(bundle, wide, location_id)
        if not forecasts:
            continue
        origin_week = forecasts[0]["origin_week"]
        session.execute(
            delete(Forecast).where(Forecast.location_id == location_id, Forecast.origin_week == origin_week)
        )
        session.add_all(
            Forecast(model_version=bundle.version, location_id=location_id, data_origin=origin, **f)
            for f in forecasts
        )
        summary = summarise(forecasts)
        summaries[location_id] = summary
        _update_alert(session, location_id, names[location_id], forecasts, summary, origin)

    session.commit()
    return {
        "model_version": bundle.version,
        "locations": len(summaries),
        "early_warnings": sorted(names[i] for i, s in summaries.items() if s["early_warning"]),
    }


def _update_alert(
    session: Session,
    location_id: int,
    name: str,
    forecasts: list[dict[str, Any]],
    summary: dict[str, Any],
    origin: DataOrigin,
) -> None:
    existing = session.scalar(
        select(Alert).where(
            Alert.alert_type == ALERT_TYPE,
            Alert.location_id == location_id,
            Alert.status != AlertStatus.RESOLVED,
        )
    )

    if not summary["early_warning"]:
        if existing is not None:
            existing.status = AlertStatus.RESOLVED
            existing.payload = {**existing.payload, "resolved_reason": "Latest forecast is below the alert cut-off."}
        return

    peak = max(forecasts, key=lambda f: f["exceedance_probability"] or 0)
    # Severity follows the risk level. The warning cut-off is tuned for recall,
    # so a MODERATE-risk location can still get a (MEDIUM) early warning.
    severity = {
        "VERY_HIGH": AlertSeverity.CRITICAL,
        "HIGH": AlertSeverity.HIGH,
    }.get(summary["risk_level"], AlertSeverity.MEDIUM)
    synthetic_note = " Based on SYNTHETIC case data." if origin is DataOrigin.SYNTHETIC else ""
    message = (
        f"Malaria cases in {name} may reach the alert threshold within "
        f"{max(f['horizon'] for f in forecasts)} weeks (estimated probability "
        f"{peak['exceedance_probability'] * 100:.0f}%). Week of {peak['target_week'].isoformat()}: "
        f"about {peak['predicted_cases']:.0f} cases (80% range {peak['lower_cases']:.0f}-"
        f"{peak['upper_cases']:.0f}) against a threshold of {peak['alert_threshold']:.0f}."
        + synthetic_note
    )
    payload = {
        "risk_score": summary["risk_score"],
        "risk_level": summary["risk_level"],
        "origin_week": peak["origin_week"].isoformat(),
        "peak_target_week": peak["target_week"].isoformat(),
        "forecasts": [
            {
                "target_week": f["target_week"].isoformat(),
                "predicted_cases": f["predicted_cases"],
                "lower_cases": f["lower_cases"],
                "upper_cases": f["upper_cases"],
                "alert_threshold": f["alert_threshold"],
                "exceedance_probability": f["exceedance_probability"],
            }
            for f in forecasts
        ],
    }
    title = f"Early warning: rising malaria risk in {name}"

    if existing is None:
        session.add(
            Alert(
                alert_type=ALERT_TYPE,
                source_module="prediction",
                severity=severity,
                location_id=location_id,
                title=title,
                message=message,
                payload=payload,
                data_origin=origin,
            )
        )
    else:
        existing.severity = severity
        existing.title = title
        existing.message = message
        existing.payload = payload
        existing.data_origin = origin


def latest_forecasts(session: Session) -> dict[int, list[Forecast]]:
    """Most recent forecast run per location."""
    latest = (
        select(Forecast.location_id, func.max(Forecast.origin_week).label("origin_week"))
        .group_by(Forecast.location_id)
        .subquery()
    )
    rows = session.scalars(
        select(Forecast)
        .join(latest, (Forecast.location_id == latest.c.location_id) & (Forecast.origin_week == latest.c.origin_week))
        .order_by(Forecast.location_id, Forecast.horizon)
    )
    grouped: dict[int, list[Forecast]] = {}
    for row in rows:
        grouped.setdefault(row.location_id, []).append(row)
    return grouped


def forecast_to_dict(row: Forecast) -> dict[str, Any]:
    return {
        "horizon": row.horizon,
        "origin_week": row.origin_week,
        "target_week": row.target_week,
        "predicted_cases": row.predicted_cases,
        "lower_cases": row.lower_cases,
        "upper_cases": row.upper_cases,
        "alert_threshold": row.alert_threshold,
        "exceedance_probability": row.exceedance_probability,
        "alert_cutoff": row.alert_cutoff,
        "drivers": row.drivers,
    }
