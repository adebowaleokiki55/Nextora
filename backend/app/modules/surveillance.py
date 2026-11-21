"""Module 1: environmental and historical surveillance API."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.surveillance.metrics import (
    CASES,
    HUMIDITY,
    LAB_POSITIVE,
    LAB_REVIEWED,
    LABELS,
    RAINFALL,
    TEMP_MAX,
    TEMP_MEAN,
    UNITS,
)
from backend.app.core.surveillance.repository import (
    load_observations,
    origins_by_metric,
    sources_by_metric,
    wide_for_location,
)
from backend.app.core.surveillance.signals import compute_signals
from backend.app.db import get_db
from backend.app.models import Location

router = APIRouter(prefix="/api/surveillance", tags=["surveillance"])

SERIES_METRICS = (CASES, RAINFALL, TEMP_MEAN, TEMP_MAX, HUMIDITY, LAB_POSITIVE, LAB_REVIEWED)


def _signal(result: dict, key: str) -> dict | None:
    return next((s for s in result["signals"] if s["key"] == key), None)


@router.get("/overview")
def overview(db: Session = Depends(get_db)):
    """Current conditions for every monitored location."""
    long = load_observations(db)
    origins = origins_by_metric(long)
    locations = list(db.scalars(select(Location).order_by(Location.name)))
    rows = []

    for loc in locations:
        result = compute_signals(wide_for_location(long, loc.id), origins)
        cases = _signal(result, "cases_vs_seasonal_baseline")
        rain = _signal(result, "rainfall_8wk_anomaly")
        lab = _signal(result, "verified_lab_positives_4wk")
        rows.append(
            {
                "location": {
                    "id": loc.id,
                    "name": loc.name,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude,
                },
                "as_of_week": result["as_of_week"],
                "level": result["level"],
                "cases_per_week": cases["value"] if cases else None,
                "cases_change_pct": cases["change_pct"] if cases else None,
                "rainfall_8wk_mm": rain["value"] if rain else None,
                "rainfall_change_pct": rain["change_pct"] if rain else None,
                "verified_lab_positives_4wk": lab["value"] if lab else 0,
                "elevated_signals": [
                    s["label"] for s in result["signals"] if s["status"] != "normal"
                ],
            }
        )

    return {
        "data_origin": {m: origins.get(m) for m in (CASES, RAINFALL, TEMP_MEAN)},
        "locations": rows,
        "note": (
            "Levels describe current conditions against the same time of year in "
            "earlier years. Forecasts and risk scores come from the prediction engine."
        ),
    }


@router.get("/locations/{location_id}")
def location_detail(
    location_id: int,
    weeks: int = Query(156, ge=8, le=1200),
    db: Session = Depends(get_db),
):
    """Signals plus weekly time series for one location."""
    location = db.get(Location, location_id)
    if location is None:
        raise HTTPException(status_code=404, detail=f"Location {location_id} was not found.")

    long = load_observations(db, location_ids=[location_id], metrics=SERIES_METRICS)
    wide = wide_for_location(long, location_id)
    origins = origins_by_metric(long)
    sources = sources_by_metric(long)
    result = compute_signals(wide, origins)

    series = {}
    if not wide.empty:
        since = wide.index.max() - timedelta(weeks=weeks - 1)
        window = wide[wide.index >= since]
        for metric in SERIES_METRICS:
            if metric not in window:
                continue
            values = window[metric]
            if metric in (LAB_POSITIVE, LAB_REVIEWED):
                values = values.fillna(0)
            series[metric] = {
                "label": LABELS[metric],
                "unit": UNITS[metric],
                "data_origin": origins.get(metric),
                "sources": sources.get(metric, []),
                "points": [
                    {"week_start": week.isoformat(), "value": None if pd.isna(v) else round(float(v), 2)}
                    for week, v in values.items()
                ],
            }

    return {
        "location": {
            "id": location.id,
            "name": location.name,
            "state": location.state,
            "latitude": location.latitude,
            "longitude": location.longitude,
            "data_origin": location.data_origin.value,
            "source": location.source,
        },
        **result,
        "series": series,
    }


@router.get("/sources")
def data_sources(db: Session = Depends(get_db)):
    """What data the platform holds, where it came from, and whether it is real."""
    long = load_observations(db)
    entries = []
    for metric, group in long.groupby("metric"):
        origins = sorted(set(group["data_origin"]))
        entries.append(
            {
                "metric": metric,
                "label": LABELS.get(metric, metric),
                "data_origin": origins[0] if len(origins) == 1 else "MIXED",
                "sources": sorted(set(group["source"])),
                "locations": int(group["location_id"].nunique()),
                "first_week": min(group["week_start"]).isoformat(),
                "last_week": max(group["week_start"]).isoformat(),
                "rows": int(len(group)),
            }
        )
    return {"sources": entries}
