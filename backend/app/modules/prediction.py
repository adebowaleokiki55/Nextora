"""Module 2: malaria prediction engine API."""

from __future__ import annotations

import json
from datetime import timedelta
from functools import lru_cache

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.config import PROJECT_ROOT
from backend.app.core.demo_seed import FORECAST_MODEL_DIR
from backend.app.core.prediction.forecast import (
    forecast_to_dict,
    latest_forecasts,
    run_forecasts,
    summarise,
)
from backend.app.core.prediction.model import ForecastBundle, load_bundle
from backend.app.core.surveillance.metrics import CASES
from backend.app.core.surveillance.repository import load_observations, wide_for_location
from backend.app.db import get_db
from backend.app.models import Location

router = APIRouter(prefix="/api/prediction", tags=["prediction"])

METRICS_DIR = PROJECT_ROOT / "ml" / "metrics"


@lru_cache(maxsize=1)
def _bundle() -> ForecastBundle | None:
    return load_bundle(FORECAST_MODEL_DIR)


def _require_bundle() -> ForecastBundle:
    bundle = _bundle()
    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail="No trained forecast model. Run `python -m ml.train_forecast` first.",
        )
    return bundle


def _location_payload(loc: Location, rows) -> dict:
    forecasts = [forecast_to_dict(r) for r in rows]
    summary = summarise(forecasts)
    return {
        "location": {"id": loc.id, "name": loc.name, "latitude": loc.latitude, "longitude": loc.longitude},
        **summary,
        "first_warning_week": summary["first_warning_week"].isoformat() if summary["first_warning_week"] else None,
        "model_version": rows[0].model_version if rows else None,
        "data_origin": rows[0].data_origin.value if rows else None,
        "forecasts": [
            {**f, "origin_week": f["origin_week"].isoformat(), "target_week": f["target_week"].isoformat()}
            for f in forecasts
        ],
    }


@router.get("/risk")
def risk(db: Session = Depends(get_db)):
    """Latest risk score, level and 1-4 week forecasts for every location."""
    grouped = latest_forecasts(db)
    locations = list(db.scalars(select(Location).order_by(Location.name)))
    rows = [_location_payload(loc, grouped.get(loc.id, [])) for loc in locations]
    rows.sort(key=lambda r: -(r["risk_score"] or -1))
    return {
        "locations": rows,
        "risk_score_definition": (
            "Highest estimated probability (x100) that weekly confirmed cases reach the "
            "location's alert threshold in the next 1-4 weeks. The threshold is the mean + 2 SD "
            "of cases in the same weeks of earlier years."
        ),
    }


@router.get("/locations/{location_id}")
def location_forecast(
    location_id: int,
    history_weeks: int = Query(26, ge=4, le=520),
    db: Session = Depends(get_db),
):
    """Recent actual cases, the latest forecasts, and the test-period backtest."""
    location = db.get(Location, location_id)
    if location is None:
        raise HTTPException(status_code=404, detail=f"Location {location_id} was not found.")

    long = load_observations(db, location_ids=[location_id], metrics=[CASES])
    wide = wide_for_location(long, location_id)
    history = []
    if not wide.empty:
        since = wide.index.max() - timedelta(weeks=history_weeks - 1)
        history = [
            {"week_start": week.isoformat(), "cases": float(value)}
            for week, value in wide.loc[wide.index >= since, CASES].dropna().items()
        ]

    backtest = []
    backtest_path = METRICS_DIR / "forecast_backtest.csv"
    if backtest_path.exists():
        frame = pd.read_csv(backtest_path)
        frame = frame[frame["location_id"] == location_id]
        backtest = frame.where(frame.notna(), None).to_dict("records")

    payload = _location_payload(location, latest_forecasts(db).get(location_id, []))
    return {
        **payload,
        "history": history,
        "backtest": backtest,
        "backtest_note": "Test-period forecasts made without seeing those weeks (time-based split).",
    }


@router.get("/evaluation")
def evaluation():
    """Accuracy of the model vs the three baselines on the held-out test period."""
    path = METRICS_DIR / "forecast_metrics.json"
    if not path.exists():
        raise HTTPException(status_code=503, detail="No evaluation yet. Run `python -m ml.train_forecast`.")
    return json.loads(path.read_text(encoding="utf-8"))


@router.post("/run")
def run(db: Session = Depends(get_db)):
    """Recompute forecasts from the current data and update early-warning alerts."""
    return run_forecasts(db, _require_bundle())
