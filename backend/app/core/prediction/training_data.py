"""Load the weekly panel the prediction engine learns from."""

from __future__ import annotations

import pandas as pd
from sqlalchemy.orm import Session

from backend.app.core.surveillance.metrics import CASES, HUMIDITY, RAINFALL, TEMP_MEAN
from backend.app.core.surveillance.repository import (
    load_observations,
    location_ids,
    origins_by_metric,
    wide_for_location,
)

MODEL_METRICS = (CASES, RAINFALL, TEMP_MEAN, HUMIDITY)


def load_training_wides(session: Session) -> tuple[dict[int, pd.DataFrame], str]:
    """Wide weekly frames per location id, and the case data's origin."""
    long = load_observations(session, metrics=MODEL_METRICS)
    wides = {
        location_id: wide_for_location(long, location_id)
        for location_id in location_ids(session).values()
    }
    origin = origins_by_metric(long).get(CASES, "SYNTHETIC")
    return {k: v for k, v in wides.items() if not v.empty}, origin
