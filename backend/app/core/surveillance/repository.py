"""Read and write weekly surveillance data in the shared observations table."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from backend.app.core.surveillance.metrics import LAB_POSITIVE, LAB_REVIEWED, UNITS
from backend.app.models import DataOrigin, Location, Observation, VerificationStatus

LAB_METRICS = (LAB_REVIEWED, LAB_POSITIVE)


def bulk_insert_weekly(
    session: Session,
    frame: pd.DataFrame,
    metrics: Iterable[str],
    location_ids: dict[str, int],
    data_origin: DataOrigin,
    source: str,
) -> int:
    """Insert wide weekly rows (location, week_start, metric columns...)."""
    rows = []
    for record in frame.to_dict("records"):
        start = record["week_start"]
        for metric in metrics:
            value = record.get(metric)
            if value is None or pd.isna(value):
                continue
            rows.append(
                {
                    "location_id": location_ids[record["location"]],
                    "metric": metric,
                    "value": float(value),
                    "unit": UNITS.get(metric),
                    "period": "week",
                    "period_start": start,
                    "period_end": start + timedelta(days=6),
                    "data_origin": data_origin,
                    "source": source,
                    "verification_status": VerificationStatus.NOT_APPLICABLE,
                }
            )
    if rows:
        session.execute(insert(Observation), rows)
    return len(rows)


def location_ids(session: Session) -> dict[str, int]:
    return {loc.name: loc.id for loc in session.scalars(select(Location))}


def load_observations(
    session: Session,
    location_ids: Iterable[int] | None = None,
    metrics: Iterable[str] | None = None,
    since: date | None = None,
) -> pd.DataFrame:
    """Long frame: location_id, metric, week_start, value, data_origin, source.

    Daily lab observations are only included when VERIFIED, and are summed
    into ISO weeks so they line up with the weekly series.
    """
    query = select(
        Observation.location_id,
        Observation.metric,
        Observation.period,
        Observation.period_start,
        Observation.value,
        Observation.data_origin,
        Observation.source,
        Observation.verification_status,
    )
    if location_ids is not None:
        query = query.where(Observation.location_id.in_(list(location_ids)))
    if metrics is not None:
        query = query.where(Observation.metric.in_(list(metrics)))
    if since is not None:
        query = query.where(Observation.period_start >= since)

    columns = [
        "location_id", "metric", "period", "period_start", "value",
        "data_origin", "source", "verification_status",
    ]
    frame = pd.DataFrame(session.execute(query).all(), columns=columns)
    if frame.empty:
        return pd.DataFrame(columns=["location_id", "metric", "week_start", "value", "data_origin", "source"])

    frame["data_origin"] = frame["data_origin"].map(lambda o: o.value if hasattr(o, "value") else o)

    is_lab = frame["metric"].isin(LAB_METRICS)
    lab = frame[is_lab & (frame["verification_status"] == VerificationStatus.VERIFIED)].copy()
    weekly = frame[~is_lab & (frame["period"] == "week")].copy()

    weekly["week_start"] = weekly["period_start"]
    if not lab.empty:
        lab["week_start"] = lab["period_start"].map(lambda d: d - timedelta(days=d.weekday()))
        lab = lab.groupby(["location_id", "metric", "week_start"], as_index=False).agg(
            value=("value", "sum"),
            data_origin=("data_origin", lambda s: "SYNTHETIC" if (s == "SYNTHETIC").any() else "REAL"),
            source=("source", lambda s: "Verified lab submissions (Module 5)"),
        )

    keep = ["location_id", "metric", "week_start", "value", "data_origin", "source"]
    return pd.concat([weekly[keep], lab[keep] if not lab.empty else None], ignore_index=True)


def wide_for_location(long: pd.DataFrame, location_id: int) -> pd.DataFrame:
    subset = long[long["location_id"] == location_id]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(index="week_start", columns="metric", values="value", aggfunc="sum")
    return wide.sort_index()


def origins_by_metric(long: pd.DataFrame) -> dict[str, str]:
    result = {}
    for metric, group in long.groupby("metric"):
        values = set(group["data_origin"])
        result[metric] = values.pop() if len(values) == 1 else "MIXED"
    return result


def sources_by_metric(long: pd.DataFrame) -> dict[str, list[str]]:
    return {metric: sorted(set(group["source"])) for metric, group in long.groupby("metric")}
