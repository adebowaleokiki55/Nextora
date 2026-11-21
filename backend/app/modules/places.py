"""Shared reference data: monitored locations and participating facilities."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db import get_db
from backend.app.models import Facility, Location

router = APIRouter(prefix="/api", tags=["places"])


@router.get("/locations")
def locations(db: Session = Depends(get_db)):
    rows = db.scalars(select(Location).order_by(Location.name))
    return {
        "locations": [
            {
                "id": loc.id,
                "name": loc.name,
                "state": loc.state,
                "latitude": loc.latitude,
                "longitude": loc.longitude,
                "data_origin": loc.data_origin.value,
                "source": loc.source,
            }
            for loc in rows
        ]
    }


@router.get("/facilities")
def facilities(
    location_id: int | None = Query(None),
    has_laboratory: bool | None = Query(None),
    db: Session = Depends(get_db),
):
    query = select(Facility).order_by(Facility.name)
    if location_id is not None:
        query = query.where(Facility.location_id == location_id)
    if has_laboratory is not None:
        query = query.where(Facility.has_laboratory == has_laboratory)
    return {
        "facilities": [
            {
                "id": f.id,
                "name": f.name,
                "location_id": f.location_id,
                "location_name": f.location.name,
                "facility_type": f.facility_type,
                "has_laboratory": f.has_laboratory,
                "latitude": f.latitude,
                "longitude": f.longitude,
                "data_origin": f.data_origin.value,
                "source": f.source,
            }
            for f in db.scalars(query)
        ]
    }
