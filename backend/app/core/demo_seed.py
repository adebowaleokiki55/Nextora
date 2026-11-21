"""Demo scenario seeding (Lagos). Later phases add weather, cases and stock.

Location names are the 20 real Lagos local government areas; their points
are approximate centres compiled by hand, not an official boundary dataset.
Facilities are fictional and marked SYNTHETIC.
"""

from __future__ import annotations

import math

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.app.models import (
    AgentMessage,
    AgentRun,
    Alert,
    DataOrigin,
    Facility,
    LabCell,
    LabSubmission,
    Location,
    Observation,
    Proposal,
)

LGA_SOURCE = "Lagos LGA names; approximate centre points compiled by hand (not an official dataset)"
FACILITY_SOURCE = "Nextora demo scenario (fictional facility)"

# name: (latitude, longitude), approximate LGA centres
LAGOS_LGAS: dict[str, tuple[float, float]] = {
    "Agege": (6.618, 3.321),
    "Ajeromi-Ifelodun": (6.455, 3.334),
    "Alimosho": (6.610, 3.296),
    "Amuwo-Odofin": (6.460, 3.280),
    "Apapa": (6.449, 3.359),
    "Badagry": (6.432, 2.888),
    "Epe": (6.584, 3.983),
    "Eti-Osa": (6.459, 3.602),
    "Ibeju-Lekki": (6.470, 3.880),
    "Ifako-Ijaiye": (6.640, 3.320),
    "Ikeja": (6.602, 3.352),
    "Ikorodu": (6.619, 3.511),
    "Kosofe": (6.590, 3.390),
    "Lagos Island": (6.454, 3.395),
    "Lagos Mainland": (6.495, 3.380),
    "Mushin": (6.530, 3.350),
    "Ojo": (6.460, 3.180),
    "Oshodi-Isolo": (6.540, 3.320),
    "Shomolu": (6.540, 3.380),
    "Surulere": (6.500, 3.350),
}

# Ikorodu, the demo hotspot, gets five facilities (Phase 4 makes three of
# them short of RDT kits). LGAs listed here also get a laboratory.
IKORODU_FACILITIES = ["A", "B", "C", "D", "E"]
LAB_LGAS = {"Ikorodu", "Alimosho", "Kosofe", "Epe", "Badagry", "Surulere"}


def _offset(lat: float, lon: float, index: int, radius_km: float = 2.5) -> tuple[float, float]:
    """Deterministic spread of facilities around an LGA centre."""
    angle = index * 2.399963  # golden angle, radians
    distance = radius_km * (0.4 + 0.6 * ((index * 37) % 10) / 10)
    dlat = distance / 111.0 * math.cos(angle)
    dlon = distance / (111.0 * math.cos(math.radians(lat))) * math.sin(angle)
    return round(lat + dlat, 5), round(lon + dlon, 5)


def _facilities_for(lga: str, location: Location) -> list[Facility]:
    lat, lon = location.latitude, location.longitude
    facilities: list[Facility] = []

    if lga == "Ikorodu":
        names = [f"Ikorodu Demo Health Centre {letter}" for letter in IKORODU_FACILITIES]
    else:
        names = [f"{lga} Demo Health Centre"]

    for index, name in enumerate(names):
        f_lat, f_lon = _offset(lat, lon, index + 1)
        facilities.append(
            Facility(
                name=name,
                location=location,
                facility_type="primary_health_centre",
                has_laboratory=False,
                latitude=f_lat,
                longitude=f_lon,
                data_origin=DataOrigin.SYNTHETIC,
                source=FACILITY_SOURCE,
            )
        )

    if lga in LAB_LGAS:
        f_lat, f_lon = _offset(lat, lon, len(names) + 1)
        facilities.append(
            Facility(
                name=f"{lga} Demo Laboratory",
                location=location,
                facility_type="laboratory",
                has_laboratory=True,
                latitude=f_lat,
                longitude=f_lon,
                data_origin=DataOrigin.SYNTHETIC,
                source=FACILITY_SOURCE,
            )
        )

    return facilities


def seed_places(session: Session) -> dict[str, int]:
    for lga, (lat, lon) in LAGOS_LGAS.items():
        location = Location(
            name=lga,
            state="Lagos",
            latitude=lat,
            longitude=lon,
            data_origin=DataOrigin.REAL,
            source=LGA_SOURCE,
        )
        session.add(location)
        session.add_all(_facilities_for(lga, location))
    session.commit()
    return {
        "locations": session.scalar(select(func.count(Location.id))),
        "facilities": session.scalar(select(func.count(Facility.id))),
    }


def reset_and_seed(session: Session) -> dict[str, int]:
    """Delete all scenario data and re-seed. Used by the demo reset action."""
    for model in (
        AgentMessage,
        Proposal,
        AgentRun,
        Alert,
        LabCell,
        LabSubmission,
        Observation,
        Facility,
        Location,
    ):
        session.execute(delete(model))
    session.commit()
    return seed_places(session)


def seed_if_empty(session: Session) -> bool:
    if session.scalar(select(func.count(Location.id))):
        return False
    seed_places(session)
    return True
