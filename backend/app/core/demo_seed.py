"""Demo scenario seeding (Lagos).

- Locations: the 20 real Lagos local government areas; their points are
  approximate centres compiled by hand, not an official boundary dataset.
- Facilities: fictional, marked SYNTHETIC.
- Weather: REAL (Open-Meteo reanalysis) when data/real/lagos_weather_weekly.csv
  exists, otherwise SYNTHETIC from a climatology.
- Weekly malaria cases: always SYNTHETIC, driven by the weather, with the
  scripted Ikorodu rise in the latest weeks.
- Stock: SYNTHETIC consumption (from each LGA's cases) and stock on hand,
  with three Ikorodu health centres short of RDT kits and nearby surplus.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from backend.app.config import DATA_DIR, PROJECT_ROOT
from backend.app.core.surveillance import synthetic_cases, weather
from backend.app.core.surveillance.metrics import CASES, WEATHER_METRICS
from backend.app.core.surveillance.repository import bulk_insert_weekly, location_ids

from backend.app.models import (
    AgentMessage,
    AgentRun,
    Alert,
    DataOrigin,
    Facility,
    Forecast,
    LabCell,
    LabSubmission,
    Location,
    Observation,
    Proposal,
    StockLevel,
    StockTransaction,
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


WEATHER_CSV = DATA_DIR / "real" / "lagos_weather_weekly.csv"
HISTORY_START = date(2016, 1, 4)  # a Monday


def load_weather(today: date | None = None, csv_path=WEATHER_CSV):
    """Weekly weather and its provenance: (frame, DataOrigin, source)."""
    real = weather.load_weekly_csv(csv_path)
    if real is not None and not real.empty:
        meta = weather.load_metadata(csv_path) or {}
        source = meta.get("source", weather.REAL_SOURCE)
        if meta.get("retrieved_on"):
            source += f"; retrieved {meta['retrieved_on']}"
        return real, DataOrigin.REAL, source

    last_week = weather.last_complete_week_start(today or date.today())
    frame = weather.synthetic_weekly_weather(LAGOS_LGAS, HISTORY_START, last_week)
    return frame, DataOrigin.SYNTHETIC, weather.SYNTHETIC_SOURCE


def seed_surveillance(session: Session, today: date | None = None, csv_path=WEATHER_CSV) -> dict[str, object]:
    ids = location_ids(session)
    weekly_weather, weather_origin, weather_source = load_weather(today, csv_path)
    weekly_weather = weekly_weather[weekly_weather["location"].isin(ids)]

    weather_rows = bulk_insert_weekly(
        session, weekly_weather, WEATHER_METRICS, ids, weather_origin, weather_source
    )
    cases = synthetic_cases.generate_cases(weekly_weather)
    case_rows = bulk_insert_weekly(
        session, cases, [CASES], ids, DataOrigin.SYNTHETIC, synthetic_cases.SOURCE
    )
    session.commit()
    return {
        "weather_rows": weather_rows,
        "weather_origin": weather_origin.value,
        "case_rows": case_rows,
        "first_week": str(weekly_weather["week_start"].min()),
        "last_week": str(weekly_weather["week_start"].max()),
    }


FORECAST_MODEL_DIR = PROJECT_ROOT / "ml" / "models" / "forecast"


def seed_forecasts(session: Session, model_dir=FORECAST_MODEL_DIR) -> dict[str, object]:
    """Run the prediction engine on the seeded data, if models are trained."""
    from backend.app.core.prediction.forecast import run_forecasts
    from backend.app.core.prediction.model import load_bundle

    bundle = load_bundle(model_dir)
    if bundle is None:
        return {"forecasts": "skipped: no trained model (run python -m ml.train_forecast)"}
    result = run_forecasts(session, bundle)
    return {"forecast_model": result["model_version"], "early_warnings": result["early_warnings"]}


STOCK_SOURCE = "Nextora demo scenario (synthetic stock and consumption)"
LEDGER_WEEKS = 12
# Share of an LGA's confirmed cases seen at each demo facility.
HEALTH_CENTRE_SHARE = {"Ikorodu": 0.12}
DEFAULT_HEALTH_CENTRE_SHARE = 0.35
LAB_RDT_SHARE = 0.10
# Scripted stock on hand, in weeks of recent consumption.
SCENARIO_STOCK_WEEKS = {
    ("Ikorodu Demo Health Centre A", "RDT"): 0.4,
    ("Ikorodu Demo Health Centre B", "RDT"): 0.0,
    ("Ikorodu Demo Health Centre C", "RDT"): 1.1,
    ("Ikorodu Demo Health Centre D", "RDT"): 7.5,
    ("Ikorodu Demo Laboratory", "RDT"): 16.0,
    ("Kosofe Demo Health Centre", "RDT"): 15.0,
    ("Epe Demo Health Centre", "RDT"): 14.0,  # surplus, but beyond the transfer radius
}


def seed_supply(session: Session, seed: int = 11) -> dict[str, object]:
    """Synthetic stock ledger (12 weeks of consumption) and current stock levels."""
    from backend.app.core.supply.items import ITEMS
    from backend.app.core.supply.service import run_supply_scan

    rng = np.random.default_rng(seed)
    last_week = session.scalar(select(func.max(Observation.period_start)).where(Observation.metric == CASES))
    if last_week is None:
        return {"stock": "skipped: no case data"}
    weeks = [last_week - timedelta(weeks=k) for k in range(LEDGER_WEEKS - 1, -1, -1)]
    cases = {
        (row.location_id, row.period_start): row.value
        for row in session.scalars(
            select(Observation).where(Observation.metric == CASES, Observation.period_start >= weeks[0])
        )
    }

    levels = transactions = 0
    for facility in session.scalars(select(Facility).order_by(Facility.id)):
        lga = facility.location.name
        if facility.has_laboratory:
            items = {"RDT": LAB_RDT_SHARE}
        else:
            share = HEALTH_CENTRE_SHARE.get(lga, DEFAULT_HEALTH_CENTRE_SHARE)
            items = {code: share for code in ITEMS}

        for code, share in items.items():
            item = ITEMS[code]
            used = []
            for week in weeks:
                expected = cases.get((facility.location_id, week), 0.0) * share * item.per_case
                quantity = int(round(expected * rng.gamma(30, 1 / 30)))
                used.append(quantity)
                session.add(
                    StockTransaction(
                        facility_id=facility.id, item_code=code, occurred_on=week, kind="CONSUMPTION",
                        quantity=-quantity, data_origin=DataOrigin.SYNTHETIC, source=STOCK_SOURCE,
                    )
                )
                transactions += 1

            recent = float(np.mean(used[-8:]))
            weeks_on_hand = SCENARIO_STOCK_WEEKS.get((facility.name, code), float(rng.uniform(3.5, 9.0)))
            session.add(
                StockLevel(
                    facility_id=facility.id, item_code=code, quantity_on_hand=int(round(recent * weeks_on_hand)),
                    data_origin=DataOrigin.SYNTHETIC, source=STOCK_SOURCE,
                )
            )
            levels += 1

    session.commit()
    scan = run_supply_scan(session)
    return {
        "stock_levels": levels,
        "stock_transactions": transactions,
        "supply_shortages": [f"{s['facility']['name']} {s['item']['code']}" for s in scan["shortages"]],
    }


def reset_and_seed(session: Session) -> dict[str, object]:
    """Delete all scenario data and re-seed. Used by the demo reset action."""
    for model in (
        AgentMessage,
        Proposal,
        StockTransaction,
        StockLevel,
        AgentRun,
        Alert,
        Forecast,
        LabCell,
        LabSubmission,
        Observation,
        Facility,
        Location,
    ):
        session.execute(delete(model))
    session.commit()
    counts = seed_places(session)
    counts.update(seed_surveillance(session))
    counts.update(seed_forecasts(session))
    counts.update(seed_supply(session))
    return counts


def seed_if_empty(session: Session) -> bool:
    if session.scalar(select(func.count(Location.id))):
        return False
    seed_places(session)
    seed_surveillance(session)
    seed_forecasts(session)
    seed_supply(session)
    return True
