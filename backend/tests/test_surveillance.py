from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from backend.app.core.demo_seed import LAGOS_LGAS, load_weather, seed_places, seed_surveillance
from backend.app.core.surveillance import weather
from backend.app.core.surveillance.metrics import CASES, LAB_POSITIVE, RAINFALL, TEMP_MEAN
from backend.app.core.surveillance.repository import (
    load_observations,
    location_ids,
    origins_by_metric,
    wide_for_location,
)
from backend.app.core.surveillance.signals import compute_signals, seasonal_values
from backend.app.core.surveillance.synthetic_cases import Scenario, generate_cases
from backend.app.db import init_db, make_engine
from backend.app.models import DataOrigin, Observation, VerificationStatus

TODAY = date(2026, 11, 21)
MISSING_CSV = weather.Path("does/not/exist.csv")


# --------------------------------------------------------------------------
# Weather aggregation and synthetic generators
# --------------------------------------------------------------------------

def test_weekly_aggregation_sums_rain_and_drops_partial_weeks():
    days = pd.date_range("2026-01-05", "2026-01-15")  # Mon 5th .. Thu 15th
    daily = pd.DataFrame(
        {"location": "X", "date": days, RAINFALL: 2.0, TEMP_MEAN: np.arange(len(days), dtype=float)}
    )
    weekly = weather.aggregate_weekly(daily)
    assert list(weekly["week_start"]) == [date(2026, 1, 5)]  # second week is partial
    assert weekly[RAINFALL].iloc[0] == 14.0
    assert weekly[TEMP_MEAN].iloc[0] == 3.0


def test_week_helpers():
    assert weather.week_start(date(2026, 11, 21)) == date(2026, 11, 16)
    assert weather.last_complete_week_start(date(2026, 11, 21)) == date(2026, 11, 9)


def synthetic_weather(locations=("Ikorodu", "Epe"), years=4):
    return weather.synthetic_weekly_weather(
        {name: LAGOS_LGAS[name] for name in locations}, date(2022, 1, 3), date(2022, 1, 3) + timedelta(weeks=52 * years)
    )


def test_synthetic_weather_has_two_rainy_seasons():
    frame = synthetic_weather(years=6)
    month = pd.to_datetime(frame["week_start"]).dt.month
    monthly = frame.groupby(month)[RAINFALL].mean()
    assert monthly[6] > 3 * monthly[1]  # June wet, January dry
    assert monthly[10] > monthly[8]  # minor season after the August break


def test_case_generator_is_deterministic_and_follows_rain():
    frame = synthetic_weather(years=6)
    a, b = generate_cases(frame, scenario=None), generate_cases(frame, scenario=None)
    pd.testing.assert_frame_equal(a, b)
    month = pd.to_datetime(a["week_start"]).dt.month
    monthly = a.groupby(month)[CASES].mean()
    assert monthly[[7, 8]].mean() > monthly[[2, 3]].mean()  # lagged after the June peak


def test_ikorodu_scenario_rises_only_in_ikorodu():
    frame = synthetic_weather(years=3)
    plain = generate_cases(frame, scenario=None)
    story = generate_cases(frame, scenario=Scenario(weeks=6, peak_multiplier=1.9))

    def last(df, name, n=6):
        return df[df["location"] == name].tail(n)["expected"].to_numpy()

    ratio = last(story, "Ikorodu") / last(plain, "Ikorodu")
    assert ratio[-1] == pytest.approx(1.9, rel=1e-3) and np.all(np.diff(ratio) > 0)
    np.testing.assert_allclose(last(story, "Epe"), last(plain, "Epe"))  # other LGAs untouched


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

def weekly_frame(values, start=date(2020, 1, 6), column=CASES):
    index = [start + timedelta(weeks=i) for i in range(len(values))]
    return pd.DataFrame({column: values}, index=index)


def test_seasonal_values_align_to_same_weeks_in_earlier_years():
    frame = weekly_frame(list(range(52 * 3)))
    target = frame.index[-1]
    values = seasonal_values(frame[CASES], target, window_weeks=1)
    # 1, 2 and 3 years back (3 years back only week 0 exists, within +/-1 week).
    assert values == [102.0, 103.0, 104.0, 50.0, 51.0, 52.0, 0.0]


def test_steady_series_is_normal_and_spike_is_high():
    rng = np.random.default_rng(0)
    steady = 100 + rng.normal(0, 5, 52 * 4)
    result = compute_signals(weekly_frame(steady))
    assert result["level"] == "normal"

    spiked = steady.copy()
    spiked[-4:] = [150, 190, 230, 260]
    result = compute_signals(weekly_frame(spiked))
    statuses = {s["key"]: s["status"] for s in result["signals"]}
    assert statuses["cases_vs_seasonal_baseline"] == "high"
    assert statuses["cases_trend_4wk"] == "high"
    assert result["level"] == "high"


def test_tiny_dry_season_rain_is_not_alarming():
    rng = np.random.default_rng(1)
    rain = np.clip(rng.normal(3, 1, 52 * 4), 0, None)
    rain[-8:] = 12  # 4x the usual, but only ~70 mm more over 8 weeks
    frame = weekly_frame(rain, column=RAINFALL)
    frame[CASES] = 50
    rain_signal = next(s for s in compute_signals(frame)["signals"] if s["key"] == "rainfall_8wk_anomaly")
    assert rain_signal["status"] != "high"


def test_temperature_is_context_only():
    frame = weekly_frame([100] * 120)
    frame[TEMP_MEAN] = 26.0
    signal = next(s for s in compute_signals(frame)["signals"] if s["key"] == "temperature_suitability")
    assert signal["favourable"] is True and signal["status"] == "normal"


# --------------------------------------------------------------------------
# Seeding and repository
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as session:
        seed_places(session)
        summary = seed_surveillance(session, today=TODAY, csv_path=MISSING_CSV)
        yield session, summary


def test_seed_without_real_csv_is_labelled_synthetic(seeded):
    session, summary = seeded
    assert summary["weather_origin"] == "SYNTHETIC"
    assert summary["last_week"] == "2026-11-09"
    origins = {
        (metric, origin)
        for metric, origin in session.query(Observation.metric, Observation.data_origin).distinct()
    }
    assert (CASES, DataOrigin.SYNTHETIC) in origins
    assert all(origin is DataOrigin.SYNTHETIC for _, origin in origins)


def test_ikorodu_stands_out_in_latest_week(seeded):
    session, _ = seeded
    long = load_observations(session)
    ids = location_ids(session)
    result = compute_signals(wide_for_location(long, ids["Ikorodu"]), origins_by_metric(long))
    assert result["level"] == "high"
    assert result["as_of_week"] == "2026-11-09"


def test_only_verified_lab_results_reach_surveillance(seeded):
    session, _ = seeded
    ikorodu = location_ids(session)["Ikorodu"]
    for status, value in ((VerificationStatus.VERIFIED, 1.0), (VerificationStatus.PENDING_REVIEW, 5.0)):
        session.add(
            Observation(
                location_id=ikorodu, metric=LAB_POSITIVE, value=value, period="day",
                period_start=date(2026, 11, 11), data_origin=DataOrigin.SYNTHETIC,
                source="test", verification_status=status,
            )
        )
    session.commit()
    long = load_observations(session, location_ids=[ikorodu], metrics=[LAB_POSITIVE])
    assert long["value"].sum() == 1.0
    assert list(long["week_start"]) == [date(2026, 11, 9)]  # daily -> ISO week


def test_real_csv_is_used_when_present(tmp_path):
    csv = tmp_path / "w.csv"
    frame = synthetic_weather(locations=("Ikorodu",), years=1)
    weather.save_weekly_csv(frame, csv, {"source": "Test reanalysis", "retrieved_on": "2026-11-20"})
    loaded, origin, source = load_weather(TODAY, csv)
    assert origin is DataOrigin.REAL
    assert source == "Test reanalysis; retrieved 2026-11-20"
    assert len(loaded) == len(frame)


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        c.post("/api/demo/seed")
        yield c


def test_overview(client):
    body = client.get("/api/surveillance/overview").json()
    assert len(body["locations"]) == 20
    assert body["data_origin"][CASES] == "SYNTHETIC"
    ikorodu = next(r for r in body["locations"] if r["location"]["name"] == "Ikorodu")
    assert ikorodu["level"] == "high"
    assert ikorodu["cases_change_pct"] > 40


def test_location_detail(client):
    ikorodu_id = next(
        r["location"]["id"] for r in client.get("/api/surveillance/overview").json()["locations"]
        if r["location"]["name"] == "Ikorodu"
    )
    body = client.get(f"/api/surveillance/locations/{ikorodu_id}", params={"weeks": 52}).json()
    cases = body["series"][CASES]
    assert len(cases["points"]) == 52
    assert cases["data_origin"] == "SYNTHETIC"
    assert "SYNTHETIC" in cases["sources"][0]
    assert {s["key"] for s in body["signals"]} >= {"cases_vs_seasonal_baseline", "rainfall_8wk_anomaly"}
    assert client.get("/api/surveillance/locations/99999").status_code == 404


def test_sources(client):
    entries = {e["metric"]: e for e in client.get("/api/surveillance/sources").json()["sources"]}
    assert entries[CASES]["data_origin"] == "SYNTHETIC"
    assert entries[CASES]["locations"] == 20
    assert entries[RAINFALL]["data_origin"] in {"REAL", "SYNTHETIC"}
