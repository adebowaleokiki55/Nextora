from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.demo_seed import seed_places, seed_surveillance
from backend.app.core.prediction.evaluation import crossing_metrics, error_metrics
from backend.app.core.prediction.features import FEATURES, location_rows, seasonal_reference
from backend.app.core.prediction.forecast import risk_level, run_forecasts, summarise
from backend.app.core.prediction.model import (
    Splits,
    choose_alert_cutoff,
    exceedance_probability,
    load_bundle,
    save_bundle,
    train_and_evaluate,
)
from backend.app.core.prediction.training_data import load_training_wides
from backend.app.core.surveillance.metrics import CASES, RAINFALL
from backend.app.db import init_db, make_engine
from backend.app.models import Alert, AlertSeverity, AlertStatus, Forecast, Location


def weekly(values, start=date(2018, 1, 1), **extra):
    index = [start + timedelta(weeks=i) for i in range(len(values))]
    frame = pd.DataFrame({CASES: values, **extra}, index=index)
    return frame


# --------------------------------------------------------------------------
# Features: no peeking into the future
# --------------------------------------------------------------------------

def test_features_do_not_change_when_future_data_changes():
    rng = np.random.default_rng(0)
    n = 52 * 4
    base = weekly(rng.poisson(100, n).astype(float), **{RAINFALL: rng.gamma(2, 10, n)})
    altered = base.copy()
    altered.iloc[-10:] = altered.iloc[-10:] * 5  # change the last 10 weeks only

    for horizon in (1, 4):
        a = location_rows(base, 1, horizon).set_index("origin_week")
        b = location_rows(altered, 1, horizon).set_index("origin_week")
        cutoff = base.index[-11]  # origins before the altered weeks
        cols = FEATURES
        pd.testing.assert_frame_equal(a.loc[:pd.Timestamp(cutoff), cols], b.loc[:pd.Timestamp(cutoff), cols])


def test_target_and_baselines_line_up():
    frame = weekly(list(range(52 * 3)))
    rows = location_rows(frame, 7, horizon=2).set_index("origin_week")
    origin = pd.Timestamp(frame.index[120])
    row = rows.loc[origin]
    assert row["target_cases"] == 122
    assert row["baseline_persistence"] == 120
    assert row["baseline_seasonal_naive"] == 122 - 52
    assert row["target_week"] == origin + pd.Timedelta(weeks=2)


def test_threshold_is_mean_plus_two_sd_of_earlier_years():
    frame = weekly([10.0] * 52 + [20.0] * 52 + [30.0] * 52)
    ref = seasonal_reference(frame[CASES])
    week = 52 * 2 + 10  # a week in year 3: history = years 1 and 2 (+/- 2 weeks)
    history = [10.0] * 5 + [20.0] * 5
    assert ref["threshold"].iloc[week] == pytest.approx(np.mean(history) + 2 * np.std(history, ddof=1))
    assert np.isnan(ref["threshold"].iloc[30])  # first year: not enough history


def test_live_rows_have_future_thresholds():
    frame = weekly([50.0 + (i % 52) for i in range(52 * 3)])
    rows = location_rows(frame, 1, horizon=4)
    last = rows.tail(1).iloc[0]
    assert pd.isna(last["target_cases"])
    assert not pd.isna(last["threshold"])


# --------------------------------------------------------------------------
# Metrics, probabilities, risk
# --------------------------------------------------------------------------

def test_error_and_crossing_metrics():
    assert error_metrics(np.array([1.0, 3.0, np.nan]), np.array([2.0, 1.0, 5.0])) == {
        "mae": 1.5, "rmse": round(np.sqrt(2.5), 2), "n": 2
    }
    m = crossing_metrics(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert (m["precision"], m["recall"], m["true_positives"], m["actual_crossings"]) == (0.5, 0.5, 1, 2)
    assert crossing_metrics(np.array([0, 0]), np.array([0, 0]))["precision"] is None


def test_exceedance_probability_uses_residual_distribution():
    residuals = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    prob = exceedance_probability(np.log1p([100.0, 100.0]), residuals, np.array([100.0, np.nan]))
    assert prob[0] == pytest.approx(0.6)  # residuals >= 0
    assert np.isnan(prob[1])


def test_alert_cutoff_maximises_validation_f1():
    probability = np.array([0.1, 0.3, 0.35, 0.8, 0.9])
    actual = np.array([False, False, True, True, True])
    assert choose_alert_cutoff(probability, actual) == 0.35
    assert choose_alert_cutoff(probability, np.zeros(5, bool)) == 0.5


def test_risk_summary():
    forecasts = [
        {"target_week": date(2026, 1, 5), "exceedance_probability": 0.3, "alert_cutoff": 0.4},
        {"target_week": date(2026, 1, 12), "exceedance_probability": 0.85, "alert_cutoff": 0.4},
    ]
    summary = summarise(forecasts)
    assert summary == {
        "risk_score": 85.0, "risk_level": "VERY_HIGH", "early_warning": True,
        "first_warning_week": date(2026, 1, 12),
    }
    assert [risk_level(s) for s in (5, 20, 55, 80, None)] == ["LOW", "MODERATE", "HIGH", "VERY_HIGH", "UNKNOWN"]


# --------------------------------------------------------------------------
# End to end on a small slice of the demo scenario
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as session:
        seed_places(session)
        seed_surveillance(session, today=date(2026, 11, 21))
        wides, origin = load_training_wides(session)
        names = {loc.id: loc.name for loc in session.scalars(select(Location))}
        keep = {i: w for i, w in wides.items() if names[i] in {"Ikorodu", "Epe", "Surulere", "Kosofe"}}
        bundle, report, backtest = train_and_evaluate(
            keep, Splits(date(2023, 12, 31), date(2024, 12, 29)), params={"n_estimators": 150}
        )
        directory = tmp_path_factory.mktemp("model")
        save_bundle(bundle, directory)
        yield session, load_bundle(directory), report, backtest, origin


def test_report_compares_model_with_all_baselines(trained):
    _, bundle, report, backtest, origin = trained
    assert origin == "SYNTHETIC"
    assert set(report) == {"h1", "h2", "h3", "h4"}
    for result in report.values():
        for key in ("model", "baseline_persistence", "baseline_seasonal_naive", "baseline_hist_mean"):
            assert result[key]["mae"] is not None and result[key]["rmse"] is not None
            assert "precision" in result[key]["crossing"]
        assert 0.05 <= result["alert_cutoff"] <= 0.7
    # Test period strictly after validation; no shuffling.
    assert pd.to_datetime(backtest["target_week"]).min() > pd.Timestamp("2024-12-29")
    assert bundle.version.startswith("xgb-")


def test_run_forecasts_raises_and_resolves_alerts(trained):
    session, bundle, *_ = trained
    result = run_forecasts(session, bundle)
    assert "Ikorodu" in result["early_warnings"]
    rows = session.scalars(select(Forecast)).all()
    assert {r.horizon for r in rows} == {1, 2, 3, 4}
    assert all(r.data_origin.value == "SYNTHETIC" for r in rows)

    ikorodu = session.scalar(select(Location).where(Location.name == "Ikorodu"))
    alert = session.scalar(select(Alert).where(Alert.alert_type == "RISK_FORECAST", Alert.location_id == ikorodu.id))
    assert alert.severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)
    assert "SYNTHETIC" in alert.message

    # Re-running replaces forecasts for the same origin week and keeps one alert.
    run_forecasts(session, bundle)
    assert len(session.scalars(select(Forecast).where(Forecast.location_id == ikorodu.id)).all()) == 4
    assert session.query(Alert).filter(Alert.alert_type == "RISK_FORECAST", Alert.location_id == ikorodu.id).count() == 1

    # If the risk subsides, the alert is resolved rather than left open.
    for model in bundle.models.values():
        model.alert_cutoff = 1.01
    run_forecasts(session, bundle)
    session.refresh(alert)
    assert alert.status is AlertStatus.RESOLVED


# --------------------------------------------------------------------------
# API (uses the committed trained model)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        c.post("/api/demo/seed")
        yield c


def test_risk_endpoint(client):
    body = client.get("/api/prediction/risk").json()
    assert len(body["locations"]) == 20
    top = body["locations"][0]
    assert top["location"]["name"] == "Ikorodu"
    assert top["early_warning"] and top["data_origin"] == "SYNTHETIC"
    assert len(top["forecasts"]) == 4
    assert "mean + 2 SD" in body["risk_score_definition"]


def test_location_and_evaluation_endpoints(client):
    ikorodu = client.get("/api/prediction/risk").json()["locations"][0]["location"]["id"]
    body = client.get(f"/api/prediction/locations/{ikorodu}").json()
    assert len(body["history"]) == 26 and body["backtest"]
    evaluation = client.get("/api/prediction/evaluation").json()
    assert "NOT" in evaluation["honesty_note"]
    assert set(evaluation["horizons"]["h1"]) >= {"model", "baseline_persistence", "baseline_seasonal_naive", "baseline_hist_mean"}
    assert client.post("/api/prediction/run").json()["early_warnings"]
