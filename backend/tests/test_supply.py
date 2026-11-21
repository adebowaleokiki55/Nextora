from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.demo_seed import reset_and_seed
from backend.app.core.supply.calculations import (
    StockPolicy,
    StockPosition,
    assess,
    demand_multiplier,
    haversine_km,
    plan_redistribution,
    priority_key,
    round_down_to_pack,
    round_up_to_pack,
)
from backend.app.core.supply.items import ITEMS
from backend.app.core.supply.service import (
    SupplyError,
    apply_transfer,
    assess_all,
    run_supply_scan,
)
from backend.app.db import init_db, make_engine
from backend.app.models import Alert, AlertSeverity, AlertStatus, Facility, StockLevel, StockTransaction

RDT = ITEMS["RDT"]  # pack of 25


def position(fid, on_hand, weekly, lat=6.6, lon=3.5, mult=1.0, risk=None, item=RDT):
    return StockPosition(fid, f"F{fid}", 1, "L", lat, lon, item, on_hand, weekly, mult, risk)


# --------------------------------------------------------------------------
# Pure calculations
# --------------------------------------------------------------------------

def test_pack_rounding():
    assert round_up_to_pack(1, 25) == 25 and round_up_to_pack(50, 25) == 50 and round_up_to_pack(-5, 25) == 0
    assert round_down_to_pack(74, 25) == 50 and round_down_to_pack(-1, 25) == 0


def test_haversine():
    assert haversine_km(6.6, 3.5, 6.6, 3.5) == 0
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.19, abs=0.05)


@pytest.mark.parametrize(
    "on_hand, status, shortage, spare",
    [
        (0, "STOCKOUT", 600, 0),     # 6 weeks x 100
        (80, "CRITICAL", 525, 0),    # 600 - 80 = 520 -> 525 (whole packs)
        (150, "LOW", 450, 0),
        (500, "OK", 0, 0),           # 5 weeks: nothing above the 6-week keep
        (790, "OK", 0, 175),         # 7.9 weeks: 190 above keep -> 175
        (1200, "SURPLUS", 0, 600),
    ],
)
def test_assess_status_and_quantities(on_hand, status, shortage, spare):
    a = assess(position(1, on_hand, 100))
    assert (a.status, a.shortage_quantity, a.transferable_quantity) == (status, shortage, spare)


def test_forecast_multiplier_scales_demand():
    a = assess(position(1, 300, 100, mult=1.5))
    assert a.expected_weekly_demand == 150 and a.weeks_of_stock == 2.0 and a.status == "OK"


def test_demand_multiplier_bounds_and_precaution():
    assert demand_multiplier(300, 100) == 3.0
    assert demand_multiplier(10, 100) == 0.5
    assert demand_multiplier(66, 100) == 0.66
    assert demand_multiplier(66, 100, risk_score=90) == 1.0  # never plan less in a high-risk area
    assert demand_multiplier(None, 100) == 1.0


def test_no_demand_means_no_judgement():
    a = assess(position(1, 40, 0))
    assert a.status == "OK" and a.weeks_of_stock is None and a.transferable_quantity == 0


def test_priority_puts_high_risk_areas_first():
    low_risk = assess(position(1, 50, 100, risk=10))
    high_risk = assess(position(2, 50, 100, risk=90))
    stockout = assess(position(3, 0, 100, risk=0))
    assert [a.position.facility_id for a in sorted([low_risk, high_risk, stockout], key=priority_key)] == [3, 2, 1]


def test_plan_nearest_donor_first_and_never_below_keep():
    recipient = assess(position(1, 0, 100, lat=6.60, lon=3.50))                    # needs 600
    near = assess(position(2, 1000, 100, lat=6.61, lon=3.50))                      # spare 400
    far = assess(position(3, 2000, 100, lat=6.75, lon=3.50))                       # spare 1400, ~17 km
    plan = plan_redistribution([recipient, near, far])
    assert [(t.from_facility_id, t.quantity) for t in plan.transfers] == [(2, 400), (3, 200)]
    assert not plan.replenishment
    for t in plan.transfers:
        assert t.quantity % RDT.pack_size == 0
    # Donor 2 keeps exactly its 6-week safety stock.
    assert 1000 - 400 >= 6 * 100


def test_plan_respects_distance_and_orders_the_rest():
    recipient = assess(position(1, 0, 100, lat=6.6, lon=3.5))
    distant = assess(position(2, 5000, 100, lat=7.2, lon=3.5))  # ~67 km
    plan = plan_redistribution([recipient, distant], StockPolicy(max_transfer_km=40))
    assert plan.transfers == []
    assert [(o.facility_id, o.quantity) for o in plan.replenishment] == [(1, 600)]


def test_plan_handles_items_separately_and_skips_shortage_donors():
    act = ITEMS["ACT_AL"]
    rdt_need = assess(position(1, 0, 100))
    act_surplus = assess(position(2, 3000, 100, item=act))
    short_donor = assess(position(3, 10, 100))
    plan = plan_redistribution([rdt_need, act_surplus, short_donor])
    assert plan.transfers == []  # ACT cannot cover RDT; a short facility never donates
    assert plan.replenishment[0].item_code == "RDT"


# --------------------------------------------------------------------------
# Demo scenario, alerts and transfers
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def seeded_engine():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as session:
        reset_and_seed(session)
    return engine


@pytest.fixture()
def scenario(seeded_engine):
    """The seeded scenario; every change a test makes is rolled back afterwards."""
    connection = seeded_engine.connect()
    outer = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        connection.close()


def by_name(session, name):
    return session.scalar(select(Facility).where(Facility.name == name))


def test_scenario_three_ikorodu_rdt_shortages(scenario):
    shortages = [a for a in assess_all(scenario) if a.is_shortage]
    assert sorted((a.position.facility_name, a.position.item.code, a.status) for a in shortages) == [
        ("Ikorodu Demo Health Centre A", "RDT", "CRITICAL"),
        ("Ikorodu Demo Health Centre B", "RDT", "STOCKOUT"),
        ("Ikorodu Demo Health Centre C", "RDT", "LOW"),
    ]
    assert all(a.position.demand_multiplier >= 1.0 for a in shortages)  # high-risk precaution


def test_scan_alerts_are_prioritised_and_need_approval(scenario):
    alerts = scenario.scalars(select(Alert).where(Alert.alert_type == "SUPPLY_SHORTAGE")).all()
    assert len(alerts) == 3
    assert {a.severity for a in alerts} <= {AlertSeverity.HIGH, AlertSeverity.CRITICAL}  # risk bump
    for alert in alerts:
        assert "Requires human approval" in alert.message and "SYNTHETIC" in alert.message
        assert alert.payload["recommended_transfers"]


def test_plan_covers_all_shortages_without_breaking_donors(scenario):
    result = run_supply_scan(scenario)
    assert result["plan"]["replenishment"] == []
    covered = {}
    for t in result["plan"]["transfers"]:
        covered[t["to_facility_name"]] = covered.get(t["to_facility_name"], 0) + t["quantity"]
    for shortage in result["shortages"]:
        assert covered[shortage["facility"]["name"]] >= shortage["shortage_quantity"]


def test_apply_transfer_moves_stock_and_resolves_alert(scenario):
    a = by_name(scenario, "Ikorodu Demo Health Centre A")
    lab = by_name(scenario, "Ikorodu Demo Laboratory")
    need = next(x for x in assess_all(scenario, facility_id=a.id, item_code="RDT"))
    before = {f.id: scenario.scalar(select(StockLevel.quantity_on_hand).where(
        StockLevel.facility_id == f.id, StockLevel.item_code == "RDT")) for f in (a, lab)}

    result = apply_transfer(scenario, lab.id, a.id, "RDT", need.shortage_quantity, "proposal:test", today=date(2026, 11, 21))
    scenario.commit()
    assert result["from_on_hand_after"] == before[lab.id] - need.shortage_quantity
    assert result["to_on_hand_after"] == before[a.id] + need.shortage_quantity
    ledger = scenario.scalars(select(StockTransaction).where(StockTransaction.reference == "proposal:test")).all()
    assert sorted(t.quantity for t in ledger) == [-need.shortage_quantity, need.shortage_quantity]

    run_supply_scan(scenario)
    alert = scenario.scalar(select(Alert).where(Alert.alert_type == "SUPPLY_SHORTAGE", Alert.facility_id == a.id))
    assert alert.status is AlertStatus.RESOLVED


def test_apply_transfer_rechecks_donor(scenario):
    a = by_name(scenario, "Ikorodu Demo Health Centre A")
    b = by_name(scenario, "Ikorodu Demo Health Centre B")
    lab = by_name(scenario, "Ikorodu Demo Laboratory")
    with pytest.raises(SupplyError) as err:
        apply_transfer(scenario, lab.id, a.id, "RDT", 100_000, "x")
    assert err.value.status_code == 409
    with pytest.raises(SupplyError):
        apply_transfer(scenario, lab.id, a.id, "RDT", 30, "x")  # not whole packs
    with pytest.raises(SupplyError):
        apply_transfer(scenario, a.id, a.id, "RDT", 25, "x")
    with pytest.raises(SupplyError):
        apply_transfer(scenario, b.id, a.id, "RDT", 25, "x")  # a stocked-out facility has nothing to give


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from backend.app.main import app

    with TestClient(app) as c:
        c.post("/api/demo/seed")
        yield c


def test_supply_endpoints(client):
    assert {i["code"] for i in client.get("/api/supply/items").json()["items"]} == set(ITEMS)
    assert "never below 1.0" in client.get("/api/supply/policy").json()["explanation"]

    shortages = client.get("/api/supply/shortages").json()["shortages"]
    assert [s["status"] for s in shortages] == ["STOCKOUT", "CRITICAL", "LOW"]

    plan = client.get("/api/supply/redistribution", params={"item_code": "RDT"}).json()
    assert plan["requires_human_approval"] and plan["transfers"]

    low = client.get("/api/supply/status", params={"status": "LOW"}).json()["stock"]
    assert len(low) == 1
    assert client.get("/api/supply/status", params={"item_code": "NOPE"}).status_code == 422

    facility_id = shortages[0]["facility"]["id"]
    detail = client.get(f"/api/supply/facilities/{facility_id}").json()
    assert detail["ledger"] and detail["stock"]
    assert client.get("/api/supply/facilities/99999").status_code == 404
    assert len(client.post("/api/supply/scan").json()["shortages"]) == 3
