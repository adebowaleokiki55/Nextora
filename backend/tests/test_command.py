import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete, select

from backend.app.config import LLMSettings
from backend.app.core.command.proposals import ProposalError, propose_public_message
from backend.app.db import SessionLocal
from backend.app.main import app
from backend.app.models import AgentMessage, AgentRun, Alert, Proposal, StockLevel, StockTransaction
from backend.app.modules.command import AgentRuntime, get_agent_runtime


def runtime(llm=None, configured=True):
    settings = LLMSettings(provider="fake", api_key="k", model="fake-model") if configured else LLMSettings(provider="none")
    return AgentRuntime(settings, lambda: llm, SessionLocal, step_delay=0)


@pytest.fixture(scope="module")
def seeded_client():
    with TestClient(app) as c:
        c.post("/api/demo/seed")
        with SessionLocal() as s:
            stock_snapshot = {(l.facility_id, l.item_code): l.quantity_on_hand for l in s.scalars(select(StockLevel))}
            alert_snapshot = {a.id: a.status for a in s.scalars(select(Alert))}
        yield c, stock_snapshot, alert_snapshot


@pytest.fixture()
def client(seeded_client):
    """Seeded once per module; each test starts from the same state."""
    c, stock_snapshot, alert_snapshot = seeded_client
    with SessionLocal() as s:
        for model in (AgentMessage, Proposal, AgentRun):
            s.execute(delete(model))
        s.execute(delete(StockTransaction).where(StockTransaction.kind.in_(["TRANSFER_IN", "TRANSFER_OUT"])))
        for level in s.scalars(select(StockLevel)):
            level.quantity_on_hand = stock_snapshot[(level.facility_id, level.item_code)]
        for alert in s.scalars(select(Alert)):
            alert.status = alert_snapshot[alert.id]
        s.commit()
    app.dependency_overrides[get_agent_runtime] = lambda: runtime(configured=False)
    yield c
    app.dependency_overrides.clear()


def location_id(client, name):
    return next(l["id"] for l in client.get("/api/locations").json()["locations"] if l["name"] == name)


def stock(facility_id, item="RDT"):
    with SessionLocal() as s:
        return s.scalar(select(StockLevel.quantity_on_hand).where(StockLevel.facility_id == facility_id, StockLevel.item_code == item))


# --------------------------------------------------------------------------
# Offline scripted mode (the stage backup)
# --------------------------------------------------------------------------

def test_auto_mode_without_provider_runs_offline_script(client):
    body = client.post("/api/command/runs", json={"location_id": location_id(client, "Ikorodu"), "wait": True}).json()
    assert body["status"] == "COMPLETED"
    assert body["mode"] == "OFFLINE_SCRIPTED" and "OFFLINE DEMO MODE" in body["mode_label"]

    agents = {m["agent"] for m in body["messages"]}
    assert agents == {"Commander", "Surveillance", "Supply", "Outreach", "Policy"}
    kinds = {m["kind"] for m in body["messages"]}
    assert {"THOUGHT", "TOOL_CALL", "TOOL_RESULT", "HANDOFF", "FINAL"} <= kinds
    # Every narrated message says it is scripted.
    for m in body["messages"]:
        if m["kind"] in ("THOUGHT", "HANDOFF", "FINAL"):
            assert m["content"].startswith("[Scripted")
    assert "## Data limitations" in body["situation_report"]
    assert "SYNTHETIC" in body["situation_report"]

    pending = client.get("/api/proposals", params={"status": "PENDING", "run_id": body["id"]}).json()["proposals"]
    types = [p["proposal_type"] for p in pending]
    assert types.count("STOCK_TRANSFER") >= 3 and types.count("PUBLIC_MESSAGE") == 1

    alerts = client.get("/api/alerts", params={"location_id": location_id(client, "Ikorodu")}).json()["alerts"]
    assert alerts and all(a["status"] == "ACKNOWLEDGED" for a in alerts)


def test_rerun_does_not_duplicate_pending_transfers(client):
    ikorodu = location_id(client, "Ikorodu")
    client.post("/api/command/runs", json={"location_id": ikorodu, "mode": "offline", "wait": True})
    first = len(client.get("/api/proposals", params={"status": "PENDING", "location_id": ikorodu}).json()["proposals"])
    client.post("/api/command/runs", json={"location_id": ikorodu, "mode": "offline", "wait": True})
    pending = client.get("/api/proposals", params={"status": "PENDING", "location_id": ikorodu}).json()["proposals"]
    transfers = [p for p in pending if p["proposal_type"] == "STOCK_TRANSFER"]
    assert len({json.dumps(p["payload"], sort_keys=True) for p in transfers}) == len(transfers)
    assert len(pending) == first + 1  # only a second message draft was added


def test_low_risk_location_gets_no_actions(client):
    body = client.post("/api/command/runs", json={"location_id": location_id(client, "Surulere"), "mode": "offline", "wait": True}).json()
    assert body["status"] == "COMPLETED"
    assert client.get("/api/proposals", params={"run_id": body["id"]}).json()["proposals"] == []


def test_live_mode_requires_a_provider(client):
    r = client.post("/api/command/runs", json={"location_id": location_id(client, "Ikorodu"), "mode": "live"})
    assert r.status_code == 503
    assert client.post("/api/command/runs", json={"location_id": 99999, "mode": "offline"}).status_code == 404


# --------------------------------------------------------------------------
# Live mode with a fake AI client
# --------------------------------------------------------------------------

def test_live_commander_delegates_and_agents_only_propose(client, fake_llm_factory):
    script = [
        # Commander
        ("Reviewing the situation.", [("get_situation_snapshot", {})]),
        ("", [("delegate_to_surveillance", {"task": "Check signals"})]),
        #   Surveillance
        ("", [("get_surveillance_signals", {}), ("get_risk_forecast", {}), ("get_verified_lab_summary", {})]),
        ("Cases far above usual; rain is normal.", []),
        # Commander
        ("", [("delegate_to_supply", {"task": "Fix RDT shortages"})]),
        #   Supply
        ("", [("get_supply_shortages", {}), ("get_redistribution_plan", {})]),
        ("", [("propose_redistribution", {})]),
        ("Three facilities short; transfers proposed.", []),
        # Commander
        ("", [("delegate_to_outreach", {"task": "Draft a message"})]),
        #   Outreach: first draft too short, corrected after the tool's error
        ("", [("get_outreach_context", {})]),
        ("", [("propose_public_message", {"audience": "Residents", "channel": "SMS", "message": "Use nets."})]),
        ("", [("propose_public_message", {"audience": "Residents of Ikorodu", "channel": "SMS",
                                          "message": "Malaria is rising in Ikorodu. Sleep under a treated net and get tested within 24 hours of a fever."})]),
        ("Drafted one SMS.", []),
        # Commander
        ("", [("delegate_to_policy", {"task": "Write the sitrep"})]),
        #   Policy
        ("", [("get_situation_snapshot", {})]),
        ("## Situation\nHigh risk.\n## Data limitations\nSYNTHETIC cases.", []),
        # Commander
        ("Approve the transfers and the SMS draft.", []),
    ]
    llm = fake_llm_factory(script)
    app.dependency_overrides[get_agent_runtime] = lambda: runtime(llm)

    body = client.post("/api/command/runs", json={"location_id": location_id(client, "Ikorodu"), "mode": "live", "wait": True}).json()
    assert body["status"] == "COMPLETED", body["summary"]
    assert body["mode"] == "LIVE" and body["summary"] == "Approve the transfers and the SMS draft."
    assert body["situation_report"].startswith("## Situation")
    handoffs = [(m["agent"], m["payload"]["to"]) for m in body["messages"] if m["kind"] == "HANDOFF"]
    assert ("Commander", "Supply") in handoffs and ("Supply", "Commander") in handoffs

    # The rejected first draft was returned to the model, not stored.
    results = [m for m in body["messages"] if m["tool_name"] == "propose_public_message" and m["kind"] == "TOOL_RESULT"]
    assert "error" in results[0]["content"] and "created" in results[1]["content"]

    proposals = client.get("/api/proposals", params={"run_id": body["id"]}).json()["proposals"]
    assert all(p["status"] == "PENDING" for p in proposals)
    assert sum(p["proposal_type"] == "PUBLIC_MESSAGE" for p in proposals) == 1
    # The specialists see the shared rules.
    assert "Never invent" in llm.calls[0]["system"]


def test_live_failure_is_recorded(client, fake_llm_factory):
    llm = fake_llm_factory([("", [("transfer_everything", {})])])
    app.dependency_overrides[get_agent_runtime] = lambda: runtime(llm)
    body = client.post("/api/command/runs", json={"location_id": location_id(client, "Ikorodu"), "mode": "live", "wait": True}).json()
    assert body["status"] == "FAILED"
    assert "unapproved tool" in body["summary"]
    assert body["messages"][-1]["kind"] == "ERROR"


# --------------------------------------------------------------------------
# Approval queue: nothing happens until a human approves
# --------------------------------------------------------------------------

def test_approve_executes_transfer_and_reject_does_nothing(client):
    ikorodu = location_id(client, "Ikorodu")
    run = client.post("/api/command/runs", json={"location_id": ikorodu, "mode": "offline", "wait": True}).json()
    transfers = [
        p for p in client.get("/api/proposals", params={"run_id": run["id"]}).json()["proposals"]
        if p["proposal_type"] == "STOCK_TRANSFER"
    ]
    first, second = transfers[0], transfers[1]
    before_from, before_to = stock(first["payload"]["from_facility_id"]), stock(first["payload"]["to_facility_id"])

    assert client.post(f"/api/proposals/{first['id']}/approve", json={"decided_by": "  "}).status_code == 422
    approved = client.post(f"/api/proposals/{first['id']}/approve", json={"decided_by": "Dr Ade", "note": "OK"}).json()
    assert approved["status"] == "EXECUTED" and approved["decided_by"] == "Dr Ade"
    qty = first["payload"]["quantity"]
    assert stock(first["payload"]["from_facility_id"]) == before_from - qty
    assert stock(first["payload"]["to_facility_id"]) == before_to + qty
    assert client.post(f"/api/proposals/{first['id']}/approve", json={"decided_by": "X"}).status_code == 409

    before = stock(second["payload"]["from_facility_id"])
    rejected = client.post(f"/api/proposals/{second['id']}/reject", json={"decided_by": "Dr Ade", "note": "Use central stock"}).json()
    assert rejected["status"] == "REJECTED"
    assert stock(second["payload"]["from_facility_id"]) == before


def test_stale_transfer_fails_safely(client):
    ikorodu = location_id(client, "Ikorodu")
    run = client.post("/api/command/runs", json={"location_id": ikorodu, "mode": "offline", "wait": True}).json()
    transfer = next(
        p for p in client.get("/api/proposals", params={"run_id": run["id"]}).json()["proposals"]
        if p["proposal_type"] == "STOCK_TRANSFER"
    )
    donor = transfer["payload"]["from_facility_id"]
    with SessionLocal() as s:  # the donor's stock drops before anyone approves
        level = s.scalar(select(StockLevel).where(StockLevel.facility_id == donor, StockLevel.item_code == "RDT"))
        level.quantity_on_hand = 10
        s.commit()
    result = client.post(f"/api/proposals/{transfer['id']}/approve", json={"decided_by": "Dr Ade"}).json()
    assert result["status"] == "FAILED" and "out of date" in result["execution_result"]["error"]
    assert stock(donor) == 10


def test_public_message_is_only_queued(client):
    ikorodu = location_id(client, "Ikorodu")
    run = client.post("/api/command/runs", json={"location_id": ikorodu, "mode": "offline", "wait": True}).json()
    message = next(
        p for p in client.get("/api/proposals", params={"run_id": run["id"]}).json()["proposals"]
        if p["proposal_type"] == "PUBLIC_MESSAGE"
    )
    result = client.post(f"/api/proposals/{message['id']}/approve", json={"decided_by": "Comms lead"}).json()
    assert result["status"] == "EXECUTED"
    assert "nothing was actually sent" in result["execution_result"]["note"]


def test_message_validation(client):
    ikorodu = location_id(client, "Ikorodu")
    with SessionLocal() as s:
        with pytest.raises(ProposalError):
            propose_public_message(s, ikorodu, "Residents", "PIGEON", "x" * 100, "test")
        with pytest.raises(ProposalError):
            propose_public_message(s, ikorodu, "Residents", "SMS", "short", "test")
        with pytest.raises(ProposalError):
            propose_public_message(s, ikorodu, "Residents", "SMS", "x" * 481, "test")
        for _ in range(3):
            propose_public_message(s, ikorodu, "Residents", "radio", "x" * 100, "test")
        with pytest.raises(ProposalError) as err:  # cap on drafts awaiting approval
            propose_public_message(s, ikorodu, "Residents", "SMS", "x" * 100, "test")
        assert err.value.status_code == 409


# --------------------------------------------------------------------------
# Live feed (Server-Sent Events) and alerts
# --------------------------------------------------------------------------

def test_event_stream_replays_steps_then_done(client):
    run = client.post("/api/command/runs", json={"location_id": location_id(client, "Ikorodu"), "mode": "offline", "wait": True}).json()
    with client.stream("GET", f"/api/command/runs/{run['id']}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    events = [block for block in body.split("\n\n") if block.startswith("id:") or block.startswith("event: done")]
    messages = [b for b in events if "event: message" in b]
    assert len(messages) == len(run["messages"])
    assert events[-1].startswith("event: done")

    with client.stream("GET", f"/api/command/runs/{run['id']}/events", params={"after": len(run["messages"]) - 2}) as response:
        body = "".join(response.iter_text())
    assert body.count("event: message") == 2


def test_background_run_and_alert_actions(client):
    import time

    app.dependency_overrides[get_agent_runtime] = lambda: runtime(configured=False)
    started = client.post("/api/command/runs", json={"location_id": location_id(client, "Alimosho"), "mode": "offline"}).json()
    assert started["status"] == "RUNNING" and started["events_url"].endswith("/events")
    for _ in range(100):
        if client.get(f"/api/command/runs/{started['id']}").json()["status"] != "RUNNING":
            break
        time.sleep(0.1)
    assert client.get(f"/api/command/runs/{started['id']}").json()["status"] == "COMPLETED"
    assert client.get("/api/command/runs").json()["runs"][0]["id"] == started["id"]

    alert = client.get("/api/alerts").json()["alerts"][0]
    assert client.post(f"/api/alerts/{alert['id']}/resolve").json()["status"] == "RESOLVED"
    assert alert["id"] not in [a["id"] for a in client.get("/api/alerts").json()["alerts"]]
