from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError, StatementError
from sqlalchemy.orm import Session

from backend.app.db import init_db, make_engine
from backend.app.models import (
    AgentMessage,
    AgentMessageKind,
    AgentRun,
    AgentRunMode,
    Alert,
    AlertSeverity,
    AlertStatus,
    DataOrigin,
    Facility,
    Location,
    Observation,
    Proposal,
    ProposalStatus,
    VerificationStatus,
)


@pytest.fixture
def session():
    engine = make_engine("sqlite://")
    init_db(engine)
    with Session(engine) as s:
        yield s


def seed_place(s):
    loc = Location(
        name="Ikorodu", state="Lagos", latitude=6.62, longitude=3.51,
        data_origin=DataOrigin.REAL, source="test",
    )
    fac = Facility(
        name="PHC A", location=loc, latitude=6.6, longitude=3.5,
        data_origin=DataOrigin.SYNTHETIC, source="demo",
    )
    s.add_all([loc, fac])
    s.commit()
    return loc, fac


def test_shared_tables_round_trip(session):
    loc, fac = seed_place(session)
    session.add(
        Observation(
            location_id=loc.id, metric="rainfall_mm", value=12.5, unit="mm",
            period_start=date(2026, 6, 1), data_origin=DataOrigin.REAL, source="NASA POWER",
        )
    )
    alert = Alert(
        alert_type="RISK_FORECAST", source_module="prediction", severity=AlertSeverity.HIGH,
        location_id=loc.id, title="Rising risk", message="m", data_origin=DataOrigin.SYNTHETIC,
    )
    run = AgentRun(mode=AgentRunMode.OFFLINE_SCRIPTED, trigger="run_response", location_id=loc.id)
    session.add_all([alert, run])
    session.flush()
    session.add_all(
        [
            AgentMessage(
                run_id=run.id, seq=1, agent="Commander",
                kind=AgentMessageKind.THOUGHT, content="start",
            ),
            Proposal(
                proposal_type="STOCK_TRANSFER", title="Move RDTs", summary="s",
                proposed_by="Supply", agent_run_id=run.id, alert_id=alert.id,
                payload={"qty": 200},
            ),
        ]
    )
    session.commit()

    obs = session.query(Observation).one()
    assert obs.verification_status is VerificationStatus.NOT_APPLICABLE
    assert session.query(Alert).one().status is AlertStatus.OPEN
    proposal = session.query(Proposal).one()
    assert proposal.status is ProposalStatus.PENDING
    assert proposal.payload == {"qty": 200}
    assert [m.content for m in session.get(AgentRun, run.id).messages] == ["start"]
    assert fac.location.name == "Ikorodu"


def test_foreign_keys_enforced(session):
    session.add(
        Observation(
            location_id=999, metric="x", value=1, period_start=date(2026, 1, 1),
            data_origin=DataOrigin.REAL, source="s",
        )
    )
    with pytest.raises(IntegrityError):
        session.commit()


def test_data_origin_is_required(session):
    session.add(Location(name="X", latitude=0, longitude=0, source="s"))
    with pytest.raises(IntegrityError):
        session.commit()


def test_data_origin_rejects_unknown_values(session):
    session.add(Location(name="Y", latitude=0, longitude=0, source="s", data_origin="MADE_UP"))
    with pytest.raises((StatementError, LookupError)):
        session.commit()
