"""Module 3: AI command centre API — runs, live feed, approvals, alerts."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agents.orchestrator import (
    RunError,
    choose_mode,
    create_run,
    execute_run,
    message_to_dict,
    run_to_dict,
    start_in_background,
)
from backend.app.config import LLMSettings, get_settings
from backend.app.core.command.context import ContextError, alert_to_dict
from backend.app.core.command.proposals import ProposalError, approve, proposal_to_dict, reject
from backend.app.core.llm import LLMClient, create_llm_client
from backend.app.db import SessionLocal, get_db
from backend.app.models import (
    AgentMessage,
    AgentRun,
    AgentRunStatus,
    Alert,
    AlertStatus,
    Proposal,
    ProposalStatus,
)

router = APIRouter(prefix="/api", tags=["command centre"])

STREAM_POLL_SECONDS = 0.4
STREAM_MAX_SECONDS = 15 * 60


@dataclass
class AgentRuntime:
    llm: LLMSettings
    client_factory: Callable[[], LLMClient]
    session_factory: Callable[[], Session]
    step_delay: float = 0.6  # pacing for the offline script, so the feed reads live


def get_agent_runtime() -> AgentRuntime:
    """Tests override this with a fake AI client and no delay."""
    settings = get_settings()
    return AgentRuntime(settings.llm, lambda: create_llm_client(settings.llm), SessionLocal)


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

class RunRequest(BaseModel):
    location_id: int
    mode: Literal["auto", "live", "offline"] = "auto"
    wait: bool = False  # run inline and return when finished (scripts/tests)


@router.post("/command/runs", status_code=201)
def start_run(
    request: RunRequest,
    db: Session = Depends(get_db),
    runtime: AgentRuntime = Depends(get_agent_runtime),
):
    """Start the agents for one location. Stream progress from /events."""
    try:
        mode = choose_mode(request.mode, runtime.llm)
        run = create_run(db, request.location_id, mode, runtime.llm)
    except (RunError, ContextError) as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc

    if request.wait:
        execute_run(run.id, runtime.session_factory, runtime.client_factory, delay=0)
        db.expire_all()
        return run_to_dict(db, db.get(AgentRun, run.id), with_messages=True)

    start_in_background(run.id, runtime.session_factory, runtime.client_factory, runtime.step_delay)
    return {**run_to_dict(db, run), "events_url": f"/api/command/runs/{run.id}/events"}


@router.get("/command/runs")
def list_runs(location_id: int | None = Query(None), limit: int = Query(20, ge=1, le=100), db: Session = Depends(get_db)):
    query = select(AgentRun).order_by(AgentRun.id.desc()).limit(limit)
    if location_id is not None:
        query = query.where(AgentRun.location_id == location_id)
    return {"runs": [run_to_dict(db, run) for run in db.scalars(query)]}


@router.get("/command/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    run = db.get(AgentRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} was not found.")
    return run_to_dict(db, run, with_messages=True)


@router.get("/command/runs/{run_id}/events")
def run_events(run_id: int, after: int = Query(0, ge=0), runtime: AgentRuntime = Depends(get_agent_runtime)):
    """Server-Sent Events: one `message` event per agent step, then `done`."""
    with runtime.session_factory() as session:
        if session.get(AgentRun, run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run {run_id} was not found.")

    def stream():
        last = after
        started = time.monotonic()
        while time.monotonic() - started < STREAM_MAX_SECONDS:
            with runtime.session_factory() as session:
                messages = session.scalars(
                    select(AgentMessage)
                    .where(AgentMessage.run_id == run_id, AgentMessage.seq > last)
                    .order_by(AgentMessage.seq)
                ).all()
                for message in messages:
                    last = message.seq
                    yield f"id: {message.seq}\nevent: message\ndata: {json.dumps(message_to_dict(message))}\n\n"
                run = session.get(AgentRun, run_id)
                if run.status is not AgentRunStatus.RUNNING and not messages:
                    yield f"event: done\ndata: {json.dumps(run_to_dict(session, run))}\n\n"
                    return
            yield ": keep-alive\n\n"
            time.sleep(STREAM_POLL_SECONDS)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --------------------------------------------------------------------------
# Approval queue
# --------------------------------------------------------------------------

class Decision(BaseModel):
    decided_by: str
    note: str | None = None


def _proposal_error(exc: ProposalError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.get("/proposals")
def list_proposals(
    status: ProposalStatus | None = Query(None),
    location_id: int | None = Query(None),
    run_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    query = select(Proposal).order_by(Proposal.id.desc())
    if status is not None:
        query = query.where(Proposal.status == status)
    if location_id is not None:
        query = query.where(Proposal.location_id == location_id)
    if run_id is not None:
        query = query.where(Proposal.agent_run_id == run_id)
    return {"proposals": [proposal_to_dict(p) for p in db.scalars(query)]}


@router.get("/proposals/{proposal_id}")
def get_proposal(proposal_id: int, db: Session = Depends(get_db)):
    proposal = db.get(Proposal, proposal_id)
    if proposal is None:
        raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} was not found.")
    return proposal_to_dict(proposal)


@router.post("/proposals/{proposal_id}/approve")
def approve_proposal(proposal_id: int, decision: Decision, db: Session = Depends(get_db)):
    """Approve and execute. A transfer re-checks stock and may end FAILED."""
    try:
        return proposal_to_dict(approve(db, proposal_id, decision.decided_by, decision.note))
    except ProposalError as exc:
        raise _proposal_error(exc) from exc


@router.post("/proposals/{proposal_id}/reject")
def reject_proposal(proposal_id: int, decision: Decision, db: Session = Depends(get_db)):
    try:
        return proposal_to_dict(reject(db, proposal_id, decision.decided_by, decision.note))
    except ProposalError as exc:
        raise _proposal_error(exc) from exc


# --------------------------------------------------------------------------
# Common alert system
# --------------------------------------------------------------------------

@router.get("/alerts")
def list_alerts(
    status: AlertStatus | None = Query(None, description="Default: all unresolved alerts"),
    location_id: int | None = Query(None),
    source_module: str | None = Query(None),
    db: Session = Depends(get_db),
):
    query = select(Alert).order_by(Alert.created_at.desc(), Alert.id.desc())
    query = query.where(Alert.status == status) if status else query.where(Alert.status != AlertStatus.RESOLVED)
    if location_id is not None:
        query = query.where(Alert.location_id == location_id)
    if source_module:
        query = query.where(Alert.source_module == source_module)
    alerts = list(db.scalars(query))
    severity_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    alerts.sort(key=lambda a: severity_rank[a.severity.value])
    return {"alerts": [alert_to_dict(a) for a in alerts]}


@router.post("/alerts/{alert_id}/{action}")
def update_alert(alert_id: int, action: Literal["acknowledge", "resolve"], db: Session = Depends(get_db)):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} was not found.")
    alert.status = AlertStatus.ACKNOWLEDGED if action == "acknowledge" else AlertStatus.RESOLVED
    db.commit()
    return alert_to_dict(alert)
