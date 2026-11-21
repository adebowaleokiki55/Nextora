"""Start and execute command-centre runs (live AI or offline scripted)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.agents.command_centre import COMMANDER, POLICY, CommandCentre
from backend.app.agents.offline_script import OfflineScript
from backend.app.agents.run_log import RunLogger
from backend.app.config import LLMSettings
from backend.app.core.command.context import alert_to_dict, get_location
from backend.app.core.llm import LLMClient, LLMError
from backend.app.models import (
    AgentMessage,
    AgentMessageKind,
    AgentRun,
    AgentRunMode,
    AgentRunStatus,
    Alert,
    AlertStatus,
)

MODES = ("auto", "live", "offline")


class RunError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def choose_mode(requested: str, llm: LLMSettings) -> AgentRunMode:
    if requested not in MODES:
        raise RunError(422, f"Mode must be one of {', '.join(MODES)}.")
    if requested == "offline":
        return AgentRunMode.OFFLINE_SCRIPTED
    if requested == "live":
        if not llm.configured:
            raise RunError(503, "No AI provider is configured; use mode 'offline' or set LLM_PROVIDER and LLM_API_KEY.")
        return AgentRunMode.LIVE
    return AgentRunMode.LIVE if llm.configured else AgentRunMode.OFFLINE_SCRIPTED


def create_run(session: Session, location_id: int, mode: AgentRunMode, llm: LLMSettings) -> AgentRun:
    get_location(session, location_id)
    risk_alert = session.scalar(
        select(Alert.id).where(
            Alert.alert_type == "RISK_FORECAST",
            Alert.location_id == location_id,
            Alert.status != AlertStatus.RESOLVED,
        )
    )
    run = AgentRun(
        status=AgentRunStatus.RUNNING,
        mode=mode,
        provider=llm.provider if mode is AgentRunMode.LIVE else None,
        model=llm.model if mode is AgentRunMode.LIVE else None,
        trigger="run_response",
        location_id=location_id,
        alert_id=risk_alert,
    )
    session.add(run)
    session.commit()
    return run


def execute_run(
    run_id: int,
    session_factory: Callable[[], Session],
    client_factory: Callable[[], LLMClient] | None = None,
    delay: float = 0.6,
) -> None:
    """Run the agents to completion, recording status and every step."""
    logger = RunLogger(session_factory, run_id)
    with session_factory() as session:
        run = session.get(AgentRun, run_id)
        mode, location_id = run.mode, run.location_id

    try:
        if mode is AgentRunMode.LIVE:
            client = client_factory() if client_factory else None
            if client is None:
                raise LLMError(503, "No AI client available for a live run.")
            centre = CommandCentre(session_factory, logger, run_id, location_id, client)
            logger.log(COMMANDER, AgentMessageKind.THOUGHT, f"Live run using {client.runtime_label} ({client.model}).")
            summary = centre.run_live()
        else:
            centre = CommandCentre(session_factory, logger, run_id, location_id)
            summary = OfflineScript(centre, delay=delay).run()
        status = AgentRunStatus.COMPLETED
    except Exception as exc:  # noqa: BLE001 - every failure is recorded on the run
        detail = exc.detail if isinstance(exc, LLMError) else f"{type(exc).__name__}: {exc}"
        logger.log(COMMANDER, AgentMessageKind.ERROR, f"Run failed: {detail}")
        summary = f"Run failed: {detail}"
        status = AgentRunStatus.FAILED

    with session_factory() as session:
        run = session.get(AgentRun, run_id)
        run.status = status
        run.summary = summary
        run.finished_at = datetime.now(timezone.utc)
        if status is AgentRunStatus.COMPLETED:
            # The response is under way: open alerts here are now acknowledged.
            for alert in session.scalars(
                select(Alert).where(Alert.location_id == location_id, Alert.status == AlertStatus.OPEN)
            ):
                alert.status = AlertStatus.ACKNOWLEDGED
        session.commit()


def start_in_background(run_id: int, session_factory, client_factory, delay: float = 0.6) -> threading.Thread:
    thread = threading.Thread(
        target=execute_run, args=(run_id, session_factory, client_factory, delay), daemon=True, name=f"agent-run-{run_id}"
    )
    thread.start()
    return thread


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------

def message_to_dict(m: AgentMessage) -> dict[str, Any]:
    return {
        "seq": m.seq,
        "created_at": m.created_at.isoformat(),
        "agent": m.agent,
        "kind": m.kind.value,
        "tool_name": m.tool_name,
        "content": m.content,
        "payload": m.payload,
    }


def run_to_dict(session: Session, run: AgentRun, with_messages: bool = False) -> dict[str, Any]:
    data = {
        "id": run.id,
        "status": run.status.value,
        "mode": run.mode.value,
        "mode_label": (
            "Live AI agents"
            if run.mode is AgentRunMode.LIVE
            else "OFFLINE DEMO MODE: scripted sequence calling the real tools, no AI model"
        ),
        "provider": run.provider,
        "model": run.model,
        "location": {"id": run.location_id, "name": get_location(session, run.location_id).name},
        "alert": alert_to_dict(session.get(Alert, run.alert_id)) if run.alert_id else None,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "summary": run.summary,
    }
    if with_messages:
        messages = list(run.messages)
        data["messages"] = [message_to_dict(m) for m in messages]
        report = [
            m for m in messages
            if m.agent == POLICY and m.kind is AgentMessageKind.HANDOFF
        ]
        data["situation_report"] = report[-1].content if report else None
    return data
