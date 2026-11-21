"""The human approval queue.

Agents can only *propose*. Proposals are stored as PENDING and nothing
happens until a named person approves one; approving executes it.

- STOCK_TRANSFER: quantities come from the deterministic redistribution
  plan, never from the language model. Executing re-checks the donor.
- REPLENISHMENT_ORDER: recorded only (demo: no real order is placed).
- PUBLIC_MESSAGE: text may be drafted by an agent; executing queues it in
  the outbox (demo: nothing is actually sent).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core.command.context import get_location, redistribution_plan
from backend.app.core.supply.service import SupplyError, apply_transfer, run_supply_scan
from backend.app.models import Alert, AlertStatus, Proposal, ProposalStatus

STOCK_TRANSFER = "STOCK_TRANSFER"
REPLENISHMENT_ORDER = "REPLENISHMENT_ORDER"
PUBLIC_MESSAGE = "PUBLIC_MESSAGE"

MESSAGE_CHANNELS = ("SMS", "RADIO", "COMMUNITY_HEALTH_WORKERS", "SOCIAL_MEDIA")
CHANNEL_LABELS = {
    "SMS": "SMS",
    "RADIO": "Radio",
    "COMMUNITY_HEALTH_WORKERS": "Community health worker",
    "SOCIAL_MEDIA": "Social media",
}
MESSAGE_MIN, MESSAGE_MAX = 40, 480
MAX_PENDING_MESSAGES = 3


class ProposalError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def proposal_to_dict(p: Proposal) -> dict[str, Any]:
    return {
        "id": p.id,
        "created_at": p.created_at.isoformat(),
        "proposal_type": p.proposal_type,
        "status": p.status.value,
        "title": p.title,
        "summary": p.summary,
        "payload": p.payload,
        "proposed_by": p.proposed_by,
        "agent_run_id": p.agent_run_id,
        "alert_id": p.alert_id,
        "location_id": p.location_id,
        "decided_by": p.decided_by,
        "decided_at": p.decided_at.isoformat() if p.decided_at else None,
        "decision_note": p.decision_note,
        "executed_at": p.executed_at.isoformat() if p.executed_at else None,
        "execution_result": p.execution_result,
    }


def _shortage_alert_id(session: Session, facility_id: int, item_code: str) -> int | None:
    for alert in session.scalars(
        select(Alert).where(
            Alert.alert_type == "SUPPLY_SHORTAGE",
            Alert.facility_id == facility_id,
            Alert.status != AlertStatus.RESOLVED,
        )
    ):
        if alert.payload.get("item_code") == item_code:
            return alert.id
    return None


def _risk_alert_id(session: Session, location_id: int) -> int | None:
    return session.scalar(
        select(Alert.id).where(
            Alert.alert_type == "RISK_FORECAST",
            Alert.location_id == location_id,
            Alert.status != AlertStatus.RESOLVED,
        )
    )


def _pending(session: Session, proposal_type: str, location_id: int) -> list[Proposal]:
    return list(
        session.scalars(
            select(Proposal).where(
                Proposal.proposal_type == proposal_type,
                Proposal.location_id == location_id,
                Proposal.status == ProposalStatus.PENDING,
            )
        )
    )


# --------------------------------------------------------------------------
# Creating proposals
# --------------------------------------------------------------------------

def propose_redistribution(
    session: Session,
    location_id: int,
    proposed_by: str,
    agent_run_id: int | None = None,
) -> dict[str, Any]:
    """Turn the current deterministic plan into PENDING proposals.

    Identical pending proposals are not duplicated.
    """
    plan = redistribution_plan(session, location_id)
    created, skipped = [], []
    pending_transfers = {
        (p.payload["item_code"], p.payload["from_facility_id"], p.payload["to_facility_id"], p.payload["quantity"])
        for p in _pending(session, STOCK_TRANSFER, location_id)
    }
    pending_orders = {
        (p.payload["item_code"], p.payload["facility_id"], p.payload["quantity"])
        for p in _pending(session, REPLENISHMENT_ORDER, location_id)
    }

    for t in plan["transfers"]:
        key = (t["item_code"], t["from_facility_id"], t["to_facility_id"], t["quantity"])
        if key in pending_transfers:
            skipped.append(t)
            continue
        proposal = Proposal(
            proposal_type=STOCK_TRANSFER,
            title=f"Transfer {t['quantity']} {t['item_code']} from {t['from_facility_name']} to {t['to_facility_name']}",
            summary=t["rationale"],
            payload=dict(t),
            proposed_by=proposed_by,
            agent_run_id=agent_run_id,
            alert_id=_shortage_alert_id(session, t["to_facility_id"], t["item_code"]),
            location_id=location_id,
        )
        session.add(proposal)
        created.append(proposal)

    for order in plan["replenishment"]:
        key = (order["item_code"], order["facility_id"], order["quantity"])
        if key in pending_orders:
            skipped.append(order)
            continue
        proposal = Proposal(
            proposal_type=REPLENISHMENT_ORDER,
            title=f"Order {order['quantity']} {order['item_code']} for {order['facility_name']}",
            summary=order["rationale"],
            payload=dict(order),
            proposed_by=proposed_by,
            agent_run_id=agent_run_id,
            alert_id=_shortage_alert_id(session, order["facility_id"], order["item_code"]),
            location_id=location_id,
        )
        session.add(proposal)
        created.append(proposal)

    session.commit()
    return {
        "created": [proposal_to_dict(p) for p in created],
        "already_pending": len(skipped),
        "note": "Quantities come from the deterministic plan. Nothing moves until a human approves.",
    }


def propose_public_message(
    session: Session,
    location_id: int,
    audience: str,
    channel: str,
    message: str,
    proposed_by: str,
    language: str = "English",
    agent_run_id: int | None = None,
) -> dict[str, Any]:
    location = get_location(session, location_id)
    channel = (channel or "").strip().upper().replace(" ", "_")
    message = (message or "").strip()
    audience = (audience or "").strip()[:200]
    if channel not in MESSAGE_CHANNELS:
        raise ProposalError(422, f"Channel must be one of {', '.join(MESSAGE_CHANNELS)}.")
    if not MESSAGE_MIN <= len(message) <= MESSAGE_MAX:
        raise ProposalError(422, f"Message must be {MESSAGE_MIN}-{MESSAGE_MAX} characters (got {len(message)}).")
    if not audience:
        raise ProposalError(422, "Audience is required.")
    if len(_pending(session, PUBLIC_MESSAGE, location_id)) >= MAX_PENDING_MESSAGES:
        raise ProposalError(409, f"{location.name} already has {MAX_PENDING_MESSAGES} messages awaiting approval.")

    proposal = Proposal(
        proposal_type=PUBLIC_MESSAGE,
        title=f"{CHANNEL_LABELS[channel]} message to {audience} ({location.name})",
        summary=message,
        payload={"audience": audience, "channel": channel, "language": language[:40], "message": message},
        proposed_by=proposed_by,
        agent_run_id=agent_run_id,
        alert_id=_risk_alert_id(session, location_id),
        location_id=location_id,
    )
    session.add(proposal)
    session.commit()
    return {
        "created": proposal_to_dict(proposal),
        "note": "Draft only. A human must review and approve before anything is sent.",
    }


# --------------------------------------------------------------------------
# Deciding
# --------------------------------------------------------------------------

def _get_pending(session: Session, proposal_id: int) -> Proposal:
    proposal = session.get(Proposal, proposal_id)
    if proposal is None:
        raise ProposalError(404, f"Proposal {proposal_id} was not found.")
    if proposal.status is not ProposalStatus.PENDING:
        raise ProposalError(409, f"Proposal {proposal_id} has already been decided ({proposal.status.value}).")
    return proposal


def _decider(decided_by: str) -> str:
    decided_by = (decided_by or "").strip()
    if not decided_by:
        raise ProposalError(422, "The name of the person deciding is required.")
    return decided_by[:120]


def approve(session: Session, proposal_id: int, decided_by: str, note: str | None = None) -> Proposal:
    """Approve and execute. Execution failures leave the proposal FAILED."""
    proposal = _get_pending(session, proposal_id)
    now = datetime.now(timezone.utc)
    proposal.status = ProposalStatus.APPROVED
    proposal.decided_by = _decider(decided_by)
    proposal.decided_at = now
    proposal.decision_note = (note or "").strip()[:2000] or None

    try:
        result = _execute(session, proposal)
        proposal.status = ProposalStatus.EXECUTED
    except SupplyError as exc:
        session.rollback()
        proposal = session.get(Proposal, proposal_id)
        proposal.status = ProposalStatus.FAILED
        proposal.decided_by = _decider(decided_by)
        proposal.decided_at = now
        proposal.decision_note = (note or "").strip()[:2000] or None
        result = {"error": exc.detail}

    proposal.executed_at = datetime.now(timezone.utc)
    proposal.execution_result = result
    session.commit()

    if proposal.proposal_type == STOCK_TRANSFER and proposal.status is ProposalStatus.EXECUTED:
        run_supply_scan(session)  # resolve shortage alerts that are now covered
    return proposal


def reject(session: Session, proposal_id: int, decided_by: str, note: str | None = None) -> Proposal:
    proposal = _get_pending(session, proposal_id)
    proposal.status = ProposalStatus.REJECTED
    proposal.decided_by = _decider(decided_by)
    proposal.decided_at = datetime.now(timezone.utc)
    proposal.decision_note = (note or "").strip()[:2000] or None
    session.commit()
    return proposal


def _execute(session: Session, proposal: Proposal) -> dict[str, Any]:
    payload = proposal.payload
    if proposal.proposal_type == STOCK_TRANSFER:
        return apply_transfer(
            session,
            payload["from_facility_id"],
            payload["to_facility_id"],
            payload["item_code"],
            int(payload["quantity"]),
            reference=f"proposal:{proposal.id}",
        )
    if proposal.proposal_type == REPLENISHMENT_ORDER:
        return {
            "status": "order_recorded",
            "note": "Demo: the order is recorded here only; no request was sent to a real medical store.",
        }
    if proposal.proposal_type == PUBLIC_MESSAGE:
        return {
            "status": "queued_for_dispatch",
            "channel": payload["channel"],
            "note": "Demo: the approved message is queued in Nextora only; nothing was actually sent.",
        }
    raise SupplyError(422, f"Unknown proposal type {proposal.proposal_type}.")
