"""Supply data access, shortage scan, alerts and transfer execution."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend.app.core.prediction.forecast import latest_forecasts, summarise, forecast_to_dict
from backend.app.core.supply.calculations import (
    DEFAULT_POLICY,
    SHORTAGE_STATUSES,
    Assessment,
    RedistributionPlan,
    StockPolicy,
    StockPosition,
    assess,
    demand_multiplier,
    plan_redistribution,
    priority_key,
)
from backend.app.core.supply.items import ITEMS, get_item
from backend.app.core.surveillance.metrics import CASES
from backend.app.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    DataOrigin,
    Facility,
    Observation,
    StockLevel,
    StockTransaction,
)

RECENT_WEEKS = 8
ALERT_TYPE = "SUPPLY_SHORTAGE"
SEVERITY_ORDER = [AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL]
BASE_SEVERITY = {"LOW": AlertSeverity.MEDIUM, "CRITICAL": AlertSeverity.HIGH, "STOCKOUT": AlertSeverity.CRITICAL}


class SupplyError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# --------------------------------------------------------------------------
# Building positions
# --------------------------------------------------------------------------

def ledger_end(session: Session) -> date | None:
    return session.scalar(select(func.max(StockTransaction.occurred_on)))


def recent_consumption(session: Session, as_of: date | None = None) -> dict[tuple[int, str], float]:
    """Average weekly consumption over the last RECENT_WEEKS weeks of the ledger."""
    as_of = as_of or ledger_end(session)
    if as_of is None:
        return {}
    start = as_of - timedelta(weeks=RECENT_WEEKS) + timedelta(days=1)
    rows = session.execute(
        select(StockTransaction.facility_id, StockTransaction.item_code, func.sum(StockTransaction.quantity))
        .where(
            StockTransaction.kind == "CONSUMPTION",
            StockTransaction.occurred_on >= start,
            StockTransaction.occurred_on <= as_of,
        )
        .group_by(StockTransaction.facility_id, StockTransaction.item_code)
    )
    return {(facility, item): -float(total) / RECENT_WEEKS for facility, item, total in rows}


def location_demand_context(session: Session) -> dict[int, dict[str, Any]]:
    """Per location: demand multiplier from the case forecast, and risk score."""
    context: dict[int, dict[str, Any]] = {}
    for location_id, rows in latest_forecasts(session).items():
        forecasts = [forecast_to_dict(r) for r in rows]
        origin_week = rows[0].origin_week
        recent = session.scalars(
            select(Observation.value).where(
                Observation.location_id == location_id,
                Observation.metric == CASES,
                Observation.period_start > origin_week - timedelta(weeks=RECENT_WEEKS),
                Observation.period_start <= origin_week,
            )
        ).all()
        forecast_mean = sum(f["predicted_cases"] for f in forecasts) / len(forecasts)
        recent_mean = sum(recent) / len(recent) if recent else None
        risk_score = summarise(forecasts)["risk_score"]
        context[location_id] = {
            "demand_multiplier": demand_multiplier(forecast_mean, recent_mean, risk_score),
            "forecast_ratio": round(forecast_mean / recent_mean, 3) if recent_mean else None,
            "risk_score": risk_score,
            "forecast_mean_cases": round(forecast_mean, 1),
            "recent_mean_cases": round(recent_mean, 1) if recent_mean else None,
        }
    return context


def load_positions(
    session: Session,
    item_code: str | None = None,
    location_id: int | None = None,
    facility_id: int | None = None,
) -> list[StockPosition]:
    query = select(StockLevel).join(Facility)
    if item_code:
        get_item(item_code)
        query = query.where(StockLevel.item_code == item_code)
    if location_id is not None:
        query = query.where(Facility.location_id == location_id)
    if facility_id is not None:
        query = query.where(StockLevel.facility_id == facility_id)

    consumption = recent_consumption(session)
    context = location_demand_context(session)
    positions = []
    for level in session.scalars(query.order_by(StockLevel.facility_id, StockLevel.item_code)):
        facility = level.facility
        ctx = context.get(facility.location_id, {})
        positions.append(
            StockPosition(
                facility_id=facility.id,
                facility_name=facility.name,
                location_id=facility.location_id,
                location_name=facility.location.name,
                latitude=facility.latitude,
                longitude=facility.longitude,
                item=get_item(level.item_code),
                on_hand=level.quantity_on_hand,
                recent_weekly_consumption=round(consumption.get((facility.id, level.item_code), 0.0), 2),
                demand_multiplier=ctx.get("demand_multiplier", 1.0),
                location_risk_score=ctx.get("risk_score"),
            )
        )
    return positions


def assess_all(session: Session, policy: StockPolicy = DEFAULT_POLICY, **filters) -> list[Assessment]:
    return [assess(position, policy) for position in load_positions(session, **filters)]


def assessment_to_dict(a: Assessment) -> dict[str, Any]:
    p = a.position
    return {
        "facility": {"id": p.facility_id, "name": p.facility_name},
        "location": {"id": p.location_id, "name": p.location_name},
        "item": {"code": p.item.code, "name": p.item.name, "unit": p.item.unit, "pack_size": p.item.pack_size},
        "on_hand": p.on_hand,
        "recent_weekly_consumption": p.recent_weekly_consumption,
        "demand_multiplier": p.demand_multiplier,
        "expected_weekly_demand": a.expected_weekly_demand,
        "weeks_of_stock": a.weeks_of_stock,
        "status": a.status,
        "shortage_quantity": a.shortage_quantity,
        "transferable_quantity": a.transferable_quantity,
        "location_risk_score": p.location_risk_score,
    }


def plan_to_dict(plan: RedistributionPlan) -> dict[str, Any]:
    return {
        "transfers": [t.__dict__ for t in plan.transfers],
        "replenishment": [r.__dict__ for r in plan.replenishment],
    }


# --------------------------------------------------------------------------
# Shortage scan and alerts
# --------------------------------------------------------------------------

def _severity(assessment: Assessment) -> AlertSeverity:
    base = BASE_SEVERITY[assessment.status]
    risk = assessment.position.location_risk_score or 0
    if risk >= 50:  # prioritise facilities in areas with elevated malaria risk
        return SEVERITY_ORDER[min(SEVERITY_ORDER.index(base) + 1, len(SEVERITY_ORDER) - 1)]
    return base


def run_supply_scan(session: Session, policy: StockPolicy = DEFAULT_POLICY) -> dict[str, Any]:
    """Assess every facility and item, raise/update/resolve shortage alerts."""
    assessments = assess_all(session, policy)
    plan = plan_redistribution(assessments, policy)
    origin_by_facility = {
        level.facility_id: level.data_origin
        for level in session.scalars(select(StockLevel))
    }

    open_alerts = {
        (alert.facility_id, alert.payload.get("item_code")): alert
        for alert in session.scalars(
            select(Alert).where(Alert.alert_type == ALERT_TYPE, Alert.status != AlertStatus.RESOLVED)
        )
    }

    shortages = sorted((a for a in assessments if a.is_shortage), key=priority_key)
    for a in assessments:
        key = (a.position.facility_id, a.position.item.code)
        existing = open_alerts.get(key)
        if not a.is_shortage:
            if existing is not None:
                existing.status = AlertStatus.RESOLVED
                existing.payload = {**existing.payload, "resolved_reason": f"Stock now {a.status}."}
            continue
        _upsert_shortage_alert(session, a, plan, existing, origin_by_facility.get(a.position.facility_id, DataOrigin.SYNTHETIC))

    session.commit()
    return {
        "assessed": len(assessments),
        "shortages": [assessment_to_dict(a) for a in shortages],
        "plan": plan_to_dict(plan),
    }


def _upsert_shortage_alert(
    session: Session,
    a: Assessment,
    plan: RedistributionPlan,
    existing: Alert | None,
    origin: DataOrigin,
) -> None:
    p = a.position
    transfers = [t for t in plan.transfers if t.to_facility_id == p.facility_id and t.item_code == p.item.code]
    orders = [o for o in plan.replenishment if o.facility_id == p.facility_id and o.item_code == p.item.code]

    recommendation = "; ".join(
        [f"transfer {t.quantity} from {t.from_facility_name} ({t.distance_km:.1f} km)" for t in transfers]
        + [f"order {o.quantity} from the central medical store" for o in orders]
    )
    weeks = "none left" if a.status == "STOCKOUT" else f"{a.weeks_of_stock:.1f} weeks left"
    message = (
        f"{p.facility_name} ({p.location_name}) has {p.on_hand} {p.item.unit} of {p.item.name} "
        f"({weeks} at an expected {a.expected_weekly_demand:.0f} a week). "
        f"Needed to restore {DEFAULT_POLICY.target_weeks:.0f} weeks of stock: {a.shortage_quantity} {p.item.unit}."
        + (f" Recommended: {recommendation}." if recommendation else "")
        + (" Requires human approval before any transfer." if transfers else "")
        + (" Based on SYNTHETIC stock data." if origin is DataOrigin.SYNTHETIC else "")
    )
    payload = {
        "item_code": p.item.code,
        "status": a.status,
        "on_hand": p.on_hand,
        "weeks_of_stock": a.weeks_of_stock,
        "expected_weekly_demand": a.expected_weekly_demand,
        "shortage_quantity": a.shortage_quantity,
        "location_risk_score": p.location_risk_score,
        "recommended_transfers": [t.__dict__ for t in transfers],
        "recommended_orders": [o.__dict__ for o in orders],
    }
    title = f"{get_item(p.item.code).name}: {a.status.lower()} at {p.facility_name}"
    severity = _severity(a)

    if existing is None:
        session.add(
            Alert(
                alert_type=ALERT_TYPE,
                source_module="supply",
                severity=severity,
                location_id=p.location_id,
                facility_id=p.facility_id,
                title=title,
                message=message,
                payload=payload,
                data_origin=origin,
            )
        )
    else:
        existing.severity, existing.title, existing.message, existing.payload = severity, title, message, payload


# --------------------------------------------------------------------------
# Transfer execution (called only after a human approves a proposal)
# --------------------------------------------------------------------------

def _level(session: Session, facility_id: int, item_code: str) -> StockLevel:
    level = session.scalar(
        select(StockLevel).where(StockLevel.facility_id == facility_id, StockLevel.item_code == item_code)
    )
    if level is None:
        raise SupplyError(404, f"Facility {facility_id} does not stock {item_code}.")
    return level


def apply_transfer(
    session: Session,
    from_facility_id: int,
    to_facility_id: int,
    item_code: str,
    quantity: int,
    reference: str,
    policy: StockPolicy = DEFAULT_POLICY,
    today: date | None = None,
) -> dict[str, Any]:
    """Move stock between facilities after re-checking the donor *now*.

    Stock may have changed since the plan was made, so the donor's current
    transferable quantity is recomputed; the transfer is refused if it
    would take the donor below its own safety level.
    """
    item = get_item(item_code)
    if from_facility_id == to_facility_id:
        raise SupplyError(422, "A facility cannot transfer stock to itself.")
    if quantity <= 0 or quantity % item.pack_size:
        raise SupplyError(422, f"Quantity must be a positive multiple of the pack size ({item.pack_size}).")

    donor_level = _level(session, from_facility_id, item_code)
    recipient_level = _level(session, to_facility_id, item_code)
    donor = assess(load_positions(session, item_code=item_code, facility_id=from_facility_id)[0], policy)
    if quantity > donor.transferable_quantity:
        raise SupplyError(
            409,
            f"{donor.position.facility_name} can now spare only {donor.transferable_quantity} {item.unit} "
            f"(asked for {quantity}). The plan is out of date; re-run the supply scan.",
        )

    day = today or date.today()
    donor_level.quantity_on_hand -= quantity
    recipient_level.quantity_on_hand += quantity
    for facility_id, signed, kind in (
        (from_facility_id, -quantity, "TRANSFER_OUT"),
        (to_facility_id, quantity, "TRANSFER_IN"),
    ):
        session.add(
            StockTransaction(
                facility_id=facility_id,
                item_code=item_code,
                occurred_on=day,
                kind=kind,
                quantity=signed,
                reference=reference,
                data_origin=donor_level.data_origin,
                source="Approved redistribution",
            )
        )
    session.flush()
    return {
        "item_code": item_code,
        "quantity": quantity,
        "from_facility_id": from_facility_id,
        "from_on_hand_after": donor_level.quantity_on_hand,
        "to_facility_id": to_facility_id,
        "to_on_hand_after": recipient_level.quantity_on_hand,
        "reference": reference,
    }


def facility_detail(session: Session, facility_id: int, weeks: int = 12) -> dict[str, Any]:
    facility = session.get(Facility, facility_id)
    if facility is None:
        raise SupplyError(404, f"Facility {facility_id} was not found.")
    end = ledger_end(session) or date.today()
    start = end - timedelta(weeks=weeks)
    ledger = session.scalars(
        select(StockTransaction)
        .where(StockTransaction.facility_id == facility_id, StockTransaction.occurred_on > start)
        .order_by(StockTransaction.occurred_on, StockTransaction.id)
    ).all()
    return {
        "facility": {
            "id": facility.id,
            "name": facility.name,
            "location": facility.location.name,
            "latitude": facility.latitude,
            "longitude": facility.longitude,
            "data_origin": facility.data_origin.value,
        },
        "stock": [assessment_to_dict(a) for a in assess_all(session, facility_id=facility_id)],
        "ledger": [
            {
                "date": t.occurred_on.isoformat(),
                "item_code": t.item_code,
                "kind": t.kind,
                "quantity": t.quantity,
                "reference": t.reference,
                "data_origin": t.data_origin.value,
            }
            for t in ledger
        ],
        "items": {code: item.name for code, item in ITEMS.items()},
        "shortage_statuses": list(SHORTAGE_STATUSES),
    }
