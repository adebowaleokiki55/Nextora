"""Deterministic stock calculations. No database, no LLM.

Weeks of stock = on hand / expected weekly demand, where expected demand is
recent consumption scaled by the location's case forecast (Module 2).

Policy (in weeks of expected demand):
    below min_weeks          -> LOW (reorder point)
    below critical_weeks     -> CRITICAL
    zero on hand             -> STOCKOUT
    above surplus_weeks      -> SURPLUS
    shortage quantity        =  restore target_weeks, rounded UP to whole packs
    a donor may give         =  stock above donor_keep_weeks of ITS OWN demand,
                                rounded DOWN to whole packs
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from backend.app.core.supply.items import SupplyItem

STATUS_RANK = {"STOCKOUT": 0, "CRITICAL": 1, "LOW": 2, "OK": 3, "SURPLUS": 4}
SHORTAGE_STATUSES = ("STOCKOUT", "CRITICAL", "LOW")


@dataclass(frozen=True)
class StockPolicy:
    critical_weeks: float = 1.0
    min_weeks: float = 2.0
    target_weeks: float = 6.0
    donor_keep_weeks: float = 6.0
    surplus_weeks: float = 10.0
    max_transfer_km: float = 40.0
    demand_multiplier_bounds: tuple[float, float] = (0.5, 3.0)
    # Precaution: where the malaria risk score is at least this high, planned
    # demand never falls below recent consumption, even if the forecast dips.
    precautionary_risk_score: float = 50.0


DEFAULT_POLICY = StockPolicy()


@dataclass(frozen=True)
class StockPosition:
    facility_id: int
    facility_name: str
    location_id: int
    location_name: str
    latitude: float
    longitude: float
    item: SupplyItem
    on_hand: int
    recent_weekly_consumption: float
    demand_multiplier: float = 1.0  # from the case forecast
    location_risk_score: float | None = None


@dataclass(frozen=True)
class Assessment:
    position: StockPosition
    expected_weekly_demand: float
    weeks_of_stock: float | None
    status: str
    shortage_quantity: int
    transferable_quantity: int

    @property
    def is_shortage(self) -> bool:
        return self.status in SHORTAGE_STATUSES


@dataclass(frozen=True)
class Transfer:
    item_code: str
    from_facility_id: int
    from_facility_name: str
    to_facility_id: int
    to_facility_name: str
    quantity: int
    distance_km: float
    rationale: str


@dataclass(frozen=True)
class ReplenishmentOrder:
    item_code: str
    facility_id: int
    facility_name: str
    quantity: int
    rationale: str


@dataclass
class RedistributionPlan:
    transfers: list[Transfer] = field(default_factory=list)
    replenishment: list[ReplenishmentOrder] = field(default_factory=list)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def round_up_to_pack(quantity: float, pack_size: int) -> int:
    if quantity <= 0:
        return 0
    return int(math.ceil(quantity / pack_size) * pack_size)


def round_down_to_pack(quantity: float, pack_size: int) -> int:
    if quantity <= 0:
        return 0
    return int(math.floor(quantity / pack_size) * pack_size)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return round(2 * radius * math.asin(math.sqrt(a)), 2)


def demand_multiplier(
    forecast_mean: float | None,
    recent_mean: float | None,
    risk_score: float | None = None,
    policy: StockPolicy = DEFAULT_POLICY,
) -> float:
    """Forecast cases (next weeks) relative to recent cases, within bounds.

    In elevated-risk areas the multiplier is floored at 1.0 (precaution).
    """
    if not forecast_mean or not recent_mean or recent_mean <= 0:
        return 1.0
    low, high = policy.demand_multiplier_bounds
    multiplier = min(max(forecast_mean / recent_mean, low), high)
    if risk_score is not None and risk_score >= policy.precautionary_risk_score:
        multiplier = max(multiplier, 1.0)
    return round(multiplier, 3)


# --------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------

def assess(position: StockPosition, policy: StockPolicy = DEFAULT_POLICY) -> Assessment:
    demand = max(position.recent_weekly_consumption, 0.0) * position.demand_multiplier
    pack = position.item.pack_size
    on_hand = max(position.on_hand, 0)

    if demand <= 0:
        weeks = None
        status = "STOCKOUT" if on_hand == 0 else "OK"
        return Assessment(position, 0.0, weeks, status, 0, 0)

    weeks = round(on_hand / demand, 2)
    if on_hand == 0:
        status = "STOCKOUT"
    elif weeks < policy.critical_weeks:
        status = "CRITICAL"
    elif weeks < policy.min_weeks:
        status = "LOW"
    elif weeks > policy.surplus_weeks:
        status = "SURPLUS"
    else:
        status = "OK"

    shortage = round_up_to_pack(policy.target_weeks * demand - on_hand, pack) if status in SHORTAGE_STATUSES else 0
    transferable = (
        round_down_to_pack(on_hand - policy.donor_keep_weeks * demand, pack)
        if status in ("OK", "SURPLUS")
        else 0
    )
    return Assessment(position, round(demand, 2), weeks, status, shortage, transferable)


def priority_key(assessment: Assessment) -> tuple:
    """Most urgent first: worst status, then higher area risk, then fewer weeks left."""
    risk = assessment.position.location_risk_score or 0.0
    weeks = assessment.weeks_of_stock if assessment.weeks_of_stock is not None else -1.0
    return (STATUS_RANK[assessment.status], -risk, weeks, assessment.position.facility_id)


# --------------------------------------------------------------------------
# Redistribution
# --------------------------------------------------------------------------

def plan_redistribution(assessments: list[Assessment], policy: StockPolicy = DEFAULT_POLICY) -> RedistributionPlan:
    """Greedy, deterministic plan: most urgent recipient first, nearest donor first.

    Donors only give stock above their own safety level, in whole packs and
    within max_transfer_km. Whatever cannot be covered becomes a
    replenishment order to the central medical store.
    """
    plan = RedistributionPlan()
    by_item: dict[str, list[Assessment]] = {}
    for assessment in assessments:
        by_item.setdefault(assessment.position.item.code, []).append(assessment)

    for item_code in sorted(by_item):
        group = by_item[item_code]
        needs = sorted((a for a in group if a.is_shortage and a.shortage_quantity > 0), key=priority_key)
        available = {a.position.facility_id: a.transferable_quantity for a in group if a.transferable_quantity > 0}
        donors = {a.position.facility_id: a for a in group if a.transferable_quantity > 0}

        for need in needs:
            recipient = need.position
            pack = recipient.item.pack_size
            remaining = need.shortage_quantity

            candidates = sorted(
                (
                    (haversine_km(recipient.latitude, recipient.longitude, d.position.latitude, d.position.longitude), donor_id)
                    for donor_id, d in donors.items()
                    if donor_id != recipient.facility_id
                ),
            )
            for distance, donor_id in candidates:
                if remaining <= 0:
                    break
                if distance > policy.max_transfer_km or available[donor_id] < pack:
                    continue
                quantity = min(remaining, round_down_to_pack(available[donor_id], pack))
                if quantity <= 0:
                    continue
                donor = donors[donor_id]
                available[donor_id] -= quantity
                remaining -= quantity
                plan.transfers.append(
                    Transfer(
                        item_code=item_code,
                        from_facility_id=donor_id,
                        from_facility_name=donor.position.facility_name,
                        to_facility_id=recipient.facility_id,
                        to_facility_name=recipient.facility_name,
                        quantity=quantity,
                        distance_km=distance,
                        rationale=(
                            f"{recipient.facility_name} has {_weeks_text(need)} of {recipient.item.unit} "
                            f"left ({need.status}); {donor.position.facility_name} is {distance:.1f} km away "
                            f"and holds {donor.weeks_of_stock:.1f} weeks, keeping at least "
                            f"{policy.donor_keep_weeks:.0f} weeks after the transfer."
                        ),
                    )
                )

            if remaining > 0:
                plan.replenishment.append(
                    ReplenishmentOrder(
                        item_code=item_code,
                        facility_id=recipient.facility_id,
                        facility_name=recipient.facility_name,
                        quantity=round_up_to_pack(remaining, pack),
                        rationale=(
                            f"No nearby facility (within {policy.max_transfer_km:.0f} km) can spare the remaining "
                            f"{remaining} {recipient.item.unit}; order from the central medical store."
                        ),
                    )
                )

    return plan


def _weeks_text(assessment: Assessment) -> str:
    if assessment.weeks_of_stock is None:
        return "no measured demand"
    return f"{assessment.weeks_of_stock:.1f} weeks"
