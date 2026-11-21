"""Tracked malaria commodities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SupplyItem:
    code: str
    name: str
    unit: str
    pack_size: int  # transfers and orders move whole packs
    per_case: float  # units used per confirmed case (demo assumption)


ITEMS: dict[str, SupplyItem] = {
    item.code: item
    for item in (
        # Several patients are tested for every confirmed case.
        SupplyItem("RDT", "Malaria rapid diagnostic test kits", "tests", 25, 2.2),
        SupplyItem("ACT_AL", "Artemether-lumefantrine (ACT) treatment courses", "courses", 30, 1.0),
        SupplyItem("ARTESUNATE_INJ", "Injectable artesunate (severe malaria)", "vials", 10, 0.12),
    )
}


def get_item(code: str) -> SupplyItem:
    try:
        return ITEMS[code]
    except KeyError as exc:
        raise ValueError(f"Unknown supply item '{code}'. Known: {', '.join(ITEMS)}.") from exc
