"""Demo controls: reset the database to the synthetic Lagos scenario."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.app.core.demo_seed import reset_and_seed
from backend.app.db import get_db

router = APIRouter(prefix="/api/demo", tags=["demo"])


@router.post("/seed")
def seed_demo_data(db: Session = Depends(get_db)):
    """Delete ALL scenario data (including lab submissions) and re-seed."""
    counts = reset_and_seed(db)
    return {"status": "seeded", "scenario": "Lagos demo (synthetic facilities)", **counts}
