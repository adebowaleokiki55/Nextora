"""Shared data model used by all five modules.

Module-specific tables (lab submissions, forecasts, stock levels) are added
in their own phases; these are the tables every module shares.
"""

from __future__ import annotations

import enum
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, deferred, mapped_column, relationship

from backend.app.db import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _enum(enum_cls: type[enum.Enum]) -> Enum:
    # Stored as plain strings so SQLite and PostgreSQL behave the same.
    return Enum(enum_cls, native_enum=False, length=32, validate_strings=True)


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

class DataOrigin(str, enum.Enum):
    REAL = "REAL"
    SYNTHETIC = "SYNTHETIC"


class VerificationStatus(str, enum.Enum):
    PENDING_REVIEW = "PENDING_REVIEW"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"
    # For data that does not go through human review (weather, reports).
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AlertSeverity(str, enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, enum.Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"


class ProposalStatus(str, enum.Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"


class AgentRunStatus(str, enum.Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AgentRunMode(str, enum.Enum):
    LIVE = "LIVE"
    OFFLINE_SCRIPTED = "OFFLINE_SCRIPTED"


class ReviewerResult(str, enum.Enum):
    """The qualified reviewer's own reading of the smear (not the model's)."""

    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


class AgentMessageKind(str, enum.Enum):
    THOUGHT = "THOUGHT"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESULT = "TOOL_RESULT"
    HANDOFF = "HANDOFF"
    FINAL = "FINAL"
    ERROR = "ERROR"


# --------------------------------------------------------------------------
# Places
# --------------------------------------------------------------------------

class Location(Base):
    """A monitored area, e.g. a Lagos local government area."""

    __tablename__ = "locations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    state: Mapped[str | None] = mapped_column(String(80))
    country: Mapped[str] = mapped_column(String(80), default="Nigeria")
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    population: Mapped[int | None] = mapped_column(Integer)
    data_origin: Mapped[DataOrigin] = mapped_column(_enum(DataOrigin))
    source: Mapped[str] = mapped_column(String(255))

    facilities: Mapped[list[Facility]] = relationship(back_populates="location")


class Facility(Base):
    """A participating health facility or laboratory."""

    __tablename__ = "facilities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    facility_type: Mapped[str] = mapped_column(String(60), default="primary_health_centre")
    has_laboratory: Mapped[bool] = mapped_column(default=False)
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    data_origin: Mapped[DataOrigin] = mapped_column(_enum(DataOrigin))
    source: Mapped[str] = mapped_column(String(255))

    location: Mapped[Location] = relationship(back_populates="facilities")


# --------------------------------------------------------------------------
# Observations: one row per measured value, any module
# --------------------------------------------------------------------------

class Observation(Base):
    """A single dated value for a location (and optionally a facility).

    Examples of `metric`: cases_confirmed, rainfall_mm, temp_mean_c,
    lab_positive_verified. Every row carries its origin, source and
    verification status.
    """

    __tablename__ = "observations"
    __table_args__ = (
        Index("ix_observations_location_metric_period", "location_id", "metric", "period_start"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"))
    facility_id: Mapped[int | None] = mapped_column(ForeignKey("facilities.id"))
    metric: Mapped[str] = mapped_column(String(60))
    value: Mapped[float] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(30))
    period: Mapped[str] = mapped_column(String(10), default="week")  # day | week | month
    period_start: Mapped[date] = mapped_column(Date)
    period_end: Mapped[date | None] = mapped_column(Date)
    data_origin: Mapped[DataOrigin] = mapped_column(_enum(DataOrigin))
    source: Mapped[str] = mapped_column(String(255))
    verification_status: Mapped[VerificationStatus] = mapped_column(
        _enum(VerificationStatus), default=VerificationStatus.NOT_APPLICABLE
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# --------------------------------------------------------------------------
# Module 5: laboratory submissions and their cell-level results
# --------------------------------------------------------------------------

class LabSubmission(Base):
    """One uploaded blood-smear image and the model's cell-level output.

    Starts as PENDING_REVIEW. Only after a qualified reviewer verifies it
    does it produce observations for surveillance and prediction.
    """

    __tablename__ = "lab_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    facility_id: Mapped[int] = mapped_column(ForeignKey("facilities.id"), index=True)
    location_id: Mapped[int] = mapped_column(ForeignKey("locations.id"), index=True)
    sample_date: Mapped[date] = mapped_column(Date)
    # Lab's own slide/sample number. Never a patient name or identifier.
    sample_reference: Mapped[str | None] = mapped_column(String(60))
    submitted_by: Mapped[str] = mapped_column(String(120))
    image_filename: Mapped[str | None] = mapped_column(String(255))
    image_width: Mapped[int] = mapped_column(Integer)
    image_height: Mapped[int] = mapped_column(Integer)

    model_name: Mapped[str] = mapped_column(String(60))
    model_version: Mapped[str] = mapped_column(String(40))
    candidate_count: Mapped[int] = mapped_column(Integer)
    candidate_infected_count: Mapped[int] = mapped_column(Integer)
    candidate_uninfected_count: Mapped[int] = mapped_column(Integer)
    # Kept in the database so it survives redeploys on hosts with ephemeral disks.
    annotated_png: Mapped[bytes | None] = deferred(mapped_column(LargeBinary))

    status: Mapped[VerificationStatus] = mapped_column(
        _enum(VerificationStatus), default=VerificationStatus.PENDING_REVIEW, index=True
    )
    reviewer_result: Mapped[ReviewerResult | None] = mapped_column(_enum(ReviewerResult))
    reviewed_by: Mapped[str | None] = mapped_column(String(120))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_notes: Mapped[str | None] = mapped_column(Text)

    data_origin: Mapped[DataOrigin] = mapped_column(_enum(DataOrigin))
    source: Mapped[str] = mapped_column(String(255))

    facility: Mapped[Facility] = relationship()
    location: Mapped[Location] = relationship()
    cells: Mapped[list[LabCell]] = relationship(
        back_populates="submission", order_by="LabCell.cell_index", cascade="all, delete-orphan"
    )


class LabCell(Base):
    __tablename__ = "lab_cells"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    submission_id: Mapped[int] = mapped_column(
        ForeignKey("lab_submissions.id", ondelete="CASCADE"), index=True
    )
    cell_index: Mapped[int] = mapped_column(Integer)
    x: Mapped[int] = mapped_column(Integer)
    y: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    predicted_class: Mapped[int] = mapped_column(Integer)  # 0 parasitized, 1 uninfected
    prediction: Mapped[str] = mapped_column(String(40))
    confidence: Mapped[float] = mapped_column(Float)

    submission: Mapped[LabSubmission] = relationship(back_populates="cells")


# --------------------------------------------------------------------------
# Common alert system
# --------------------------------------------------------------------------

class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    # e.g. RISK_FORECAST, SUPPLY_SHORTAGE, LAB_VERIFIED_POSITIVE
    alert_type: Mapped[str] = mapped_column(String(40), index=True)
    # surveillance | prediction | command | supply | lab
    source_module: Mapped[str] = mapped_column(String(20))
    severity: Mapped[AlertSeverity] = mapped_column(_enum(AlertSeverity))
    status: Mapped[AlertStatus] = mapped_column(_enum(AlertStatus), default=AlertStatus.OPEN)
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    facility_id: Mapped[int | None] = mapped_column(ForeignKey("facilities.id"))
    title: Mapped[str] = mapped_column(String(200))
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # SYNTHETIC if any input that raised this alert was synthetic.
    data_origin: Mapped[DataOrigin] = mapped_column(_enum(DataOrigin))


# --------------------------------------------------------------------------
# Human approval queue
# --------------------------------------------------------------------------

class Proposal(Base):
    """A consequential action proposed by an agent; executed only once approved."""

    __tablename__ = "proposals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    # e.g. STOCK_TRANSFER, PUBLIC_MESSAGE
    proposal_type: Mapped[str] = mapped_column(String(40))
    status: Mapped[ProposalStatus] = mapped_column(
        _enum(ProposalStatus), default=ProposalStatus.PENDING, index=True
    )
    title: Mapped[str] = mapped_column(String(200))
    summary: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    proposed_by: Mapped[str] = mapped_column(String(40))
    agent_run_id: Mapped[int | None] = mapped_column(ForeignKey("agent_runs.id"))
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.id"))
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    decided_by: Mapped[str | None] = mapped_column(String(120))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    decision_note: Mapped[str | None] = mapped_column(Text)
    executed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    execution_result: Mapped[dict[str, Any] | None] = mapped_column(JSON)


# --------------------------------------------------------------------------
# Agent runs and their live message feed
# --------------------------------------------------------------------------

class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[AgentRunStatus] = mapped_column(
        _enum(AgentRunStatus), default=AgentRunStatus.RUNNING
    )
    mode: Mapped[AgentRunMode] = mapped_column(_enum(AgentRunMode))
    provider: Mapped[str | None] = mapped_column(String(40))
    model: Mapped[str | None] = mapped_column(String(120))
    trigger: Mapped[str] = mapped_column(String(60))
    location_id: Mapped[int | None] = mapped_column(ForeignKey("locations.id"))
    alert_id: Mapped[int | None] = mapped_column(ForeignKey("alerts.id"))
    summary: Mapped[str | None] = mapped_column(Text)

    messages: Mapped[list[AgentMessage]] = relationship(
        back_populates="run", order_by="AgentMessage.seq"
    )


class AgentMessage(Base):
    __tablename__ = "agent_messages"
    __table_args__ = (Index("ix_agent_messages_run_seq", "run_id", "seq", unique=True),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("agent_runs.id"))
    seq: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    agent: Mapped[str] = mapped_column(String(40))  # Commander, Surveillance, ...
    kind: Mapped[AgentMessageKind] = mapped_column(_enum(AgentMessageKind))
    content: Mapped[str] = mapped_column(Text, default="")
    tool_name: Mapped[str | None] = mapped_column(String(80))
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    run: Mapped[AgentRun] = relationship(back_populates="messages")
