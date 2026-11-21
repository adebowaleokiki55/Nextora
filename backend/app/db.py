"""Shared database engine and session handling.

SQLite locally, PostgreSQL in deployment; switched with DATABASE_URL.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.config import DATA_DIR, get_settings


class Base(DeclarativeBase):
    pass


def make_engine(url: str) -> Engine:
    if url.startswith("sqlite"):
        in_memory = url in {"sqlite://", "sqlite:///:memory:"}
        if not in_memory:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            url,
            # Agent runs write from a background thread while the live feed
            # reads: wait for locks instead of failing immediately.
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=StaticPool if in_memory else None,
        )

        @event.listens_for(engine, "connect")
        def _configure_sqlite(dbapi_connection, _record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            if not in_memory:
                cursor.execute("PRAGMA journal_mode=WAL")  # readers do not block the writer
            cursor.close()

        return engine

    return create_engine(url, pool_pre_ping=True)


engine = make_engine(get_settings().database_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db(bind: Engine | None = None) -> None:
    """Create all tables. Hackathon scope: no migrations yet."""
    from backend.app import models  # noqa: F401  (registers the tables)

    Base.metadata.create_all(bind=bind or engine)


def get_db() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def check_database() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - health check reports, never raises
        return False
