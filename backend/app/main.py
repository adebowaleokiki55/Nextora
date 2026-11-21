"""FastAPI application: one backend for all five modules."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.core.lab.inference import get_model
from backend.app.core.demo_seed import seed_if_empty
from backend.app.db import SessionLocal, init_db
from backend.app.modules import demo, lab, places, prediction, surveillance, system, treatment


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings = get_settings()
    init_db()
    with SessionLocal() as session:
        seed_if_empty(session)
    # Load the CNN at startup so the first upload is not slow.
    get_model(settings.lab_model_path)
    yield


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Nextora AI API",
        version="2.0.0-dev",
        description=(
            "Nextora malaria intelligence platform backend. Currently serves the laboratory "
            "module (malaria microscopy research prototype with image "
            "classification and a tool-using research assistant) and "
            "preliminary treatment decision support."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(system.router)
    app.include_router(places.router)
    app.include_router(surveillance.router)
    app.include_router(prediction.router)
    app.include_router(lab.router)
    app.include_router(demo.router)
    app.include_router(treatment.router)
    return app


app = create_app()
