"""FastAPI application factory and startup wiring."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="railway-crowd-analytics",
        version="0.1.0",
        description="Local API for railway crowd analytics processing and metrics.",
    )
    app.include_router(router)
    return app


app = create_app()
