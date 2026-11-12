"""FastAPI application entry point.

Run with:  uvicorn app.main:app --port 8000
"""
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.api.routes import router
from app.config import get_settings
from app.container import Container, build_container
from app.core.errors import AppError
from app.core.logging_setup import log_event, request_id_var, setup_logging

logger = logging.getLogger("app.http")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Open the durable checkpointer once at startup and build every component."""
    settings = get_settings()
    setup_logging(settings.log_file)
    settings.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_path)) as checkpointer:
        app.state.container = build_container(settings, checkpointer)
        log_event(logger, "application_started")
        yield


def create_app(container: Container | None = None) -> FastAPI:
    """`container` lets tests inject fakes; in production it is built in `lifespan`."""
    app = FastAPI(
        title="Roads and Transport Knowledge Assistant",
        description="Evidence-grounded question answering over Dubai Roads and Transport legislation.",
        version="1.0.0",
        lifespan=None if container else lifespan,
    )
    if container:
        app.state.container = container

    @app.middleware("http")
    async def trace_request(request: Request, call_next):
        """HTTP middleware: give every request an id and log start/end."""
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request_id_var.set(request_id)
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        if not request.url.path.startswith("/ui"):
            log_event(logger, "http_request", method=request.method, path=request.url.path,
                      status_code=response.status_code,
                      duration_ms=round((time.perf_counter() - started) * 1000))
        return response

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, error: AppError):
        log_event(logger, "controlled_error", code=error.code, message=error.message)
        return JSONResponse(status_code=error.status_code, content={
            "error": {"code": error.code, "message": error.message},
            "request_id": request_id_var.get()})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, error: Exception):
        logger.exception("unexpected_error")
        return JSONResponse(status_code=500, content={
            "error": {"code": "internal_error", "message": "an unexpected error occurred"},
            "request_id": request_id_var.get()})

    app.include_router(router)
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def open_test_ui():
        return RedirectResponse("/ui/")

    return app


app = create_app()
