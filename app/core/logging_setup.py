"""Structured (JSON-lines) logging.

Use `log_event(logger, "event_name", key=value, ...)` everywhere so that every
log line is one JSON object that can be filtered by request_id or thread_id.
Never pass secrets or full document text to `log_event`.
"""
import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

# Set once per HTTP request / chat run. Context variables follow the request
# through async tasks and worker threads, so every log line below carries them
# without passing ids through every function.
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
thread_id_var: ContextVar[str | None] = ContextVar("thread_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "request_id": request_id_var.get(),
            "thread_id": thread_id_var.get(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["error"] = repr(record.exc_info[1])
        return json.dumps(entry, ensure_ascii=False, default=str)


def setup_logging(log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)
    # Third-party libraries are noisy at INFO level.
    for noisy in ("httpx", "httpcore", "openai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, **fields) -> None:
    logger.info(event, extra={"fields": fields})
