"""`@logged_node` logs the start and end of a graph node with its duration and
the size of every list it wrote (candidate counts), never the content itself."""
import functools
import inspect
import logging
import time

from app.core.logging_setup import log_event

logger = logging.getLogger("app.graph")


def _summarise(output) -> dict:
    if not isinstance(output, dict):
        return {}
    return {f"{key}_count": len(value) for key, value in output.items() if isinstance(value, list)}


def logged_node(node_name: str):
    def decorator(node_function):
        if inspect.iscoroutinefunction(node_function):
            @functools.wraps(node_function)
            async def async_wrapper(state, *args, **kwargs):
                log_event(logger, "node_start", node=node_name)
                started = time.perf_counter()
                output = await node_function(state, *args, **kwargs)
                log_event(logger, "node_end", node=node_name,
                          duration_ms=round((time.perf_counter() - started) * 1000),
                          **_summarise(output))
                return output
            return async_wrapper

        @functools.wraps(node_function)
        def sync_wrapper(state, *args, **kwargs):
            log_event(logger, "node_start", node=node_name)
            started = time.perf_counter()
            output = node_function(state, *args, **kwargs)
            log_event(logger, "node_end", node=node_name,
                      duration_ms=round((time.perf_counter() - started) * 1000),
                      **_summarise(output))
            return output
        return sync_wrapper
    return decorator
