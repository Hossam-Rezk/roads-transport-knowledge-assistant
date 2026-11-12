"""LangChain agent middleware.

This is NOT FastAPI/HTTP middleware. It runs inside the answer agent's loop and
wraps each individual model call and tool call the agent makes:

  agent loop:  [model call] -> tool call(s) -> [model call] -> ... -> final answer
                  ^ awrap_model_call            ^ awrap_tool_call

`ExecutionLoggingMiddleware` logs the timing of every call and turns a failing
tool call into an error message the model can read, instead of crashing the run.
Call limits and retries are added with LangChain's built-in middleware in
`answer_agent.py`.
"""
import logging
import time

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

from app.core.logging_setup import log_event

logger = logging.getLogger("app.agent")


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


class ExecutionLoggingMiddleware(AgentMiddleware):

    async def awrap_model_call(self, request, handler):
        started = time.perf_counter()
        try:
            response = await handler(request)
        except Exception as error:
            log_event(logger, "middleware_model_call_failed",
                      duration_ms=_elapsed_ms(started), error=type(error).__name__)
            raise
        requested_tools = [tool_call["name"]
                           for message in getattr(response, "result", [])
                           for tool_call in getattr(message, "tool_calls", [])]
        log_event(logger, "middleware_model_call", duration_ms=_elapsed_ms(started),
                  requested_tools=requested_tools)
        return response

    async def awrap_tool_call(self, request, handler):
        tool_call = request.tool_call
        started = time.perf_counter()
        try:
            result = await handler(request)
        except Exception as error:
            # Controlled failure: report the problem to the model as a tool result.
            log_event(logger, "middleware_tool_call_failed", tool=tool_call["name"],
                      arguments=tool_call["args"], duration_ms=_elapsed_ms(started),
                      error=str(error)[:300])
            return ToolMessage(content=f"Tool error: {error}", tool_call_id=tool_call["id"],
                               name=tool_call["name"], status="error")
        log_event(logger, "middleware_tool_call", tool=tool_call["name"],
                  arguments=tool_call["args"], duration_ms=_elapsed_ms(started),
                  status=getattr(result, "status", None))
        return result
