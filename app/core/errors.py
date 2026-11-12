"""Controlled application errors. The API layer turns each into a JSON response."""


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class IndexUnavailableError(AppError):
    status_code = 503
    code = "index_unavailable"


class ThreadBusyError(AppError):
    status_code = 409
    code = "thread_busy"


class PendingClarificationError(AppError):
    status_code = 409
    code = "clarification_pending"


class InvalidResumeError(AppError):
    status_code = 409
    code = "invalid_resume"


class ModelFailureError(AppError):
    status_code = 502
    code = "model_failure"


class RequestTimeoutError(AppError):
    status_code = 504
    code = "request_timeout"
