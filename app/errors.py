from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Error returned to the client as {"code", "detail", "params"}.

    `code` is stable so clients can show their own localized message;
    `detail` is an English message for logs and API consumers.
    """

    status_code: int = status.HTTP_400_BAD_REQUEST

    def __init__(self, code: str, detail: str, *, status_code: int | None = None, **params: Any) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.params = params
        if status_code is not None:
            self.status_code = status_code


async def app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "detail": exc.detail, "params": exc.params},
    )
