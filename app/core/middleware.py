import logging
import time
import uuid
import traceback
from typing import Callable
from fastapi import Request, Response

logger = logging.getLogger("app")

async def request_context_middleware(request: Request, call_next: Callable) -> Response:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    start = time.perf_counter()
    status_code = 500  # default if something crashes before response exists

    try:
        response: Response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as exc:
        status_code = 500
        traceback.print_exc()
        raise
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "path": str(request.url.path),
                "method": request.method,
                "status_code": status_code,
                "latency_ms": latency_ms,
            },
        )