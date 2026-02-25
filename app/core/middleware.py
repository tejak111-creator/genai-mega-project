import logging
import time
import uuid
from typing import Callable
from fastapi import Request,Response

logger = logging.getLogger("app")

async def request_context_middleware(request: Request, call_next: Callable) -> Response:
    request_id = request.headers.get("x-request-id",str(uuid.uuid4()))
    #Flow:

    #If client sends header x-request-id, use it.

    #Otherwise, generate a new UUID.
    start = time.perf_counter()
    try:
        response: Response = await call_next(request)
        #call_next will be endpoint or handler
        status_code = response.status_code
        return response
    except Exception:
        status_code = 500
        return Response(content="Internal Server Error", status_code=500)
    finally: #ALWAYS RUNS EVEN FOR EXCEPTIONS
        latency_ms = int((time.perf_counter() - start)*1000)
        #Attach useful fields into log record via "extra"
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