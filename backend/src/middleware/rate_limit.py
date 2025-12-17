from fastapi import HTTPException, status
from fastapi.requests import Request
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict
import time


class InMemoryRateLimiter:
    def __init__(self, requests: int, window: int):
        """
        :param requests: Number of requests allowed
        :param window: Time window in seconds
        """
        self.requests = requests
        self.window = window
        self.requests_log: Dict[str, list] = defaultdict(list)

    def is_allowed(self, identifier: str) -> bool:
        """
        Check if request is allowed for the given identifier
        """
        now = time.time()
        # Remove old requests outside the time window
        self.requests_log[identifier] = [
            req_time for req_time in self.requests_log[identifier]
            if now - req_time < self.window
        ]

        # Check if within limit
        if len(self.requests_log[identifier]) < self.requests:
            self.requests_log[identifier].append(now)
            return True

        return False


# Create rate limiter instance (100 requests per hour per IP for unauthenticated users)
rate_limiter = InMemoryRateLimiter(requests=100, window=3600)  # 100 requests per hour


async def rate_limit_middleware(request: Request, call_next):
    # Get client IP address
    client_ip = request.client.host

    # Skip rate limiting for health checks and certain endpoints
    if request.url.path.endswith("/health"):
        response = await call_next(request)
        return response

    # Check if request is allowed
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": {
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded",
                    "details": f"You have exceeded the rate limit of 100 requests per hour"
                }
            }
        )

    response = await call_next(request)
    return response