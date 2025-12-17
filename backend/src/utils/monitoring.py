import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass
from enum import Enum
from ..config.settings import settings
from ..utils.logging import logger


class ServiceType(Enum):
    QDRANT = "qdrant"
    COHERE = "cohere"
    GEMINI = "gemini"
    POSTGRES = "postgres"


@dataclass
class UsageMetrics:
    count: int = 0
    last_reset: float = 0.0
    limit: int = 0
    window_seconds: int = 3600  # 1 hour default


class UsageMonitor:
    """
    Monitor usage of external services to stay within free tier limits
    """
    def __init__(self):
        self.metrics: Dict[ServiceType, UsageMetrics] = {
            ServiceType.QDRANT: UsageMetrics(
                limit=10000,  # Example: 10k requests per month for Qdrant free tier
                window_seconds=30 * 24 * 3600  # 30 days
            ),
            ServiceType.COHERE: UsageMetrics(
                limit=1000,  # Example: 1000 requests per month for Cohere free tier
                window_seconds=30 * 24 * 3600  # 30 days
            ),
            ServiceType.GEMINI: UsageMetrics(
                limit=1500,  # Example: 1500 requests per day for Gemini free tier
                window_seconds=24 * 3600  # 1 day
            ),
            ServiceType.POSTGRES: UsageMetrics(
                limit=10000000,  # Example: 10M rows for Neon free tier
                window_seconds=30 * 24 * 3600  # 30 days
            )
        }

    def increment_usage(self, service: ServiceType) -> bool:
        """
        Increment usage count for a service and check if limit is exceeded.
        Returns True if within limits, False if limit exceeded.
        """
        current_time = time.time()
        metric = self.metrics[service]

        # Check if we need to reset the counter (based on window)
        if current_time - metric.last_reset >= metric.window_seconds:
            metric.count = 0
            metric.last_reset = current_time

        # Increment the count
        metric.count += 1

        # Check if we've exceeded the limit
        if metric.count > metric.limit:
            logger.warning(f"Usage limit exceeded for {service.value}: {metric.count}/{metric.limit}")
            return False

        # Log usage when approaching limits (at 80% and 90%)
        usage_percentage = (metric.count / metric.limit) * 100
        if usage_percentage >= 90:
            logger.warning(f"High usage warning for {service.value}: {metric.count}/{metric.limit} ({usage_percentage:.1f}%)")
        elif usage_percentage >= 80:
            logger.info(f"Usage alert for {service.value}: {metric.count}/{metric.limit} ({usage_percentage:.1f}%)")

        return True

    def get_usage(self, service: ServiceType) -> Dict[str, int]:
        """
        Get current usage information for a service
        """
        metric = self.metrics[service]
        current_time = time.time()

        # Calculate remaining time in window
        time_remaining = max(0, metric.window_seconds - (current_time - metric.last_reset))

        return {
            "current_usage": metric.count,
            "limit": metric.limit,
            "percentage_used": min(100, int((metric.count / metric.limit) * 100)),
            "time_remaining_seconds": time_remaining,
            "reset_timestamp": metric.last_reset + metric.window_seconds
        }

    def get_all_usage(self) -> Dict[ServiceType, Dict[str, int]]:
        """
        Get usage information for all services
        """
        return {service: self.get_usage(service) for service in ServiceType}


# Global monitor instance
usage_monitor = UsageMonitor()


def check_service_usage(service: ServiceType) -> bool:
    """
    Check if a service is within its usage limits
    """
    return usage_monitor.increment_usage(service)


def get_usage_report() -> Dict[str, Dict[str, int]]:
    """
    Get a comprehensive usage report for all services
    """
    all_usage = usage_monitor.get_all_usage()
    return {service.value: usage for service, usage in all_usage.items()}


def track_performance(endpoint_path: str):
    """
    Decorator to track performance of API endpoints
    """
    def decorator(func):
        import functools
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
                logger.info(f"Endpoint {endpoint_path} executed in {execution_time:.2f}ms")
                return result
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                logger.error(f"Endpoint {endpoint_path} failed after {execution_time:.2f}ms: {str(e)}")
                raise
        return wrapper
    return decorator