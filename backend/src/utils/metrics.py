import time
import functools
from typing import Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta
from threading import Lock
from ..utils.logging import logger


@dataclass
class RequestMetrics:
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    error_count: int = 0


class MetricsCollector:
    """
    Collects and tracks performance metrics for API endpoints and services
    """
    def __init__(self):
        self._metrics: Dict[str, RequestMetrics] = defaultdict(RequestMetrics)
        self._lock = Lock()
        self._start_times: Dict[str, float] = {}
        self._request_id_counter = 0

    def start_request_timer(self, endpoint: str) -> str:
        """
        Start timing a request and return a request ID
        """
        with self._lock:
            self._request_id_counter += 1
            req_id = f"req_{self._request_id_counter}_{int(time.time())}"
            self._start_times[req_id] = time.time()
            return req_id

    def record_request(self, req_id: str, endpoint: str, status_code: int = 200) -> None:
        """
        Record a completed request with its metrics
        """
        with self._lock:
            start_time = self._start_times.pop(req_id, None)
            if start_time is None:
                return  # Request not found or already recorded

            duration = time.time() - start_time

            # Update metrics for this endpoint
            metrics = self._metrics[endpoint]
            metrics.count += 1
            metrics.total_time += duration
            metrics.min_time = min(metrics.min_time, duration)
            metrics.max_time = max(metrics.max_time, duration)
            if status_code >= 400:
                metrics.error_count += 1

    def get_metrics(self, endpoint: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics, optionally filtered by endpoint
        """
        with self._lock:
            if endpoint:
                if endpoint in self._metrics:
                    m = self._metrics[endpoint]
                    avg_time = m.total_time / m.count if m.count > 0 else 0
                    return {
                        endpoint: {
                            "count": m.count,
                            "avg_time": avg_time,
                            "min_time": m.min_time if m.min_time != float('inf') else 0,
                            "max_time": m.max_time,
                            "error_count": m.error_count,
                            "error_rate": m.error_count / m.count if m.count > 0 else 0
                        }
                    }
                else:
                    return {}
            else:
                result = {}
                for ep, m in self._metrics.items():
                    avg_time = m.total_time / m.count if m.count > 0 else 0
                    result[ep] = {
                        "count": m.count,
                        "avg_time": avg_time,
                        "min_time": m.min_time if m.min_time != float('inf') else 0,
                        "max_time": m.max_time,
                        "error_count": m.error_count,
                        "error_rate": m.error_count / m.count if m.count > 0 else 0
                    }
                return result

    def reset_metrics(self, endpoint: Optional[str] = None) -> None:
        """
        Reset metrics, optionally for a specific endpoint
        """
        with self._lock:
            if endpoint:
                if endpoint in self._metrics:
                    del self._metrics[endpoint]
            else:
                self._metrics.clear()
                self._start_times.clear()


# Global metrics collector instance
metrics_collector = MetricsCollector()


def track_performance(endpoint_name: str):
    """
    Decorator to track performance of functions
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            req_id = metrics_collector.start_request_timer(endpoint_name)
            status_code = 200

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status_code = 500
                raise
            finally:
                metrics_collector.record_request(req_id, endpoint_name, status_code)

                # Log slow requests (longer than 1 second)
                # In a real implementation, we'd get the duration here
                # For now, we'll just log that the request completed
                metrics = metrics_collector.get_metrics(endpoint_name)
                if endpoint_name in metrics:
                    avg_time = metrics[endpoint_name]["avg_time"]
                    if avg_time > 1.0:  # Log if average time is over 1 second
                        logger.warning(f"Slow performance detected for {endpoint_name}: avg {avg_time:.2f}s")

        return wrapper
    return decorator


def get_performance_report() -> Dict[str, Dict[str, float]]:
    """
    Get a comprehensive performance report
    """
    return metrics_collector.get_metrics()