import time
from typing import Any, Optional, Dict
from threading import Lock


class SimpleCache:
    """
    A simple in-memory cache with TTL (Time To Live) functionality
    """
    def __init__(self):
        self._cache: Dict[str, tuple] = {}  # key -> (value, expiry_time)
        self._lock = Lock()

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set a value in the cache with a TTL in seconds
        """
        expiry_time = time.time() + ttl
        with self._lock:
            self._cache[key] = (value, expiry_time)

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache, return None if expired or not found
        """
        with self._lock:
            if key not in self._cache:
                return None

            value, expiry_time = self._cache[key]
            if time.time() > expiry_time:
                # Remove expired entry
                del self._cache[key]
                return None

            return value

    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """
        Clear all entries from the cache
        """
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries and return count of removed entries
        """
        current_time = time.time()
        removed_count = 0

        with self._lock:
            expired_keys = [
                key for key, (_, expiry_time) in self._cache.items()
                if current_time > expiry_time
            ]

            for key in expired_keys:
                del self._cache[key]
                removed_count += 1

        return removed_count


# Global cache instance for embeddings
embeddings_cache = SimpleCache()