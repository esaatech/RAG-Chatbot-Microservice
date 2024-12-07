# app/services/cache.py
from typing import Dict, Optional, Any
from collections import OrderedDict
from threading import Lock
from datetime import datetime, timedelta
import json

class ConfigCache:
    """
    Thread-safe LRU cache for document configurations with time-based expiration.
    
    Attributes:
        capacity (int): Maximum number of items to store in cache
        expiration_time (int): Time in seconds before a cache entry expires
        _cache (OrderedDict): Ordered dictionary storing cache entries
        _lock (Lock): Thread lock for synchronization
    """

    def __init__(self, capacity: int = 100, expiration_time: int = 3600):
        """
        Initialize the cache.

        Args:
            capacity (int): Maximum number of items to store (default: 100)
            expiration_time (int): Time in seconds before entry expires (default: 1 hour)
        """
        self.capacity = capacity
        self.expiration_time = expiration_time
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._timestamps: Dict[str, datetime] = {}

    def get(self, key: str) -> Optional[Dict]:
        """
        Get item from cache if it exists and hasn't expired.

        Args:
            key (str): Cache key to retrieve

        Returns:
            Optional[Dict]: Cached config or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                return None

            # Check expiration
            if self._is_expired(key):
                self._remove(key)
                return None

            # Move to end (most recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            return value

    def put(self, key: str, value: Dict) -> None:
        """
        Add or update item in cache.

        Args:
            key (str): Cache key
            value (Dict): Configuration to cache
        """
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
            elif len(self._cache) >= self.capacity:
                # Remove least recently used item
                self._cache.popitem(last=False)
                # Also remove its timestamp
                oldest_key = next(iter(self._timestamps))
                self._timestamps.pop(oldest_key)

            self._cache[key] = value
            self._timestamps[key] = datetime.utcnow()

    def remove(self, key: str) -> None:
        """
        Remove item from cache.

        Args:
            key (str): Cache key to remove
        """
        with self._lock:
            self._remove(key)

    def _remove(self, key: str) -> None:
        """Internal method to remove item from cache and timestamps."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry has expired."""
        timestamp = self._timestamps.get(key)
        if not timestamp:
            return True
        return (datetime.utcnow() - timestamp).total_seconds() > self.expiration_time

    def clear(self) -> None:
        """Clear all items from cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    @property
    def size(self) -> int:
        """Get current number of items in cache."""
        return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "size": len(self._cache),
                "capacity": self.capacity,
                "expiration_time": self.expiration_time,
                "oldest_entry": min(self._timestamps.values()) if self._timestamps else None,
                "newest_entry": max(self._timestamps.values()) if self._timestamps else None
            }