from redis import Redis
from typing import Any, Dict
from crewai.agents.cache.cache_handler import CacheHandler
from pydantic import PrivateAttr
import os


class RedisCrewCacheHandler(CacheHandler):
    """
    Handle multiple crew run caches without losing the cache using Redis.
    """

    _cache: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def add(self, tool: str, input: str, output: str) -> None:
        """
        Add a cache entry.
        """
        _cache: Redis = Redis.from_url(os.getenv("REDIS_CREW_CACHE_URL"))
        _cache.set(f"{tool}-{input}", output, ex=60 * 5)  # set expiration for 5 minutes

    def read(self, tool: str, input: str) -> str:
        """
        Read a cache entry.
        """
        _cache: Redis = Redis.from_url(os.getenv("REDIS_CREW_CACHE_URL"))
        result = _cache.get(f"{tool}-{input}")
        if result:
            return result.decode("utf-8")
