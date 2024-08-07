from redis import Redis
from crewai.agents.cache.cache_handler import CacheHandler
import os


class RedisCrewCacheHandler(CacheHandler):
    """
    Handle multiple crew run caches without losing the cache using Redis.
    """

    def __init__(self, url: str = None) -> None:
        if url is None:
            url = os.getenv("REDIS_CREW_CACHE_URL")
            if url is None:
                raise ValueError("Redis URL is required, set REDIS_CREW_CACHE_URL.")
        else:
            self.url = url
        self._cache: Redis = Redis.from_url(url)

    def add(self, tool: str, input: str, output: str) -> None:
        """
        Add a cache entry.
        """
        self._cache.set(
            f"{tool}-{input}", output, ex=60 * 5
        )  # set expiration for 5 minutes

    def read(self, tool: str, input: str) -> str:
        """
        Read a cache entry.
        """
        result = self._cache.get(f"{tool}-{input}")
        if result:
            return result.decode("utf-8")
