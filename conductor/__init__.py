import os
import langchain
import langchain.cache
from langchain_community.cache import RedisCache
from redis import Redis


# if llm cache is set, use it and set up the redis cache
if os.getenv("LLM_CACHE"):
    langchain.cache = RedisCache(
        redis_=Redis.from_url(os.getenv("REDIS_URL")),
        ttl=os.getenv("LLM_CACHE_TTL", None),
    )
