from conductor.crews.rag_marketing.crew import RagUrlMarketingCrew
from elasticsearch import Elasticsearch
from conductor.crews.models import CrewRun
from langsmith import traceable
from typing import Callable


@traceable
def run_rag_marketing_crew(
    company_url: str,
    elasticsearch: Elasticsearch,
    index_name: str,
    cache: bool = False,
    redis: bool = False,
    task_callback: Callable = None,
) -> CrewRun:
    """Start with a url and generate a marketing report

    Args:
        company_url (str): URL of a company
        elasticsearch (Elasticsearch): Elasticsearch client
        index_name (str): Name of the Elasticsearch index
        cache (bool, optional): Cache the results. Defaults to False.
        redis (bool, optional): Use Redis for caching. Defaults to False.
        task_callback ([Callable], optional): Task function callback. Defaults to None.

    Returns:
        CrewRun: Data about the crew run including the results.
    """
    crew = RagUrlMarketingCrew(
        company_url=company_url,
        elasticsearch=elasticsearch,
        index_name=index_name,
        cache=cache,
        redis=redis,
        task_callback=task_callback,
    )
    crew_run = crew.run()
    return crew_run
