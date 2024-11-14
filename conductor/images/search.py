"""
Conduct relationship based image search
"""
from conductor.reports.models import ImageSearchResult
from conductor.images.signatures import RelationshipImageSearch
from conductor.rag.ingest import queries_to_image_results_parallel
from conductor.graph import models
import dspy
import concurrent.futures
from loguru import logger


class SearchBuilder:
    """
    Build search queries
    """

    def __init__(self, graph: models.Graph) -> None:
        self.graph = graph
        self.relationship_image_search = dspy.Predict(RelationshipImageSearch)

    def _relationship_image_search(self, relationship: models.Relationship) -> str:
        """
        Light wrapper for logging
        """
        query = self.relationship_image_search(relationship=relationship)
        logger.info(f"Created query: {query.query}")
        return query.query

    def build_search_queries(self) -> dict[str, models.Relationship]:
        """
        Build search queries
        Queries will
        """
        search_queries = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {}
            for idx, relationship in enumerate(self.graph.relationships):
                futures[idx] = executor.submit(
                    self._relationship_image_search, relationship=relationship
                )
            for idx, future in futures.items():
                search_queries[future.result()] = self.graph.relationships[idx]
        return search_queries


class ImageCollector:
    """
    GGet images from search queries
    """

    def __init__(self, queries: list[str], api_key: str) -> None:
        self.queries = queries
        self.api_key = api_key

    def collect(self) -> list[ImageSearchResult]:
        """
        Collect images from search queries
        """
        return queries_to_image_results_parallel(
            search_queries=self.queries,
            api_key=self.api_key,
        )


class GraphImageCollector:
    def __init__(self, graph: models.Graph, api_key: str) -> None:
        self.graph = graph
        self.api_key = api_key
        self.search_builder = SearchBuilder(graph=graph)

    def collect(self) -> list[ImageSearchResult]:
        """
        Collect images from search queries
        """
        search_queries = self.search_builder.build_search_queries()
        return queries_to_image_results_parallel(
            search_queries=search_queries.keys(),
            api_key=self.api_key,
        )


def build_searches_from_graph(
    graph: models.Graph,
) -> dict[str, models.Relationship]:
    """
    Build search queries from a graph
    """
    search_builder = SearchBuilder(graph=graph)
    return search_builder.build_search_queries()


def collect_images_from_queries(
    queries: list[str], api_key: str
) -> list[ImageSearchResult]:
    """
    Collect images from search queries
    """
    image_collector = ImageCollector(queries=queries, api_key=api_key)
    return image_collector.collect()


def collect_images_from_graph(
    graph: models.Graph, api_key: str
) -> list[ImageSearchResult]:
    """
    Collect images from a graph
    """
    image_collector = GraphImageCollector(graph=graph, api_key=api_key)
    return image_collector.collect()
