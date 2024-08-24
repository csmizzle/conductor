"""
Extract and enrich a report with more data
"""
from pydantic import BaseModel
from conductor.reports.models import ReportV2
from conductor.chains.models import Graph, Timeline, RelationshipType, ImageSearchResult
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.chains import relationships_to_image_query
from conductor.rag.ingest import queries_to_image_results, insert_image_urls_to_db
from conductor.crews.rag_marketing.chains import (
    extract_graph_from_report,
    extract_timeline_from_report,
)


class ParagraphPictures(BaseModel):
    """
    Map images to paragraphs
    """

    paragraph_index: int
    images: list[ImageSearchResult]


class EnrichedReport(BaseModel):
    report: ReportV2
    graph: Graph
    timeline: Timeline
    paragraph_pictures: list[ParagraphPictures]


class ReportEnricher:
    def __init__(
        self,
        report: ReportV2,
        serp_api_key: str,
        client: ElasticsearchRetrieverClient,
        image_relationship_types: list[RelationshipType],
        graph_sections: list[str],
        timeline_sections: list[str],
        n_images: int = 1,
    ) -> None:
        self.report = report
        self.serp_api_key = serp_api_key
        self.client = client
        self.image_relationship_types = image_relationship_types
        self.graph_sections = graph_sections
        self.timeline_sections = timeline_sections
        self.n_images = n_images

    def extract_graph(self) -> Graph:
        """
        Extract the graph from the report
        """
        return extract_graph_from_report(
            report=self.report,
            sections_filter=self.graph_sections,
        )

    def extract_timeline(self) -> Timeline:
        """Extract the timeline from the report

        Returns:
            Timeline: timeline of events
        """
        return extract_timeline_from_report(
            report=self.report,
            sections_filter=self.timeline_sections,
        )

    def get_graph_image_queries(self, graph: Graph) -> list[str]:
        """Get image queries from the graph relationships

        Returns:
            list[str]: list of image queries
        """
        return relationships_to_image_query(
            graph=graph,
            api_key=self.serp_api_key,
            relationship_types=self.image_relationship_types,
        )

    def queries_to_image_results(
        self, search_queries: list[str]
    ) -> list[ImageSearchResult]:
        """
        Convert queries to image search results
        """
        return queries_to_image_results(search_queries, self.n_images)

    def insert_image_urls_to_db(
        self, image_results: list[ImageSearchResult]
    ) -> list[str]:
        """
        Insert image urls to the database
        """
        return insert_image_urls_to_db(image_results, self.client)

    # def enrich(self) -> ReportV2:
    #     """
    #     Enrich the report with more data
    #     """
    #     graph = self.extract_graph()
    #     # create search queries from graph
    #     search_queries = self.get_graph_image_queries(graph)
    #     # get image results from queries
    #     image_results = self.queries_to_image_results(search_queries)
    #     # match images to paragraphs
    #     matches = match_queries_to_paragraphs(
    #         image_results, self.graph_sections, self.report
    #     )
    #     # map images to paragraphs
    #     return self.report
