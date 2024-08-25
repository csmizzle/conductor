"""
Extract and enrich a report with more data
"""
from typing import Optional
from conductor.reports.models import ReportV2, Graph, Timeline, RelationshipType
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.chains import relationships_to_image_query, match_queries_to_paragraphs
from conductor.rag.ingest import queries_to_image_results, insert_image_urls_to_db
from conductor.crews.rag_marketing.chains import (
    extract_graph_from_report,
    extract_timeline_from_report,
)
import logging


logger = logging.getLogger(__name__)


class ReportEnricher:
    def __init__(
        self,
        report: ReportV2,
        serp_api_key: str,
        image_relationship_types: list[RelationshipType],
        client: ElasticsearchRetrieverClient,
        graph_sections: list[str],
        timeline_sections: list[str],
        n_images: int = 1,
    ) -> None:
        """Extract and enrich a report with more data

        Args:
            report (ReportV2): Report to enrich
            serp_api_key (str): Serp API key
            image_relationship_types (list[RelationshipType]): Graph relationship types to convert to image search
            client (ElasticsearchRetrieverClient): Elasticsearch client
            graph_sections (list[str]): Report sections to extract graph
            timeline_sections (list[str]): Report sections to extract timeline
            n_images (int, optional): Number of images to get for each image search. Defaults to 1.
        """
        self.report = report
        self.serp_api_key = serp_api_key
        self.client = client
        self.image_relationship_types = image_relationship_types
        self.graph_sections = graph_sections
        self.timeline_sections = timeline_sections
        self.n_images = n_images
        # containers for extracted data
        self.graph: Optional[Graph] = None
        self.search_queries: Optional[list[dict]] = None
        self.timeline: Optional[Timeline] = None
        self.image_results: Optional[list[str]] = None
        self.added_images: Optional[list[str]] = None

    def extract_graph(self) -> None:
        """
        Extract the graph from the report
        """
        logger.info("Extracting graph from report")
        self.graph = extract_graph_from_report(
            report=self.report,
            sections_filter=self.graph_sections,
        )

    def extract_timeline(self) -> None:
        """Extract the timeline from the report

        Returns:
            Timeline: timeline of events
        """
        logger.info("Extracting timeline from report")
        self.timeline = extract_timeline_from_report(
            report=self.report,
            sections_filter=self.timeline_sections,
        )

    def get_graph_image_queries(self) -> None:
        """Get image queries from the graph relationships

        Returns:
            list[str]: list of image queries
        """
        logger.info("Getting image queries from graph")
        self.search_queries = relationships_to_image_query(
            graph=self.graph,
            api_key=self.serp_api_key,
            relationship_types=self.image_relationship_types,
        )

    def queries_to_image_results(self) -> None:
        """
        Convert queries to image search results
        """
        logger.info("Getting image results from queries")
        self.image_results = queries_to_image_results(
            self.search_queries, self.n_images
        )

    def match_queries_to_paragraphs(self) -> None:
        """
        Match images to paragraphs
        """
        logger.info("Matching images to paragraphs")
        self.report = match_queries_to_paragraphs(
            image_search_results=self.image_results,
            sections_filter=self.graph_sections,
            report=self.report,
        )

    def save_images_to_db(self) -> None:
        """
        Iterate through image results appended to paragraphs and save to db
        """
        logger.info("Saving images to db")
        images_to_save = []
        for section in self.report.report.sections:
            for paragraph in section.paragraphs:
                if paragraph.images:
                    images_to_save.append(paragraph.images)
        self.added_images = insert_image_urls_to_db(
            image_results=images_to_save, client=self.client
        )

    def enrich(self) -> ReportV2:
        """
        Enrich the report with more data
        """
        self.extract_graph()
        # create search queries from graph
        self.get_graph_image_queries()
        # get image results from queries
        self.queries_to_image_results()
        # match images to paragraphs
        self.match_queries_to_paragraphs()
        # save matched images to db
        self.save_images_to_db()
        return self.report
