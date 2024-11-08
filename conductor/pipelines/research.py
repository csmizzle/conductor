from conductor.crews.rag_marketing import RagUrlMarketingCrew
from conductor.crews.rag_marketing.chains import crew_run_to_report
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.crews.models import CrewRun
from conductor.reports.enrich import ReportEnricher
from conductor.reports.outputs import report_v2_to_pdf, report_v2_to_docx
from conductor.reports.models import (
    ReportV2,
    ReportStyleV2,
    ReportTone,
    ReportPointOfView,
    RelationshipType,
)
from conductor.utils.graph import graph_to_networkx, draw_networkx
from typing import Callable, Optional
from reportlab.platypus import SimpleDocTemplate
from tempfile import NamedTemporaryFile
from docx import Document
import logging
import shutil


logger = logging.getLogger(__name__)


class ResearchPipeline:
    """
    Pipeline for conducting research and generating reports.
    Args:
        url (str): The URL to conduct research on.
        report_style (ReportStyleV2): The style of the report.
        report_tone (ReportTone): The tone of the report.
        report_point_of_view (ReportPointOfView): The point of view of the report.
        client (ElasticsearchRetrieverClient): The Elasticsearch client.
        document_index_name (str): The name of the document index.
        image_index_name (str): The name of the image index.
        report_title (str): The title of the report.
        report_description (str): The description of the report.
        image_search_relationships (list[RelationshipType]): The relationships to search for images.
        serpapi_key (str): The API key for the SERP API.
        report_watermark (bool, optional): Whether to include a watermark in the report. Defaults to True.
        cache (bool, optional): Whether to use caching. Defaults to False.
        graph (bool, optional): Whether to generate a graph. Defaults to True.
        pdf (bool, optional): Whether to generate a PDF. Defaults to True.
        docx (bool, optional): Whether to generate a DOCX. Defaults to False.
        task_callback (Callable, optional): A callback function for tasks. Defaults to None.
        graph_sections (list[str], optional): The sections to include in the graph. Defaults to None.
        timeline_sections (list[str], optional): The sections to include in the timeline. Defaults to None.
        sections_titles_endswith_filter (str, optional): The filter for section titles. Defaults to None.
    Attributes:
        url (str): The URL to conduct research on.
        report_style (ReportStyleV2): The style of the report.
        report_tone (ReportTone): The tone of the report.
        report_point_of_view (ReportPointOfView): The point of view of the report.
        client (ElasticsearchRetrieverClient): The Elasticsearch client.
        document_index_name (str): The name of the document index.
        image_index_name (str): The name of the image index.
        report_title (str): The title of the report.
        report_description (str): The description of the report.
        image_search_relationships (list[RelationshipType]): The relationships to search for images.
        report_watermark (bool): Whether to include a watermark in the report.
        cache (bool): Whether to use caching.
        graph (bool): Whether to generate a graph.
        pdf (bool): Whether to generate a PDF.
        docx (bool): Whether to generate a DOCX.
        _serp_api_key (str): The API key for the SERP API.
        graph_sections (list[str]): The sections to include in the graph.
        timeline_sections (list[str]): The sections to include in the timeline.
        sections_titles_endswith_filter (str): The filter for section titles.
        crew (RagUrlMarketingCrew): The marketing crew for conducting research.
        crew_run (CrewRun): The result of running the marketing crew.
        report (ReportV2): The generated report.
        enricher (ReportEnricher): The enricher for the report.
        pdf_document (SimpleDocTemplate): The PDF document.
        graph_file (NamedTemporaryFile): The temporary file for the graph.
        pdf_filename (NamedTemporaryFile): The temporary file for the PDF.
        docx_filename (NamedTemporaryFile): The temporary file for the DOCX.
    Methods:
        run_crew() -> CrewRun:
            Runs the marketing crew and returns the result.
        crew_run_to_report() -> ReportV2:
            Converts the crew run to a report and returns it.
        enrich_report() -> ReportV2:
            Enriches the report with additional information and returns it.
        report_to_pdf() -> SimpleDocTemplate:
            Converts the report to a PDF document and returns it.
        report_to_docx() -> Document:
            Converts the report to a DOCX document and returns it.
        save_docx(output_dir: str) -> bool:
            Saves the generated DOCX document to the specified output directory.
        save_pdf(output_dir: str) -> bool:
            Saves the generated PDF document to the specified output directory.
        save_graph(output_dir: str) -> bool:
            Saves the generated graph file to the specified output directory.
    """

    def __init__(
        self,
        url: str,
        report_style: ReportStyleV2,
        report_tone: ReportTone,
        report_point_of_view: ReportPointOfView,
        client: ElasticsearchRetrieverClient,
        report_title: str,
        report_description: str,
        image_search_relationships: list[RelationshipType],
        serpapi_key: str,
        report_watermark: bool = False,
        cache: bool = False,
        enrich: bool = False,
        pdf: bool = False,
        docx: bool = False,
        task_callback: Callable = None,
        graph_sections: list[str] = None,
        timeline_sections: list[str] = None,
        sections_titles_endswith_filter: str = None,
    ) -> None:
        self.url = url
        self.report_style = report_style
        self.report_tone = report_tone
        self.report_point_of_view = report_point_of_view
        self.client = client
        self.report_title = report_title
        self.report_description = report_description
        self.image_search_relationships = image_search_relationships
        self.report_watermark = report_watermark
        self.cache = cache
        self.enrich = enrich
        self.pdf = pdf
        self.docx = docx
        self._serp_api_key = serpapi_key
        self.graph_sections = graph_sections
        self.timeline_sections = timeline_sections
        self.sections_titles_endswith_filter = sections_titles_endswith_filter
        self.crew = RagUrlMarketingCrew(
            url=url,
            elasticsearch=client.elasticsearch,
            index_name=client.index_name,
            cache=True if cache else False,
            redis=True if cache else False,
            task_callback=task_callback,
        )
        # pipeline components
        self.crew_run: Optional[CrewRun] = None
        self.report: Optional[ReportV2] = None
        self.enricher: Optional[ReportEnricher] = None
        self.pdf_document: Optional[SimpleDocTemplate] = None
        # temp files for pdf and graph
        if self.enrich:
            self.graph_file = NamedTemporaryFile(
                delete=False, delete_on_close=True, suffix=".png"
            )
        else:
            self.graph_file = None
        if self.pdf:
            self.pdf_filename = NamedTemporaryFile(
                delete=False, delete_on_close=True, suffix=".pdf"
            )
        else:
            self.pdf_filename = None
        if self.docx:
            self.docx_filename = NamedTemporaryFile(
                delete=False, delete_on_close=True, suffix=".docx"
            )
        else:
            self.docx_filename = None

    def __del__(self):
        if self.enrich:
            self.graph_file.close()
        if self.pdf:
            self.pdf_filename.close()
        if self.docx:
            self.docx_filename.close()

    def run_crew(self) -> CrewRun:
        """
        Runs the crew and returns the crew run object.
        Returns:
            CrewRun: The crew run object.
        """

        self.crew_run = self.crew.run()
        return self.crew_run

    def crew_run_to_report(self) -> ReportV2:
        """
        Generates a report based on the crew run.
        Returns:
            ReportV2: The generated report.
        """

        if not self.sections_titles_endswith_filter:
            logger.warning("No section titles filter provided, using default")
            self.sections_titles_endswith_filter = "Research"
        self.report = crew_run_to_report(
            crew_run=self.crew_run,
            title=self.report_title,
            description=self.report_description,
            section_titles_endswith_filter=self.sections_titles_endswith_filter,
            tone=self.report_tone,
            style=self.report_style,
            point_of_view=self.report_point_of_view,
        )
        return self.report

    def enrich_report(self) -> ReportV2:
        """
        Enriches the report by checking if graph and timeline sections are provided.
        If not provided, it uses default sections.
        Then, it initializes a ReportEnricher object with the necessary parameters.
        Finally, it calls the enrich() method of the ReportEnricher object to enrich the report.
        If a graph file is provided, it saves the graph image.
        Returns:
            The enriched report (ReportV2 object).
        """

        # check if graph and timeline sections are provided
        if not self.graph_sections:
            logger.warning("No graph sections provided, using default")
            self.graph_sections = ["Company Structure", "Personnel"]
        if not self.timeline_sections:
            logger.warning("No timeline sections provided, using default")
            self.timeline_sections = ["Company History", "Recent Events"]
        self.enricher = ReportEnricher(
            report=self.report,
            serp_api_key=self._serp_api_key,
            image_relationship_types=self.image_search_relationships,
            client=self.client,
            graph_sections=self.graph_sections,
            timeline_sections=self.timeline_sections,
        )
        self.report = self.enricher.enrich()
        return self.report

    def draw_graph(self) -> None:
        """
        Draws the graph and saves it to a temporary file.
        """

        if not self.enrich:
            raise ValueError("Enrichment not enabled")
        graph = graph_to_networkx(self.enricher.graph)
        draw_networkx(graph, self.graph_file.name)

    def report_to_pdf(self) -> SimpleDocTemplate:
        """
        Converts the report to a PDF document.
        Returns:
            SimpleDocTemplate: The PDF document.
        """

        self.pdf_document = report_v2_to_pdf(
            report=self.report,
            filename=self.pdf_filename.name,
            graph_file=self.graph_file.name,
            watermark=self.report_watermark,
        )
        return self.pdf_document

    def report_to_docx(self) -> Document:
        """
        Converts the report to a Microsoft Word document (docx) format.
        Returns:
            Document: The generated docx document.
        """

        self.docx_document = report_v2_to_docx(
            report=self.report,
            filename=self.docx_filename,
            watermark=self.report_watermark,
        )
        return self.docx_document

    def save_docx(self, output_dir: str) -> bool:
        """
        Saves the generated DOCX document to the specified output directory.
        Args:
            output_dir (str): The directory where the DOCX document will be saved.
        Returns:
            bool: True if the document is successfully saved, False otherwise.
        Raises:
            ValueError: If the DOCX document has not been generated.
        """

        if not self.docx_filename:
            raise ValueError("DOCX document not generated")
        shutil.copy(self.docx_filename.name, output_dir)
        return True

    def save_pdf(self, output_dir: str) -> bool:
        """
        Saves the generated PDF document to the specified output directory.
        Args:
            output_dir (str): The directory where the PDF document will be saved.
        Returns:
            bool: True if the PDF document is successfully saved, False otherwise.
        Raises:
            ValueError: If the PDF document has not been generated.
        """

        if not self.pdf_filename:
            raise ValueError("PDF document not generated")
        shutil.copy(self.pdf_filename.name, output_dir)
        return True

    def save_graph(self, output_dir: str) -> bool:
        """
        Save the generated graph file to the specified output directory.
        Args:
            output_dir (str): The directory where the graph file will be saved.
        Returns:
            bool: True if the graph file is successfully saved, False otherwise.
        Raises:
            ValueError: If the graph file is not generated.
        """

        if not self.graph_file:
            raise ValueError("Graph file not generated")
        shutil.copy(self.graph_file.name, output_dir)
        return True

    def run(self) -> None:
        """
        Runs the pipeline.
        """

        self.run_crew()
        self.crew_run_to_report()
        if self.enrich:
            self.enrich_report()
            self.draw_graph()
        if self.pdf:
            self.report_to_pdf()
        if self.docx:
            self.report_to_docx()


class ResearchPipelineV2:
    """
    Updated version of the ResearchPipeline class.
    """

    pass
