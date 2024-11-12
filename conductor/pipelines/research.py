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
from conductor.builder.agent import ResearchTeamTemplate
from conductor.flow import models as flow_models
from conductor.reports.builder import models
from conductor.builder.agent import build_from_report_sections_parallel
from conductor.flow.flow import (
    SearchFlow,
    ResearchFlow,
    RunResult,
    run_flow,
    run_search_flow,
)
from conductor.flow import runner, builders, research, team, retriever
from conductor.crews.rag_marketing import tools
from conductor.reports.builder.outline import (
    build_outline,
    build_refined_outline,
)
from conductor.reports.builder.writer import write_report
from conductor.reports.builder import models as report_models
from conductor.profiles.generate import generate_profile_parallel
from pydantic import BaseModel
from typing import Callable, Optional
from reportlab.platypus import SimpleDocTemplate
from tempfile import NamedTemporaryFile
from docx import Document
import shutil
import dspy
from dspy import LM
from crewai import LLM
from crewai.crews.crew_output import CrewOutput
from elasticsearch import Elasticsearch
from langchain_core.embeddings import Embeddings
from loguru import logger
import sys


logger.add(sys.stdout, colorize=True, enqueue=True)


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

        Steps:
        1. Build the research team.
        2. Run the research flow.
        3. Build the search team.
        4. Run the search flow.
        5. Create the run result.
        6. Build the report outline.
        7. Refine the report outline.
        8. Write the final report.
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
    - Research the entity
    - Build out entity profile
    - Search to answer key research questions
    """

    def __init__(
        self,
        url: str,
        team_title: str,
        perspective: str,
        section_titles: list[str],
        elasticsearch: Elasticsearch,
        elasticsearch_index: str,
        embeddings: Embeddings,
        profile: BaseModel = None,
        cohere_api_key: str = None,
        run_in_parallel: bool = False,
        team_builder_llm: LM = None,
        research_llm: LLM = None,
        search_llm: LM = None,
        outline_llm: LM = None,
        report_llm: LM = None,
        profile_llm: LM = None,
        k: int = 3,
        research_max_iterations: int = 1,
    ) -> None:
        self.url = url
        self.team_title = team_title
        self.perspective = perspective
        self.section_titles = section_titles
        self.elasticsearch = elasticsearch
        self.elasticsearch_index = elasticsearch_index
        self.embeddings = embeddings
        # optional parameters
        self.profile = profile
        self.cohere_api_key = cohere_api_key
        self.team_builder_llm = team_builder_llm
        self.research_llm = research_llm
        self.search_llm = search_llm
        self.outline_llm = outline_llm
        self.report_llm = report_llm
        self.profile_llm = profile_llm
        self.run_in_parallel = run_in_parallel
        self.k = k
        self.research_max_iterations = research_max_iterations
        # pipeline components
        self.team: ResearchTeamTemplate = None
        self.research_team: flow_models.Team = None
        self.research_results: Optional[list[CrewOutput]] = None
        self.research_flow: ResearchFlow = None
        self.specification: str = None
        self.search_team: flow_models.SearchTeam = None
        self.search_flow: SearchFlow = None
        self.search_answers: Optional[list[runner.SearchTeamAnswers]] = None
        self.run_result: RunResult = None
        self.outline: models.ReportOutline = None
        self.refined_outline: models.ReportOutline = None
        self.report: models.Report = None
        self.generated_profile: BaseModel = None

    def build_team_template(self) -> ResearchTeamTemplate:
        """
        Builds the research team.
        Returns:
            ResearchTeamTemplate: The research team.
        """
        logger.info("Building team template ...")
        if self.team_builder_llm:
            dspy.configure(lm=self.team_builder_llm)
        self.team = build_from_report_sections_parallel(
            team_title=self.team_title,
            section_titles=self.section_titles,
            perspective=self.perspective,
        )
        logger.info("Team template built.")
        return self.team

    def build_research_team(self) -> flow_models.Team:
        if self.team_builder_llm:
            dspy.configure(lm=self.team_builder_llm)
        if self.team:
            logger.info("Building research team ...")
            self.research_team = builders.build_team_from_template(
                team_template=self.team,
                llm=self.research_llm,
                tools=[
                    tools.SerpSearchEngineIngestTool(
                        elasticsearch=self.elasticsearch,
                        index_name=self.elasticsearch_index,
                    )
                ],
                agent_factory=research.ResearchAgentFactory,
                task_factory=research.ResearchQuestionAgentSearchTaskFactory,
                team_factory=team.ResearchTeamFactory,
                max_iter=self.research_max_iterations,
            )
            logger.info("Research team built.")
            return self.research_team
        else:
            logger.error(
                "Missing team template, try running build_team_template() first."
            )
            raise ValueError(
                "Missing team template, try running build_team_template() first."
            )

    def run_research(self) -> list[CrewOutput]:
        """
        Runs the research and search.
        Returns:
            RunResult: The result of the research and search.
        """
        if self.research_team:
            logger.info("Running research ...")
            self.research_flow = ResearchFlow(
                research_team=self.research_team,
                url=self.url,
                elasticsearch=self.elasticsearch,
                index_name=self.elasticsearch_index,
                llm=self.research_llm,
                parallel=self.run_in_parallel,
            )
            self.research_results = run_flow(flow=self.research_flow)
            # set specification after flow is completed
            self.specification = self.research_flow.state.organization_determination.raw
            logger.info("Research completed.")
            return self.research_results
        else:
            logger.error(
                "Missing research team, try running build_research_team() first."
            )
            raise ValueError(
                "Missing research team, try running build_research_team() first."
            )

    def build_search_team(self) -> flow_models.SearchTeam:
        """
        Builds the search team.
        Returns:
            models.SearchTeam: The search team.
        """
        if self.team_builder_llm:
            dspy.configure(lm=self.team_builder_llm)
        logger.info("Building search team ...")
        self.search_team = builders.build_search_team_from_template(team=self.team)
        logger.info("Search team built.")
        return self.search_team

    def run_search(self) -> list[runner.SearchTeamAnswers]:
        """
        Runs the search flow.
        Returns:
            list[runner.SearchTeamAnswers]: The search results.
        """
        if self.search_llm:
            dspy.configure(lm=self.search_llm)
        self.search_flow = SearchFlow(
            search_team=self.search_team,
            organization_determination=self.research_flow.state.organization_determination.raw,
            elastic_retriever=retriever.ElasticRMClient(
                elasticsearch=self.elasticsearch,
                index_name=self.elasticsearch_index,
                embeddings=self.embeddings,
                cohere_api_key=self.cohere_api_key,
                k=self.k,
            ),
        )
        self.search_answers = run_search_flow(flow=self.search_flow)
        return self.search_answers

    def build_profile(self) -> BaseModel:
        """
        Generate a profile for the entity.
        """
        if self.specification:
            if self.profile_llm:
                dspy.configure(lm=self.profile_llm)
            logger.info("Generating profile ...")
            self.generated_profile = generate_profile_parallel(
                model=self.profile,
                embeddings=self.embeddings,
                specification=self.specification,
                elasticsearch=self.elasticsearch,
                index_name=self.elasticsearch_index,
                cohere_api_key=self.cohere_api_key,
            )
            logger.info("Profile generated.")
            return self.generated_profile
        else:
            logger.error(
                "Missing specification to generate profile, try running run_research() first."
            )
            raise ValueError("Missing specification to generate profile")

    def create_run_result(self) -> RunResult:
        """
        Creates the run result.
        Returns:
            RunResult: The run result.
        """
        if self.refined_outline and self.search_answers and self.research_flow:
            self.run_result = RunResult(
                research=self.research_results,
                search=self.search_answers,
                specification=self.research_flow.state.organization_determination.raw,
            )
            return self.run_result
        else:
            raise ValueError("Missing required components to create run result")

    def build_outline(self) -> report_models.ReportOutline:
        """
        Builds the report outline.
        Returns:
            models.ReportOutline: The report outline.
        """
        if self.outline_llm:
            dspy.configure(lm=self.outline_llm)
        self.outline = build_outline(
            specification=self.specification, section_titles=self.section_titles
        )
        return self.outline

    def build_refined_outline(self) -> report_models.ReportOutline:
        """
        Builds the refined report outline.
        Returns:
            models.ReportOutline: The refined report outline.
        """
        self.refined_outline = build_refined_outline(
            perspective=self.perspective,
            draft_outline=self.outline,
        )
        return self.refined_outline

    def write_report(self) -> report_models.Report:
        """
        Writes the report.
        Returns:
            models.Report: The generated report.
        """
        if self.report_llm:
            dspy.configure(lm=self.report_llm)
        self.report = write_report(
            outline=self.refined_outline,
            elastic_retriever=retriever.ElasticRMClient(
                elasticsearch=self.elasticsearch,
                index_name=self.elasticsearch_index,
                embeddings=self.embeddings,
                cohere_api_key=self.cohere_api_key,
                k=self.k,
            ),
        )
        return self.report

    def build_teams(self) -> None:
        """
        Builds the research and search teams.
        """
        logger.info("Building teams ...")
        self.build_team_template()
        self.build_research_team()
        self.build_search_team()

    def run(self) -> report_models.Report:
        """
        Runs the pipeline.
        """
        self.build_team_template()
        self.build_research_team()
        self.run_research()
        self.build_search_team()
        self.run_search()
        self.create_run_result()
        self.build_outline()
        self.build_refined_outline()
        self.write_report()
