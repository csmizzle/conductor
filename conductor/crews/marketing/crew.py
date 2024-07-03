from conductor.crews.marketing.agents import MarketingAgents
from conductor.reports.models import ReportStyle
from conductor.crews.marketing.tasks import MarketingTasks
from conductor.crews.models import CrewRun
from conductor.crews.marketing.utils import task_to_task_run
from conductor.crews.cache import RedisCrewCacheHandler
from conductor.llms import claude_sonnet
from crewai import Crew
from crewai.telemetry import Telemetry
from crewai.utilities import FileHandler, Logger, RPMController
import logging
from pydantic import PrivateAttr, model_validator


logger = logging.getLogger(__name__)


class RedisCacheHandlerCrew(Crew):
    _cache_handler: RedisCrewCacheHandler = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def set_private_attrs(self) -> "RedisCacheHandlerCrew":
        """Set private attributes."""
        self._cache_handler = RedisCrewCacheHandler()
        self._logger = Logger(self.verbose)
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        self._telemetry = Telemetry()
        self._telemetry.set_tracer()
        self._telemetry.crew_creation(self)
        return self


class UrlMarketingCrew:
    """
    Start with a company URL and generate a marketing report
    """

    def __init__(
        self,
        url: str,
        report_style: ReportStyle,
        verbose: bool = True,
        output_log_file: bool | str = None,
        step_callback=None,
        task_callback=None,
        cache=None,
        proxy=None,
    ) -> None:
        self.url = url
        self.report_style = report_style
        self.output_log_file = output_log_file
        self.step_callback = step_callback
        self.task_callback = task_callback
        self.cache = cache
        if self.cache:
            self.cache_handler = RedisCrewCacheHandler()
        self.verbose = verbose
        self.proxy = proxy

    def run(self) -> CrewRun:
        # create agents and tasks
        agents = MarketingAgents()
        tasks = MarketingTasks()
        # agents
        company_identification_agent = agents.url_research_agent(
            llm=claude_sonnet,
            cache=self.cache,
            proxy=self.proxy,
            cache_handler=self.cache_handler,
        )
        company_research_agent = agents.company_research_agent(
            llm=claude_sonnet,
            cache=self.cache,
            proxy=self.proxy,
            cache_handler=self.cache_handler,
        )
        search_engine_agent = agents.search_engine_agent(
            llm=claude_sonnet,
            cache=self.cache,
            cache_handler=self.cache_handler,
        )
        swot_agent = agents.swot_agent(
            llm=claude_sonnet,
            cache=self.cache,
            proxy=self.proxy,
            cache_handler=self.cache_handler,
        )
        competitor_agent = agents.competitor_agent(
            llm=claude_sonnet,
            cache=self.cache,
            proxy=self.proxy,
            cache_handler=self.cache_handler,
        )
        writer_agent = agents.writer_agent(
            llm=claude_sonnet,
            cache=self.cache,
            cache_handler=self.cache_handler,
        )
        editor_agent = agents.editor_agent(
            llm=claude_sonnet,
        )

        # tasks
        company_identification_task = tasks.company_identification_task(
            agent=company_identification_agent,
            company_url=self.url,
        )
        company_research_task = tasks.company_research_task(
            agent=company_research_agent,
            company_url=self.url,
            context=[company_identification_task],
        )
        search_engine_task = tasks.search_engine_task(
            agent=search_engine_agent,
            context=[company_identification_task, company_research_task],
        )
        swot_task = tasks.company_swot_task(
            agent=swot_agent,
            context=[
                company_identification_task,
                company_research_task,
                search_engine_task,
            ],
        )
        competitor_task = tasks.company_competitor_task(
            agent=competitor_agent, context=[company_research_task, search_engine_task]
        )
        writer_task = tasks.company_report_task(
            agent=writer_agent,
            context=[
                company_identification_task,
                company_research_task,
                swot_task,
                competitor_task,
                search_engine_task,
            ],
            report_style=self.report_style,
        )
        review_task = tasks.review_task(
            agent=editor_agent,
            context=[
                company_identification_task,
                writer_task,
                company_research_task,
                swot_task,
                competitor_task,
                search_engine_task,
            ],
            report_style=self.report_style,
        )
        # create crew
        if self.cache:
            crew = RedisCacheHandlerCrew(
                agents=[
                    company_identification_agent,
                    company_research_agent,
                    search_engine_agent,
                    swot_agent,
                    competitor_agent,
                    writer_agent,
                    editor_agent,
                ],
                tasks=[
                    company_identification_task,
                    company_research_task,
                    search_engine_task,
                    swot_task,
                    competitor_task,
                    writer_task,
                    review_task,
                ],
                verbose=self.verbose,
                step_callback=self.step_callback,
                output_log_file=self.output_log_file,
                task_callback=self.task_callback,
                cache=self.cache,
                _cache_handler=self.cache_handler,
            )
        else:
            crew = Crew(
                agents=[
                    company_identification_agent,
                    company_research_agent,
                    search_engine_agent,
                    swot_agent,
                    competitor_agent,
                    writer_agent,
                    editor_agent,
                ],
                tasks=[
                    company_identification_task,
                    company_research_task,
                    search_engine_task,
                    swot_task,
                    competitor_task,
                    writer_task,
                    review_task,
                ],
                verbose=self.verbose,
                step_callback=self.step_callback,
                output_log_file=self.output_log_file,
                task_callback=self.task_callback,
            )

        result = crew.kickoff()

        # create and return crew run
        crew_run = CrewRun(
            tasks=[task_to_task_run(task) for task in crew.tasks],
            result=result,
        )
        return crew_run
