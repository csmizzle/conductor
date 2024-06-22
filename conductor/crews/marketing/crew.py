from conductor.crews.marketing.agents import MarketingAgents
from conductor.reports.models import ReportStyle
from conductor.crews.marketing.tasks import MarketingTasks
from conductor.crews.models import CrewRun
from conductor.llms import claude_sonnet
from crewai import Crew
import logging


logger = logging.getLogger(__name__)


class UrlMarketingCrew:
    """
    Start with a company URL and generate a marketing report
    """

    def __init__(
        self,
        url: str,
        report_style: ReportStyle,
        output_log_file: bool | str = None,
        step_callback=None,
        task_callback=None,
    ) -> None:
        self.url = url
        self.report_style = report_style
        self.output_log_file = output_log_file
        self.step_callback = step_callback
        self.task_callback = task_callback

    def run(self) -> CrewRun:
        # create agents and tasks
        agents = MarketingAgents()
        tasks = MarketingTasks()

        # agents
        company_research_agent = agents.company_research_agent(claude_sonnet)
        search_engine_agent = agents.search_engine_agent(claude_sonnet)
        swot_agent = agents.swot_agent(claude_sonnet)
        competitor_agent = agents.competitor_agent(claude_sonnet)
        writer_agent = agents.writer_agent(claude_sonnet)

        # tasks
        company_research_task = tasks.company_research_task(
            agent=company_research_agent, company_url=self.url
        )
        search_engine_task = tasks.search_engine_task(
            agent=search_engine_agent,
            context=[company_research_task],
        )
        swot_task = tasks.company_swot_task(
            agent=swot_agent, context=[company_research_task, search_engine_task]
        )
        competitor_task = tasks.company_competitor_task(
            agent=competitor_agent, context=[company_research_task, search_engine_task]
        )
        writer_task = tasks.company_report_task(
            agent=writer_agent,
            context=[
                company_research_task,
                swot_task,
                competitor_task,
                search_engine_task,
            ],
            report_style=self.report_style,
        )

        # create crew
        crew = Crew(
            agents=[
                company_research_agent,
                search_engine_agent,
                swot_agent,
                competitor_agent,
                writer_agent,
            ],
            tasks=[
                company_research_task,
                search_engine_task,
                swot_task,
                competitor_task,
                writer_task,
            ],
            verbose=True,
            step_callback=self.step_callback,
            output_log_file=self.output_log_file,
            task_callback=self.task_callback,
        )
        result = crew.kickoff()

        # create and return crew run
        crew_run = CrewRun(
            task_outputs=[
                company_research_task.output,
                search_engine_task.output,
                swot_task.output,
                competitor_task.output,
                writer_task.output,
            ],
            result=result,
        )
        return crew_run
