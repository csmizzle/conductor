from conductor.crews.marketing.agents import MarketingAgents
from conductor.reports.models import ReportStyle
from conductor.crews.marketing.tasks import MarketingTasks
from conductor.llms import claude_sonnet
from crewai import Crew
import logging


logger = logging.getLogger(__name__)


class UrlMarketingCrew:
    def __init__(self, url: str, report_style: ReportStyle) -> None:
        self.url = url
        self.report_style = report_style

    def run(self) -> dict:
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
        )
        result = crew.kickoff()
        return result
