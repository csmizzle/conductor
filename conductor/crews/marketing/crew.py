from conductor.crews.marketing.agents import MarketingAgents
from conductor.crews.marketing.tasks import MarketingTasks
from conductor.llms import claude_sonnet
from crewai import Crew


class UrlMarketingCrew:
    def __init__(self, url: str) -> None:
        self.url = url

    def run(self) -> dict:
        agents = MarketingAgents()
        tasks = MarketingTasks()

        # agents
        company_research_agent = agents.company_research_agent(claude_sonnet)
        swot_agent = agents.swot_agent(claude_sonnet)
        competitor_agent = agents.competitor_agent(claude_sonnet)
        writer_agent = agents.writer_agent(claude_sonnet)
        # tasks
        company_research_task = tasks.company_research_task(
            agent=company_research_agent, company_url=self.url
        )
        swot_task = tasks.company_swot_task(
            agent=swot_agent, context=[company_research_task]
        )
        competitor_task = tasks.company_competitor_task(
            agent=competitor_agent, context=[company_research_task]
        )
        writer_task = tasks.company_report_task(
            agent=writer_agent,
            context=[company_research_task, swot_task, competitor_task],
        )
        # create crew
        crew = Crew(
            agents=[company_research_agent, swot_agent, competitor_agent, writer_agent],
            tasks=[company_research_task, swot_task, competitor_task, writer_task],
            verbose=True,
        )
        result = crew.kickoff()
        return result


if __name__ == "__main__":
    url = "https://www.trssllc.com"
    crew = UrlMarketingCrew(url)
    result = crew.run()
    print(result)
