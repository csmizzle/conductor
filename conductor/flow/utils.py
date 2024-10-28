from elasticsearch import Elasticsearch
from crewai import LLM, Task, Agent, Crew
from conductor.crews.rag_marketing import tools


def build_organization_determination_crew(
    website_url: str, elasticsearch: Elasticsearch, index_name: str, llm: LLM
):
    organization_determination_agent = Agent(
        role="Organization Determination Agent",
        goal="Determine the organization from the website",
        backstory="The agent is tasked with determining the organization from the website.",
        tools=[
            tools.ScrapeWebsiteWithContentIngestTool(
                elasticsearch=elasticsearch,
                index_name=index_name,
            )
        ],
        allow_delegation=False,
        llm=llm,
    )
    organization_determination_task = Task(
        description=f"Determine the organization from the website {website_url}",
        agent=organization_determination_agent,
        expected_output="The determined organization from the website content with a sentence on reasoning.",
    )
    company_determination_crew = Crew(
        name="company_determination_crew",
        agents=[organization_determination_agent],
        tasks=[organization_determination_task],
    )
    return company_determination_crew
