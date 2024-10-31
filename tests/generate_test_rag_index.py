"""
Create sample data for testing the ElasticsearchRetrieverClient
"""
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.crews.rag_marketing import tools
from conductor.flow.builders import build_team_from_template
from conductor.flow.research import (
    ResearchAgentFactory,
    ResearchQuestionAgentSearchTaskFactory,
)
from conductor.flow.team import ResearchTeamFactory
from conductor.flow.flow import run_flow, ResearchFlow
from crewai.llm import LLM
from elasticsearch import Elasticsearch
import os


def generate_data() -> None:
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    title = "Company Research Team"
    website_url = "https://www.trssllc.com"
    research_llm = LLM("openai/gpt-4o")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch.indices.create(
        index=elasticsearch_test_index, ignore=400
    )  # ignore 400 cause by index already exists

    agent_templates = [
        ResearchAgentTemplate(
            title="Company Structure Researcher",
            research_questions=[
                "Who leads their business divisions?",
                "What are the company divisions?",
            ],
        ),
    ]
    team_template = ResearchTeamTemplate(title=title, agent_templates=agent_templates)
    research_team = build_team_from_template(
        team_template=team_template,
        llm=research_llm,
        tools=[
            tools.SerpSearchEngineIngestTool(
                elasticsearch=elasticsearch, index_name=elasticsearch_test_index
            )
        ],
        agent_factory=ResearchAgentFactory,
        task_factory=ResearchQuestionAgentSearchTaskFactory,
        team_factory=ResearchTeamFactory,
    )
    research_flow = ResearchFlow(
        research_team=research_team,
        website_url=website_url,
        llm=research_llm,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
    )
    run_flow(flow=research_flow)


if __name__ == "__main__":
    generate_data()
