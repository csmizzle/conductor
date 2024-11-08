from conductor.builder.agent import build_from_report_sections_parallel
from conductor.flow.flow import (
    RunResult,
    TeamFlow,
    run_research_and_search,
    run_flow,
)
from conductor.flow.retriever import ElasticRMClient
from conductor.flow.builders import build_research_team_from_template
from conductor.rag.embeddings import BedrockEmbeddings
from conductor.reports.builder.runner import (
    run_team_simulated_conversations,
    refine_team_from_conversations,
)
from conductor.reports.builder.outline import (
    build_outline,
    build_refined_outline,
)
from conductor.reports.builder.writer import write_report
from conductor.reports.builder import models
from tests.utils import save_model_to_test_data
from crewai import LLM
from elasticsearch import Elasticsearch
import os


def test_pipeline_v2(elasticsearch_test_agent_index) -> None:
    # search and answer research questions
    team_title = "Company Due Diligence"
    perspective = "Looking for strategic gaps in the company's operations and what they also do well."
    section_titles = [
        "Company Overview",
        "Financial Performance",
        "Key Customers",
        "Competitors",
        "Partnership Opportunities",
        "SWOT Analysis",
        "Strategy",
        "Strategy Recommendations",
    ]
    url = "https://trssllc.com"
    team = build_from_report_sections_parallel(
        team_title=team_title, section_titles=section_titles, perspective=perspective
    )
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_agent_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    run = run_research_and_search(
        research_llm=LLM("openai/gpt-4o-mini"),
        website_url=url,
        research_team=team,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_agent_index,
        embeddings=BedrockEmbeddings(),
    )
    assert isinstance(run, RunResult)
    # kickoff report generation
    team_conversation = run_team_simulated_conversations(team=team, retriever=retriever)
    assert isinstance(team_conversation, list)
    # refine team from conversations
    refined_team = refine_team_from_conversations(
        team=team, conversations=team_conversation
    )
    # collect additional based on feedback
    additional_research_team = build_research_team_from_template(
        team_template=refined_team,
        research_llm=LLM("openai/gpt-4o-mini"),
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_agent_index,
    )
    team_flow = TeamFlow(team=additional_research_team)
    results = run_flow(team_flow)
    assert isinstance(results, list)
    # write report
    outline = build_outline(
        specification=run.specification, section_titles=section_titles
    )
    # refine outline
    refined_outline = build_refined_outline(
        perspective=perspective,
        draft_outline=outline,
        conversations=team_conversation,
    )
    # write report
    report = write_report(outline=refined_outline, elastic_retriever=retriever)
    assert isinstance(report, models.Report)
    save_model_to_test_data(report, "test_full_report_v3.json")
