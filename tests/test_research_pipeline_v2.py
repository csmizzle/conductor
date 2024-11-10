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
    summarize_team_conversations_parallel,
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
import dspy
import os


def test_pipeline_v2(elasticsearch_cloud_test_research_index) -> None:
    # search and answer research questions
    # set openai to litellm proxy
    litellm_proxy_url = "http://0.0.0.0:4000"
    llm = dspy.LM("gpt-4o", api_base=litellm_proxy_url)
    dspy.configure(lm=llm)
    mini = LLM(
        model="gpt-4o-mini",
        base_url=litellm_proxy_url,
    )
    team_title = "Company Due Diligence"
    perspective = "Looking for strategic gaps in the company's operations and what they also do well."
    section_titles = [
        "Company Overview",
        # "Financial Performance",
        "Key Customers",
        "Competitors",
        # "Partnership Opportunities",
        # "SWOT Analysis",
        # "Strategy",
        # "Strategy Recommendations",
    ]
    url = "https://trssllc.com"
    team = build_from_report_sections_parallel(
        team_title=team_title, section_titles=section_titles, perspective=perspective
    )
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
        api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
    )
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    run = run_research_and_search(
        research_llm=mini,
        website_url=url,
        research_team=team,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
    )
    assert isinstance(run, RunResult)
    # kickoff report generation
    print("Having simulated conversations ...")
    team_conversation = run_team_simulated_conversations(team=team, retriever=retriever)
    assert isinstance(team_conversation, list)
    # refine team from conversations
    refined_team = refine_team_from_conversations(
        initial_team=team, conversations=team_conversation
    )
    # collect additional based on feedback
    additional_research_team = build_research_team_from_template(
        team_template=refined_team,
        research_llm=mini,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
    )
    print("Running additional research team ...")
    team_flow = TeamFlow(team=additional_research_team)
    results = run_flow(team_flow)
    assert isinstance(results, list)
    # write report
    print("Building outline ...")
    outline = build_outline(
        specification=run.specification, section_titles=section_titles
    )
    # refine outline
    print("Summarizing conversations ...")
    team_conversation_summaries = summarize_team_conversations_parallel(
        team_conversations=team_conversation
    )
    print("Refining outline ...")
    refined_outline = build_refined_outline(
        perspective=perspective,
        draft_outline=outline,
        conversation_summaries=team_conversation_summaries,
    )
    # write report
    print("Writing report ...")
    report = write_report(outline=refined_outline, elastic_retriever=retriever)
    assert isinstance(report, models.Report)
    save_model_to_test_data(report, "test_full_report_v3.json")
