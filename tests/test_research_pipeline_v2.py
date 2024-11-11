from conductor.builder.agent import build_from_report_sections_parallel
from conductor.flow.flow import (
    RunResult,
    run_research_and_search,
)
from conductor.flow.retriever import ElasticRMClient
from conductor.rag.embeddings import BedrockEmbeddings
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
    # litellm_proxy_url = "http://0.0.0.0:4000"
    llm = dspy.LM(
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        # api_base=litellm_proxy_url
    )
    dspy.configure(lm=llm)
    mini = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        # base_url=litellm_proxy_url,
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
        parallel=True,
    )
    assert isinstance(run, RunResult)
    # kickoff report generation
    print("Building outline ...")
    outline = build_outline(
        specification=run.specification, section_titles=section_titles
    )
    # refine outline
    print("Refining outline ...")
    refined_outline = build_refined_outline(
        perspective=perspective,
        draft_outline=outline,
        # conversation_summaries=team_conversation_summaries,
    )
    # write report
    print("Writing report ...")
    report = write_report(
        outline=refined_outline.refined_outline, elastic_retriever=retriever
    )
    assert isinstance(report, models.Report)
    save_model_to_test_data(report, "test_full_report_v3.json")
