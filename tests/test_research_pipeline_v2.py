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
from conductor.pipelines.research import ResearchPipelineV2
from conductor.profiles.models import Company
from tests.utils import save_model_to_test_data
from crewai import LLM
from elasticsearch import Elasticsearch
import dspy
import os
from langtrace_python_sdk import langtrace
from langtrace_python_sdk.utils.with_root_span import with_langtrace_root_span
import pytest


langtrace.init()

# url
url = "https://altana.ai/"
# team title
team_title = "Company Due Diligence"
# perspective
perspective = (
    "Looking for strategic gaps in the company's operations and what they also do well."
)
# section titles
section_titles = [
    "Company Overview",
    "Products and Services",
    "Customers Profiles",
    "Competitors",
]
# elasticsearch
elasticsearch = Elasticsearch(
    hosts=[os.getenv("ELASTICSEARCH_CLOUD_URL")],
    api_key=os.getenv("ELASTICSEARCH_CLOUD_API_ADMIN_KEY"),
)
# llms
bedrock_claude_sonnet = dspy.LM(
    model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    max_tokens=3000,
)
gpt_4o_mini_dspy = dspy.LM(
    "gpt-4o-mini",
)
# gpt_4o = dspy.LM(
#     "gpt-4o",
#     max_tokens=3000,
# )
gpt_4o_mini = LLM(
    model="gpt-4o-mini",
)


@with_langtrace_root_span(name="test_pipeline_v2")
def test_pipeline_v2(elasticsearch_cloud_test_research_index) -> None:
    # search and answer research questions
    # set openai to litellm proxy
    # litellm_proxy_url = "http://0.0.0.0:4000"
    llm = dspy.LM(
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=3000,
        # api_base=litellm_proxy_url
    )
    dspy.configure(lm=llm)
    mini = LLM(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=3000,
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


@with_langtrace_root_span(name="test_pipeline_v2_build_teams")
def test_pipeline_v2_build_teams(elasticsearch_cloud_test_research_index) -> None:
    # llms
    bedrock_claude_sonnet = dspy.LM(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        max_tokens=3000,
    )
    gpt_4o_mini_dspy = dspy.LM(
        "gpt-4o-mini",
    )
    # gpt_4o = dspy.LM(
    #     "gpt-4o",
    #     max_tokens=3000,
    # )
    gpt_4o_mini = LLM(
        model="gpt-4o-mini",
    )
    research_retriver = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    graph_retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        k=10,
        rerank_top_n=5,
    )
    # run pipeline
    pipeline = ResearchPipelineV2(
        url=url,
        team_title=team_title,
        perspective=perspective,
        section_titles=section_titles,
        elasticsearch=elasticsearch,
        elasticsearch_index=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        run_in_parallel=True,
        team_builder_llm=gpt_4o_mini_dspy,
        research_llm=gpt_4o_mini,
        search_llm=bedrock_claude_sonnet,
        outline_llm=bedrock_claude_sonnet,
        report_llm=bedrock_claude_sonnet,
        research_retriever=research_retriver,
        graph_retriever=graph_retriever,
    )
    pipeline.build_teams()
    assert pipeline.team is not None
    assert pipeline.research_team is not None
    assert pipeline.search_team is not None


def test_pipeline_class_v2_research(elasticsearch_cloud_test_research_index) -> None:
    # run pipeline
    research_retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    pipeline = ResearchPipelineV2(
        url=url,
        team_title=team_title,
        perspective=perspective,
        section_titles=section_titles,
        elasticsearch=elasticsearch,
        elasticsearch_index=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        research_retriever=research_retriever,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        team_builder_llm=gpt_4o_mini_dspy,
        research_llm=gpt_4o_mini,
        search_llm=bedrock_claude_sonnet,
        outline_llm=bedrock_claude_sonnet,
        report_llm=bedrock_claude_sonnet,
    )
    # assert this raises a value error because the teams have not been built
    with pytest.raises(ValueError):
        pipeline.run_research()
    # build teams
    pipeline.build_teams()
    # run research
    run_result = pipeline.run_research()
    assert isinstance(run_result, list)
    assert len(run_result) > 0


def test_pipeline_class_v2_research_parallel(
    elasticsearch_cloud_test_research_index,
) -> None:
    # run pipeline
    research_retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    pipeline = ResearchPipelineV2(
        url=url,
        team_title=team_title,
        perspective=perspective,
        section_titles=section_titles,
        elasticsearch=elasticsearch,
        elasticsearch_index=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        run_in_parallel=True,
        research_retriever=research_retriever,
        team_builder_llm=gpt_4o_mini_dspy,
        research_llm=gpt_4o_mini,
        search_llm=bedrock_claude_sonnet,
        outline_llm=bedrock_claude_sonnet,
        report_llm=bedrock_claude_sonnet,
    )
    # assert this raises a value error because the teams have not been built
    with pytest.raises(ValueError):
        pipeline.run_research()
    # build teams
    pipeline.build_teams()
    # run research
    run_result = pipeline.run_research()
    assert isinstance(run_result, list)
    assert len(run_result) > 0


def test_pipeline_class_v2_research_and_search(
    elasticsearch_cloud_test_research_index,
) -> None:
    """
    Run the entire pipeline with search
    """
    pipeline = ResearchPipelineV2(
        url=url,
        team_title=team_title,
        perspective=perspective,
        section_titles=section_titles,
        elasticsearch=elasticsearch,
        elasticsearch_index=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        run_in_parallel=True,
        team_builder_llm=gpt_4o_mini_dspy,
        research_llm=gpt_4o_mini,
        search_llm=bedrock_claude_sonnet,
        outline_llm=bedrock_claude_sonnet,
        report_llm=bedrock_claude_sonnet,
    )
    # create search assets and run research and search
    pipeline.build_team_template()
    assert pipeline.team is not None
    pipeline.build_research_team()
    assert pipeline.research_team is not None
    pipeline.run_research()
    assert pipeline.research_results is not None
    pipeline.build_search_team()
    assert pipeline.search_team is not None
    pipeline.run_search()
    assert pipeline.search_results is not None


def test_research_pipeline_max_iter_small(
    elasticsearch_cloud_test_research_index: str,
) -> None:
    pipeline = ResearchPipelineV2(
        url=url,
        team_title=team_title,
        perspective=perspective,
        section_titles=section_titles,
        elasticsearch=elasticsearch,
        elasticsearch_index=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        run_in_parallel=True,
        team_builder_llm=gpt_4o_mini_dspy,
        research_llm=gpt_4o_mini,
        search_llm=bedrock_claude_sonnet,
        outline_llm=bedrock_claude_sonnet,
        report_llm=bedrock_claude_sonnet,
        research_max_iterations=1,
    )
    # create search assets and run research and search
    pipeline.build_team_template()
    assert pipeline.team is not None
    pipeline.build_research_team()
    assert pipeline.research_team is not None
    assert pipeline.research_team.agents[0].max_iter == 1
    pipeline.run_research()


def test_research_and_search_pipeline_max_iter_small(
    elasticsearch_cloud_test_research_index: str,
) -> None:
    pipeline = ResearchPipelineV2(
        url=url,
        team_title=team_title,
        perspective=perspective,
        section_titles=section_titles,
        elasticsearch=elasticsearch,
        elasticsearch_index=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        run_in_parallel=True,
        team_builder_llm=gpt_4o_mini_dspy,
        research_llm=gpt_4o_mini,
        search_llm=bedrock_claude_sonnet,
        outline_llm=bedrock_claude_sonnet,
        report_llm=bedrock_claude_sonnet,
        research_max_iterations=1,
    )
    # create search assets and run research and search
    pipeline.build_team_template()
    assert pipeline.team is not None
    pipeline.build_research_team()
    assert pipeline.research_team is not None
    assert pipeline.research_team.agents[0].max_iter == 1
    pipeline.run_research()
    assert pipeline.research_results is not None
    pipeline.build_search_team()
    assert pipeline.search_team is not None
    # run search team
    pipeline.run_search()
    assert pipeline.search_answers is not None


def test_research_and_search_pipeline_profile(
    elasticsearch_cloud_test_research_index: str,
) -> None:
    pipeline = ResearchPipelineV2(
        url=url,
        team_title=team_title,
        perspective=perspective,
        section_titles=section_titles,
        elasticsearch=elasticsearch,
        elasticsearch_index=elasticsearch_cloud_test_research_index,
        embeddings=BedrockEmbeddings(),
        profile=Company,
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        run_in_parallel=True,
        team_builder_llm=gpt_4o_mini_dspy,
        research_llm=gpt_4o_mini,
        search_llm=bedrock_claude_sonnet,
        outline_llm=bedrock_claude_sonnet,
        report_llm=bedrock_claude_sonnet,
        profile_llm=bedrock_claude_sonnet,
        research_max_iterations=1,
    )
    # create search assets and run research and search
    pipeline.build_team_template()
    assert pipeline.team is not None
    pipeline.build_research_team()
    assert pipeline.research_team is not None
    assert pipeline.research_team.agents[0].max_iter == 1
    pipeline.run_research()
    assert pipeline.research_results is not None
    pipeline.build_search_team()
    assert pipeline.search_team is not None
    # run search team
    pipeline.run_search()
    assert pipeline.search_answers is not None
    # create company profile
    pipeline.build_profile()
    assert pipeline.profile is not None
