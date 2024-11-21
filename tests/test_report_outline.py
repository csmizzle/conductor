from conductor.reports.builder.outline import (
    OutlineBuilder,
    build_outline,
    build_refined_outline,
)
from conductor.reports.builder.writer import write_section, write_report
from conductor.reports.builder import models
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.reports.builder.runner import (
    run_team_simulated_conversations,
    summarize_team_conversations_parallel,
)
from conductor.reports.builder.models import ReportOutline, SourcedSection
from conductor.flow.rag import WebSearchRAG
from conductor.flow.retriever import ElasticRMClient
from conductor.rag.embeddings import BedrockEmbeddings
import os
from elasticsearch import Elasticsearch
import dspy
from tests.utils import save_model_to_test_data, load_model_from_test_data


def test_outline_builder() -> None:
    section_titles = [
        "Executive Summary",
        "Introduction",
        "Background",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    outline_builder = OutlineBuilder(section_titles=section_titles)
    specification = "The report should be about Thomson Reuters Special Services."
    outline = outline_builder(specification=specification)
    assert isinstance(outline, list)
    assert len(outline) == 7


def test_build_outline() -> None:
    claude = dspy.LM("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
    dspy.configure(lm=claude)
    section_titles = [
        "Executive Summary",
        "Introduction",
        "Background",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    specification = "The report should be about Thomson Reuters Special Services."
    outline = build_outline(specification=specification, section_titles=section_titles)
    assert isinstance(outline, list)
    assert len(outline) == 7


def test_build_refined_outline() -> None:
    agents = [
        ResearchAgentTemplate(
            title="Social Media Analyst",
            research_questions=[
                "What is the social media impact of Thomson Reuters?",
                "What are the social media values of Thomson Reuters?",
            ],
        ),
        ResearchAgentTemplate(
            title="Financial Analyst",
            research_questions=[
                "What strong financial points of Thomson Reuters?",
                "What are the financial values of Thomson Reuters?",
            ],
        ),
    ]
    team = ResearchTeamTemplate(
        title="Thomson Reuters Research Team",
        perspective="Looking for the areas of strategic growth in Thomson Reuters social media and finance.",
        agent_templates=agents,
    )
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    team_conversations = run_team_simulated_conversations(
        team=team, retriever=retriever, max_conversation_turns=3
    )
    assert isinstance(team_conversations, list)
    section_titles = [
        "Executive Summary",
        "Introduction",
        "Background",
        "Methodology",
        "Results",
        "Discussion",
        "Conclusion",
    ]
    specification = "The report should be about Thomson Reuters Special Services."
    outline = build_outline(specification=specification, section_titles=section_titles)
    assert isinstance(outline, list)
    assert len(outline) == 7
    # summarize conversations
    team_conversation_summaries = summarize_team_conversations_parallel(
        team_conversations=team_conversations
    )
    # build refined outline
    refined_outline = build_refined_outline(
        perspective=team.perspective,
        draft_outline=outline,
        conversation_summaries=team_conversation_summaries,
    )
    assert isinstance(refined_outline, dspy.Prediction)
    assert isinstance(refined_outline.refined_outline, ReportOutline)
    save_model_to_test_data(refined_outline.refined_outline, "refined_outline.json")


def test_write_section() -> None:
    lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
    )
    dspy.configure(lm=lm)
    outline = load_model_from_test_data(ReportOutline, "refined_outline.json")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    rag = WebSearchRAG.with_elasticsearch_id_retriever(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    section = write_section(
        section=outline.report_sections[0],
        rag=rag,
        specification="Thomson Reuters Special Services",
        perspective="Looking for strategic gaps in the company's operations and what they also do well.",
    )
    assert isinstance(section, SourcedSection)
    save_model_to_test_data(section, "section.json")


def test_write_report_claude() -> None:
    lm = dspy.LM(
        model="claude-3-5-sonnet",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        provider="bedrock",
        cache=False,
    )
    dspy.configure(lm=lm)
    outline = load_model_from_test_data(ReportOutline, "refined_outline.json")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    rag = WebSearchRAG.with_elasticsearch_id_retriever(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    report = write_report(
        outline=outline,
        rag=rag,
        specification="Thomson Reuters Special Services",
        perspective="Looking for strategic gaps in the company's operations and what they also do well.",
    )
    assert isinstance(report, models.Report)
    save_model_to_test_data(report, "report.json")


def test_write_report_gpt4o() -> None:
    lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
    )
    dspy.configure(lm=lm)
    outline = load_model_from_test_data(ReportOutline, "refined_outline.json")
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    rag = WebSearchRAG.with_elasticsearch_id_retriever(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
        cohere_api_key=os.getenv("COHERE_API_KEY"),
    )
    report = write_report(
        outline=outline,
        rag=rag,
        specification="Thomson Reuters Special Services",
        perspective="Looking for strategic gaps in the company's operations and what they also do well.",
    )
    assert isinstance(report, models.Report)
    save_model_to_test_data(report, "report_gpt4o.json")
