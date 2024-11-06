from conductor.reports.builder.outline import (
    OutlineBuilder,
    build_outline,
    build_refined_outline,
)
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.reports.builder.runner import (
    run_team_simulated_conversations,
)
from conductor.reports.builder.models import ReportOutline
from conductor.flow.retriever import ElasticRMClient
from conductor.rag.embeddings import BedrockEmbeddings
import os
from elasticsearch import Elasticsearch
import dspy
from tests.utils import save_model_to_test_data


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
    # build refined outline
    refined_outline = build_refined_outline(
        perspective=team.perspective,
        draft_outline=outline,
        conversations=team_conversations,
    )
    assert isinstance(refined_outline, dspy.Prediction)
    assert isinstance(refined_outline.refined_outline, ReportOutline)
    save_model_to_test_data(refined_outline.refined_outline, "refined_outline.json")
