"""
Tests for report runners
"""
from conductor.builder.agent import ResearchAgentTemplate, ResearchTeamTemplate
from conductor.reports.builder.runner import (
    ResearchAgentSimulatedConversationRunner,
    ResearchTeamSimulatedConversationRunner,
    run_team_simulated_conversations,
    refine_team_from_conversations,
    summarize_conversation,
    summarize_conversations_parallel,
    summarize_team_conversations_parallel,
)
from conductor.reports.builder.models import ResearchAgentConversations
from conductor.flow.retriever import ElasticRMClient
from conductor.flow.flow import run_flow, TeamFlow
from conductor.flow.builders import build_research_team_from_template
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
from tests.utils import save_model_to_test_data, load_model_from_test_data
import os
from crewai import LLM


def test_research_agent_simulated_conversation_runner() -> None:
    """
    Test ResearchAgentSimulatedConversationRunner
    """
    research_agent_template = ResearchAgentTemplate(
        title="Social Media Analyst",
        research_questions=[
            "What is the social media impact of Thomson Reuters?",
            "What are the social media values of Thomson Reuters?",
        ],
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
    runner = ResearchAgentSimulatedConversationRunner(
        agent=research_agent_template, retriever=retriever, max_conversation_turns=5
    )
    conversations = runner.run_research_questions()
    assert isinstance(conversations, ResearchAgentConversations)


def test_research_team_simulated_conversations() -> None:
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
        perspective="Looking for strategic gaps in the company's operations and what they also do well.",
        title="Thomson Reuters Research Team",
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
    runner = ResearchTeamSimulatedConversationRunner(
        team=team, retriever=retriever, max_conversation_turns=3
    )
    team_conversations = runner.run_team_conversations()
    assert isinstance(team_conversations, list)
    save_model_to_test_data(team_conversations, "team_conversations.json")


def test_summarize_team_conversations() -> None:
    team_conversations = load_model_from_test_data(
        ResearchAgentConversations, "team_conversations.json"
    )
    summary = summarize_conversation(team_conversations[0].conversations[0])
    assert isinstance(summary, str)


def test_summarize_conversations_parallel() -> None:
    team_conversations = load_model_from_test_data(
        ResearchAgentConversations, "team_conversations.json"
    )
    summaries = summarize_conversations_parallel(team_conversations[0].conversations)
    assert isinstance(summaries, list)


def test_summarize_team_conversations_parallel() -> None:
    team_conversations = load_model_from_test_data(
        ResearchAgentConversations, "team_conversations.json"
    )
    summaries = summarize_team_conversations_parallel(team_conversations)
    assert isinstance(summaries, list)


def test_run_simulated_conversations() -> None:
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
        title="Thomson Reuters Research Team", agent_templates=agents
    )
    save_model_to_test_data(team, "test_research_team.json")
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


def test_refine_team_from_conversation() -> None:
    team = load_model_from_test_data(ResearchTeamTemplate, "test_research_team.json")
    team_conversations = load_model_from_test_data(
        ResearchAgentConversations, "team_conversations.json"
    )
    refined_team = refine_team_from_conversations(
        initial_team=team, conversations=team_conversations
    )
    assert isinstance(refined_team, ResearchTeamTemplate)
    save_model_to_test_data(refined_team, "refined_team.json")


def test_run_refined_team() -> None:
    team = load_model_from_test_data(ResearchTeamTemplate, "test_research_team.json")
    team_conversations = load_model_from_test_data(
        ResearchAgentConversations, "team_conversations.json"
    )
    refined_team = refine_team_from_conversations(
        initial_team=team, conversations=team_conversations
    )
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    research_team = build_research_team_from_template(
        team_template=refined_team,
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        research_llm=LLM("openai/gpt-4o"),
    )
    team_flow = TeamFlow(team=research_team)
    results = run_flow(team_flow)
    assert isinstance(results, list)
