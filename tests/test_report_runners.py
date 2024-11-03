"""
Tests for report runners
"""
from conductor.builder.agent import ResearchAgentTemplate
from conductor.reports.builder.runner import ResearchAgentSimulatedConversationRunner
from conductor.reports.builder.models import ResearchAgentConversations
from conductor.flow.retriever import ElasticRMClient
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
import os


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
