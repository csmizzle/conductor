import dspy
from conductor.reports.builder.conversations import SimulatedConversation
from conductor.flow.retriever import ElasticRMClient
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
import os


def test_simulated_conversation() -> None:
    research_question = "What is the social media impact of Thomson Reuters?"
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
    simulated_conversation = SimulatedConversation(
        max_conversation_turns=5, retriever=retriever
    )
    question = simulated_conversation(research_question)
    assert isinstance(question, dspy.Prediction)
