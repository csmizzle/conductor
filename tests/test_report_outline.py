from conductor.reports.builder.outline import (
    OutlineBuilder,
    build_outline,
)
from conductor.reports.builder.writer import write_section, write_report
from conductor.reports.builder import models
from conductor.reports.builder.models import ReportOutline, SourcedSection
from conductor.flow.rag import WebSearchRAG, WebDocumentRetriever
from conductor.flow.retriever import ElasticDocumentIdRMClient
from conductor.rag.embeddings import BedrockEmbeddings
import os
from elasticsearch import Elasticsearch
import dspy
from tests.utils import save_model_to_test_data, load_model_from_test_data


def test_outline_builder() -> None:
    section_titles = [
        "Company Overview",
        "Company History",
        "Company Values",
    ]
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebDocumentRetriever(elastic_id_retriever=retriever)
    specification = "Thomson Reuters Special Services"
    outline_builder = OutlineBuilder(
        section_titles=section_titles,
        perspective="Looking for strategic gaps in the company's operations and what they also do well.",
        specification=specification,
        rag=rag,
    )
    outline = outline_builder.build()
    assert isinstance(outline, list)
    assert len(outline) == 3
    save_model_to_test_data(outline, "test_outline_builder.json")


def test_build_outline() -> None:
    lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=lm)
    section_titles = [
        "Company Overview",
        "Company History",
        "Company Values",
    ]
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    elasticsearch_test_index = os.getenv("ELASTICSEARCH_TEST_RAG_INDEX")
    retriever = ElasticDocumentIdRMClient(
        elasticsearch=elasticsearch,
        index_name=elasticsearch_test_index,
        embeddings=BedrockEmbeddings(),
    )
    rag = WebDocumentRetriever(elastic_id_retriever=retriever)
    specification = "Thomson Reuters Special Services"
    outline = build_outline(
        report_title="Thomson Reuters Special Services",
        section_titles=section_titles,
        perspective="Looking for strategic gaps in the company's operations and what they also do well.",
        specification=specification,
        rag=rag,
    )
    assert isinstance(outline, ReportOutline)
    assert len(outline.report_sections) == 3
    save_model_to_test_data(outline, "test_build_outline.json")


def test_write_section() -> None:
    lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=lm)
    outline = load_model_from_test_data(ReportOutline, "test_build_outline.json")
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


def test_write_report() -> None:
    lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=lm)
    outline = load_model_from_test_data(ReportOutline, "test_build_outline.json")
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


def test_write_report_gemini() -> None:
    lm = dspy.LM(
        "openai/gpt-4o",
        api_base=os.getenv("LITELLM_HOST"),
        api_key=os.getenv("LITELLM_API_KEY"),
        cache=False,
        max_tokens=3000,
    )
    dspy.configure(lm=lm)
    outline = load_model_from_test_data(ReportOutline, "test_build_outline.json")
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
