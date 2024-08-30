from conductor.pipelines.research import ResearchPipeline
from conductor.reports.models import (
    RelationshipType,
    ReportTone,
    ReportStyleV2,
    ReportPointOfView,
    ReportV2,
)
from elasticsearch import Elasticsearch
from conductor.rag.client import ElasticsearchRetrieverClient
import os
from conductor.rag.embeddings import BedrockEmbeddings
import pytest


def test_research_pipeline_default_args(elasticsearch_test_agent_index) -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=elasticsearch_test_agent_index,
    )
    pipeline = ResearchPipeline(
        url="https://trssllc.com",
        report_style=ReportStyleV2.NARRATIVE,
        report_tone=ReportTone.ANALYTICAL,
        report_point_of_view=ReportPointOfView.THIRD_PERSON,
        client=client,
        report_title="TRSS Report",
        report_description="TRSS Report",
        image_search_relationships=[RelationshipType.EMPLOYEE],
        serpapi_key=os.getenv("SERPAPI_API_KEY"),
    )
    pipeline.run()
    assert isinstance(pipeline.report, ReportV2)
    # assert that all save operations dont work
    with pytest.raises(ValueError):
        pipeline.save_docx("test")
    with pytest.raises(ValueError):
        pipeline.save_pdf("test")
    with pytest.raises(ValueError):
        pipeline.save_graph("test")


def test_research_pipeline_enriched_pdf(elasticsearch_test_agent_index) -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=elasticsearch_test_agent_index,
    )
    pipeline = ResearchPipeline(
        url="https://parenthetic.io",
        report_style=ReportStyleV2.NARRATIVE,
        report_tone=ReportTone.ANALYTICAL,
        report_point_of_view=ReportPointOfView.THIRD_PERSON,
        client=client,
        report_title="Parenthetic Report",
        report_description="Parenthetic Report",
        image_search_relationships=[
            RelationshipType.EMPLOYEE.value,
            RelationshipType.EXECUTIVE.value,
        ],
        serpapi_key=os.getenv("SERPAPI_API_KEY"),
        pdf=True,
        enrich=True,
    )
    pipeline.run()
    assert isinstance(pipeline.report, ReportV2)
    # assert that enriched operations work
    assert pipeline.pdf_document
    pipeline.save_pdf("./parenthetic_pipeline_enriched.pdf")
