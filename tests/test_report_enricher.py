from conductor.reports.enrich import ReportEnricher
from conductor.reports.models import RelationshipType, ReportV2
from conductor.rag.client import ElasticsearchRetrieverClient
from elasticsearch import Elasticsearch
from conductor.rag.embeddings import BedrockEmbeddings
from tests.constants import REPORT_V2_JSON
import os
import json


def test_report_enricher(elasticsearch_test_image_index) -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=elasticsearch_test_image_index,
    )
    report = ReportV2.model_validate(REPORT_V2_JSON)
    report_enricher = ReportEnricher(
        report=report,
        serp_api_key=os.getenv("SERPAPI_API_KEY"),
        image_relationship_types=[
            RelationshipType.FOUNDER.value,
            RelationshipType.EMPLOYEE.value,
        ],
        client=client,
        graph_sections=["Company Structure", "Personnel"],
        timeline_sections=["Company History", "Recent Events"],
        n_images=1,
    )
    enriched_report = report_enricher.enrich()
    assert isinstance(enriched_report, ReportV2)
    with open("tests/data/test_enriched_report.json", "w") as f:
        json.dump(enriched_report.model_dump(), f, indent=2)
