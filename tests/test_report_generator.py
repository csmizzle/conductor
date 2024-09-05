from conductor.reports.pydantic_templates.generate import (
    BasicSourcedHyDEReportTemplateGenerator,
)
from conductor.reports.models import (
    ReportStyleV2,
    ReportTone,
    ReportPointOfView,
    ReportV2,
)
from conductor.chains import run_sourced_hyde_report_generation
import json
import os
from conductor.rag.client import ElasticsearchRetrieverClient
from conductor.rag.embeddings import BedrockEmbeddings
from elasticsearch import Elasticsearch


def test_generate_basic_sourced_template() -> None:
    title = "Test Report"
    style = ReportStyleV2.BULLETED
    tone = ReportTone.INFORMAL
    point_of_view = ReportPointOfView.FIRST_PERSON
    hyde_context = "The context"
    section_titles = ["Section 1", "Section 2"]
    hyde_section_objectives = ["Objective 1", "Objective 2"]
    generator = BasicSourcedHyDEReportTemplateGenerator(
        title=title,
        style=style,
        tone=tone,
        point_of_view=point_of_view,
        hyde_context=hyde_context,
        section_titles=section_titles,
        hyde_section_objectives=hyde_section_objectives,
    )
    template = generator.generate()
    assert template.title == title
    assert template.section_templates[0].title == section_titles[0]
    assert template.section_templates[1].title == section_titles[1]
    assert template.section_templates[0].style == style
    assert template.section_templates[1].style == style
    assert template.section_templates[0].tone == tone
    assert template.section_templates[1].tone == tone
    assert template.section_templates[0].point_of_view == point_of_view
    assert template.section_templates[1].point_of_view == point_of_view
    assert template.section_templates[0].hyde_context == hyde_context
    assert template.section_templates[1].hyde_context == hyde_context
    assert template.section_templates[0].hyde_objective == hyde_section_objectives[0]
    assert template.section_templates[1].hyde_objective == hyde_section_objectives[1]


def test_run_sourced_hyde_report_generation() -> None:
    elasticsearch = Elasticsearch(
        hosts=[os.getenv("ELASTICSEARCH_URL")],
    )
    embeddings = BedrockEmbeddings()
    client = ElasticsearchRetrieverClient(
        elasticsearch=elasticsearch,
        embeddings=embeddings,
        index_name=os.getenv("ELASTICSEARCH_INDEX"),
    )
    report = run_sourced_hyde_report_generation(
        title="TRSS Report",
        description="A report on TRSS",
        style=ReportStyleV2.NARRATIVE,
        tone=ReportTone.ANALYTICAL.name,
        point_of_view=ReportPointOfView.THIRD_PERSON.name,
        hyde_context="Thomson Reuters Special Services",
        section_titles=["Company Leadership", "Product & Services"],
        hyde_section_objectives=[
            "Build out a report section on the company's leadership team",
            "Build out a report section on the company's product and services",
        ],
        retriever=client,
    )
    assert isinstance(report, ReportV2)
    with open("tests/data/sourced_hyde_report.json", "w") as f:
        json.dump(report.model_dump(), f, indent=4)
