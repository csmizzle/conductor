from conductor.crews.rag_marketing.chains import (
    task_run_to_report_section,
    crew_run_to_report,
    extract_graph_from_report,
    extract_timeline_from_report,
    Graph,
    Timeline,
    ReportV2,
)
from conductor.chains import relationships_to_image_query
from conductor.reports.models import RelationshipType
from conductor.crews.models import CrewRun, TaskRun
from conductor.reports.models import (
    ReportStyleV2,
    ReportTone,
    ReportPointOfView,
    SectionV2,
)
from tests.constants import REPORT_V2_JSON, GRAPH_JSON, BASEDIR
import os
import vcr

example_data = """
Thomson Reuters Special Services (TRSS) operates in multiple markets and industries, primarily serving government agencies, law enforcement, defense/intelligence communities, and commercial businesses.

Their main markets and customers include:

1. Government - TRSS provides data analysis, intelligence, risk mitigation and specialized services to federal law enforcement agencies, the U.S. Department of Defense, and intelligence communities. Key offerings include due diligence, insider threat detection, network analysis, and supporting national security missions.

2. Law Enforcement - TRSS offers investigative tools like CLEAR for law enforcement to enable searches, subject identification, asset tracking and crime prevention/detection. Their solutions aid public safety efforts.

3. Commercial Businesses - They deliver tailored data, risk management and analytic services to support critical decision making for commercial entities across industries like finance, legal, supply chain, and more.

4. International Markets - TRSS has international commercial customers they serve with data enablement and risk mitigation solutions for operating across borders.

5. Human Rights - TRSS focuses on combating human rights crimes like human trafficking through intelligence gathering and empowering investigations into exploitative criminal networks.

In summary, while a subsidiary of Thomson Reuters, TRSS specializes in serving the data analysis, risk management and intelligent service needs for government and commercial clients engaged in fields like law enforcement, national security, finance, legal and human rights. Their technical capabilities and data access allow them to operate across this diverse market landscape.

Source Links:
https://trssllc.com/business/
https://www.crunchbase.com/organization/thomson-reuters-special-services
https://www.corporategray.com/employers/32278/public_profile
https://topworkplaces.com/company/thomson-reuters-special/
https://uk.linkedin.com/company/trss
"""

example_data_2 = """
Thought: Based on the information gathered through search engine queries, here are my key findings about TRSS's market and estimated market size:

TRSS operates in several markets related to data analytics, intelligence, and risk mitigation services. Their main customer segments appear to be:

1. Government agencies (law enforcement, defense, intelligence)
2. Commercial businesses
3. International markets

Some of the specific market areas TRSS is involved in include:

- Data analytics and data enablement solutions
- Network analysis and threat intelligence
- Risk analytics and due diligence solutions
- Investigations into human rights crimes
- Supply chain risk analysis

While exact market size numbers are difficult to find, some potentially relevant global market estimates are:

- Risk analytics market size was valued at $29.9 billion in 2020 and projected to grow to $89.2 billion by 2028 (Source: MarketsandMarkets)
- Threat intelligence market size was $8.8 billion in 2020 and forecasted to reach $20.8 billion by 2026 (Source: Mordor Intelligence)

However, these are very broad markets. TRSS likely occupies specialized niches within these markets focused on serving government/defense clients and providing customized data solutions.
"""


def test_task_run_to_report_section() -> None:
    # lengthy paragraph on background of microsoft for test
    test_task_run = TaskRun(
        name="Test Task",
        agent_role="Test Agent",
        description="Test Description",
        result=example_data,
        section_name="TRSS Background",
    )
    section = task_run_to_report_section(
        task_run=test_task_run,
        style=ReportStyleV2.NARRATIVE,
        tone=ReportTone.PROFESSIONAL,
        point_of_view=ReportPointOfView.FIRST_PERSON,
    )
    assert isinstance(section, SectionV2)


def test_crew_run_to_report() -> None:
    test_task_run = TaskRun(
        name="Test Task",
        agent_role="Test Agent",
        description="Test Description",
        result=example_data,
    )
    test_task_run_2 = TaskRun(
        name="Test Task 2",
        agent_role="Test Agent",
        description="Test Description",
        result=example_data_2,
    )
    test_crew_run = CrewRun(
        tasks=[test_task_run, test_task_run_2], result="Test Result"
    )
    report = crew_run_to_report(
        crew_run=test_crew_run,
        title="TRSS Report",
        description="Test Description",
        style=ReportStyleV2.NARRATIVE,
        tone=ReportTone.PROFESSIONAL,
        point_of_view=ReportPointOfView.FIRST_PERSON,
    )
    assert isinstance(report, ReportV2)


def test_extract_graph_from_report() -> None:
    report = ReportV2.parse_obj(REPORT_V2_JSON)
    graph = extract_graph_from_report(
        report, sections_filter=["Company Structure", "Personnel"]
    )
    assert isinstance(graph, Graph)


def test_extract_timeline_from_report() -> None:
    report = ReportV2.parse_obj(REPORT_V2_JSON)
    timeline = extract_timeline_from_report(
        report, sections_filter=["Company History", "Recent Events"]
    )
    assert isinstance(timeline, Timeline)


# adding vcr to save credits
@vcr.use_cassette(
    os.path.join(BASEDIR, "cassettes", "test_relationship_to_image_search.yaml")
)
def test_relationship_to_image_search() -> None:
    graph = Graph.model_validate(GRAPH_JSON)
    images = relationships_to_image_query(
        graph=graph,
        api_key=os.getenv("SERPAPI_API_KEY"),
        relationship_types=[
            RelationshipType.EMPLOYEE.value,
            RelationshipType.FOUNDER.value,
            RelationshipType.EXECUTIVE.value,
        ],
    )
    assert isinstance(images, list)
