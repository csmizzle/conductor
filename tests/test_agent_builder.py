from conductor.builder.agent import (
    ResearchAgentTemplate,
    ResearchTeamTemplate,
    build_from_section,
    build_from_report_sections,
    build_from_report_sections_parallel,
)
from tests.utils import save_model_to_test_data


def test_build_from_section() -> None:
    team_title = "Person Due Diligence"
    perspective = "Focus on the person's background and any foreign connections"
    agent = build_from_section(
        team_title=team_title,
        section_title="Person Background",
        perspective=perspective,
    )
    assert isinstance(agent, ResearchAgentTemplate)
    save_model_to_test_data(agent, "research_agent_template.json")


def test_build_from_report_sections() -> None:
    report_title = "Person Due Diligence"
    section_titles = [
        "Person Background",
        "Person Employment History",
        "Person Financial History",
    ]
    team = build_from_report_sections(
        report_title=report_title, section_titles=section_titles
    )
    assert isinstance(team, ResearchTeamTemplate)
    for agent in team.agents:
        assert isinstance(agent, ResearchAgentTemplate)


def test_build_from_report_sections_parallel() -> None:
    report_title = "Person Due Diligence"
    section_titles = [
        "Person Background",
        "Person Employment History",
        "Person Financial History",
    ]
    team = build_from_report_sections_parallel(
        report_title=report_title, section_titles=section_titles
    )
    assert isinstance(team, ResearchTeamTemplate)
    for agent in team.agents:
        assert isinstance(agent, ResearchAgentTemplate)
