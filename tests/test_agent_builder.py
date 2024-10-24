from conductor.builder.agent import (
    build_from_section,
    build_from_report_sections,
    build_from_report_sections_parallel,
    ResearchAgentTemplate,
    ResearchTeamTemplate,
)


def test_build_from_section() -> None:
    report_title = "Person Due Diligence"
    agent = build_from_section(
        team_title=report_title, section_title="Person Background"
    )
    assert isinstance(agent, ResearchAgentTemplate)


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
