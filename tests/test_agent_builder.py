from conductor.builder.agent import build_from_section, ResearchAgentTemplate


def test_build_from_section() -> None:
    report_title = "Marketing"
    agent = build_from_section(report_title)
    assert isinstance(agent, ResearchAgentTemplate)
