"""
Test running the crew to answer a simple question
"""
from conductor.agents import run_crew
import agentops
from tests.vars import TEST_CREW_PROMPT


def test_run_crew() -> None:
    agentops.init(tags=["unit-test"])
    output = run_crew(TEST_CREW_PROMPT)
    assert isinstance(output, str) and len(output) > 0
