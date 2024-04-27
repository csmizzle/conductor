"""
Test running the crew to answer a simple question
"""
from conductor.agents import run_crew, gmail_crew
import agentops
from tests.vars import TEST_CREW_PROMPT, TEST_GMAIL_CREW_PROMPT
from langsmith import unit


@unit
def test_run_crew() -> None:
    agentops.init(tags=["run-unit-test"])
    output = run_crew(TEST_CREW_PROMPT)
    assert isinstance(output, str) and len(output) > 0


@unit
def test_gmail_crew() -> None:
    agentops.init(tags=["gmail-unit-test"])
    output = gmail_crew.kickoff({"context": TEST_GMAIL_CREW_PROMPT})
    assert isinstance(output, str) and len(output) > 0
