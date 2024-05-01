"""
Test email functions
"""
from langsmith import unit
from conductor.functions.email import write_email_from_context
from conductor.parsers import Email
from tests.vars import TEST_EMAIL_CONTEXT


@unit
def test_write_email_from_context() -> None:
    email = write_email_from_context(
        context=TEST_EMAIL_CONTEXT, sign_off="Best, John Envoy"
    )
    assert isinstance(email, Email)
    assert email.sign_off == "Best, John Envoy"
