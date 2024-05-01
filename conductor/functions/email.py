"""
Email writing functions
"""
from conductor.chains import create_email_from_context
from langsmith import traceable


@traceable
def write_email_from_context(tone: str, context: str, sign_off: str) -> str:
    """
    Write an email from context and sign off
    """
    return create_email_from_context(tone=tone, context=context, sign_off=sign_off)[
        "text"
    ]
