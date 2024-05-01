"""
Email writing functions
"""
from conductor.chains import create_email_from_context
from conductor.parsers import email_parser
from langsmith import traceable


@traceable
def write_email_from_context(context: str, sign_off: str) -> str:
    """
    Write an email from context and sign off
    """
    email = create_email_from_context(context=context, sign_off=sign_off)
    email_structured = email_parser.parse(text=email["text"])
    return email_structured
