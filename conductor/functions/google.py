"""
Gmail functions
"""
from conductor.chains import create_gmail_input
from conductor.parsers import gmail_input_parser
from langchain_community.agent_toolkits import GmailToolkit
from langchain_community.tools.gmail.utils import (
    get_gmail_credentials,
    build_resource_service,
)
from langchain_community.tools.gmail.create_draft import GmailCreateDraft
from langsmith import traceable


def create_gmail_draft(
    to: list[str], subject: str, message: str, credentials: str = None
) -> str:
    """Create a Gmail draft for the prospective customer.

    Args:
        to (str): where to send
        subject (str): subject of email
        message (str): message of email
        credentials (str): credentials for gmail API

    Returns:
        dict: context with data
    """
    if credentials:
        credentials = get_gmail_credentials(
            client_secrets_file=credentials,
        )
        api_resource = build_resource_service(credentials=credentials)
        gmail = GmailToolkit(api_resource=api_resource)
    else:
        gmail = GmailToolkit()
    draft = GmailCreateDraft(api_resource=gmail.api_resource)
    result = draft(
        {
            "to": to,
            "subject": subject,
            "message": message,
        }
    )
    return result


@traceable
def create_gmail_input_from_input(input_: str) -> dict:
    """Create a gmail draft from a natural language input.

    Args:
        input_ (str): general input
    """
    input = create_gmail_input(input_)
    parsed_inputs = gmail_input_parser.parse(input["text"])
    return parsed_inputs.dict()


@traceable
def create_gmail_draft_from_input(input_: str) -> str:
    """Create a Gmail draft from a natural language input."""
    parsed_inputs = create_gmail_input_from_input(input_)
    created_draft = create_gmail_draft(**parsed_inputs)
    print("Created draft:", created_draft)
    return created_draft
