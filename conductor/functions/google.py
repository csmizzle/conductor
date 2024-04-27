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
from langchain_community.tools.gmail.send_message import GmailSendMessage
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


def send_gmail(
    to: list[str],
    subject: str,
    message: str,
    cc: list[str] = None,
    bcc: list[str] = None,
    credentials: str = None,
) -> str:
    """Send a Gmail message to the prospective customer.

    Args:
        to (str): where to send
        subject (str): subject of email
        message (str): message of email
        cc (str): carbon copy
        bcc (str): blind carbon copy
        credentials (str): credentials for gmail API

    Returns:
        str: result of sending email
    """
    if credentials:
        credentials = get_gmail_credentials(
            client_secrets_file=credentials,
        )
        api_resource = build_resource_service(credentials=credentials)
        gmail = GmailToolkit(api_resource=api_resource)
    else:
        gmail = GmailToolkit()
    send_message = GmailSendMessage(api_resource=gmail.api_resource)
    result = send_message(
        {
            "to": to,
            "subject": subject,
            "message": message,
            "cc": cc,
            "bcc": bcc,
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
    return created_draft


@traceable
def send_gmail_from_input(input_: str) -> str:
    """Send a Gmail message from a natural language input."""
    parsed_inputs = create_gmail_input_from_input(input_)
    sent_email = send_gmail(**parsed_inputs)
    return sent_email
