"""
Langchain parsers
"""
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class ApolloInput(BaseModel):
    person_titles: list[str] = Field(
        "An array of the person's title. Apollo will return results matching ANY of the titles passed in"
    )
    person_locations: list[str] = Field(
        'An array of strings denoting allowed locations of the person. Be sure to include city and country separated by a comma. Example: "San Francisco, US" or "London, GB"'
    )


class EngagementStrategy(BaseModel):
    strategy: str = Field("The engagement strategy for the potential customer")
    reasoning: str = Field("The reasoning behind the strategy")


class PersonEngagementStrategy(BaseModel):
    person: dict
    engagement_strategy: EngagementStrategy
    context: str


class GmailInput(BaseModel):
    to: list[str] = Field("The email address to send the email to")
    subject: str = Field("The subject of the email")
    message: str = Field("The message of the email")


class HtmlSummary(BaseModel):
    content: str = Field("The content of the web page")
    summary: str = Field("The summary of the web page")


class Email(BaseModel):
    subject: str = Field("The subject of the email")
    email_body: str = Field("The body of the email")


engagement_strategy_parser = PydanticOutputParser(pydantic_object=EngagementStrategy)
gmail_input_parser = PydanticOutputParser(pydantic_object=GmailInput)
html_summary_parser = PydanticOutputParser(pydantic_object=HtmlSummary)
email_parser = PydanticOutputParser(pydantic_object=Email)
apollo_input_parser = PydanticOutputParser(pydantic_object=ApolloInput)
