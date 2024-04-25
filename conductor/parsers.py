"""
Langchain parsers
"""
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class CompanyBackground(BaseModel):
    name: str = Field("The company name")
    background: str = Field("The background information of the company")


class KeyPlayer(BaseModel):
    name: str = Field("The name of the key player")
    title: str = Field("The title of the key player")
    city: str = Field("The city of the key player")
    state: str = Field("The state of the key player")
    country: str = Field("The country of the key player")
    company: CompanyBackground = None
    strategy: str = Field("The engagement strategy for the key player")
    urls: list[str] = Field("The URLs for the company")


class EngagementStrategy(BaseModel):
    strategy: str = Field("The engagement strategy for the potential customer")
    reasoning: str = Field("The reasoning behind the strategy")


class CustomerObservation(BaseModel):
    key_players: list[KeyPlayer] = Field(
        "The key players identified in the search results"
    )


class PersonEngagementStrategy(BaseModel):
    person: dict
    engagement_strategy: EngagementStrategy
    context: str


customer_observation_parser = PydanticOutputParser(pydantic_object=CustomerObservation)


engagement_strategy_parser = PydanticOutputParser(pydantic_object=EngagementStrategy)
