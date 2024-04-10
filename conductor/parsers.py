"""
Langchain parsers
"""
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


class KeyPlayer(BaseModel):
    name: str = Field("The name of the key player")
    title: str = Field("The title of the key player")
    company: str = Field("The company of the key player")


class CompanyBackground(BaseModel):
    company: str = Field("The company name")
    background: str = Field("The background information of the company")


class EngagementStrategy(BaseModel):
    key_player: str = Field("The key player that the strategy is for")
    strategy: str = Field("The engagement strategy for the key player")


class KeyURL(BaseModel):
    company: str = Field("The company name")
    urls: list[str] = Field("The URLs for the company")


class CustomerObservation(BaseModel):
    key_players: list[KeyPlayer] = Field(
        "The key players identified in the search results"
    )
    company_backgrounds: list[CompanyBackground] = Field(
        "The background information of the companies tied to the key players"
    )
    engagement_strategy: list[EngagementStrategy] = Field(
        "The engagement strategy for each key player"
    )
    key_urls: list[KeyURL] = Field("The key URLs for each company and person")


customer_observation_parser = PydanticOutputParser(pydantic_object=CustomerObservation)
