"""
Pydantic models for the profiles module.
"""
from pydantic import BaseModel, Field
from conductor.flow.rag import CitedValueWithCredibility


class SpecifiedField(BaseModel):
    """
    Field with specification query
    """

    field: str = Field(description="Field value")
    query: str = Field(description="Specified query")


class Company(BaseModel):
    """
    Company profile
    """

    name: CitedValueWithCredibility = Field(description="Official company name")
    address: CitedValueWithCredibility = Field(description="Official company address")
    size: CitedValueWithCredibility = Field(description="Company size estimate")
    industry: CitedValueWithCredibility = Field(description="Company industry sector")
    website: CitedValueWithCredibility = Field(description="Company website URL")
    naics_code: CitedValueWithCredibility = Field(
        description="Inferred Company NAICS code"
    )
    ceo: CitedValueWithCredibility = Field(description="Company CEO")
    president: CitedValueWithCredibility = Field(description="Company president")
    cto: CitedValueWithCredibility = Field(description="Company CTO")
    cfo: CitedValueWithCredibility = Field(description="Company CFO")
    coo: CitedValueWithCredibility = Field(description="Company COO")
    key_product: CitedValueWithCredibility = Field(description="Company key product")
    slogan: CitedValueWithCredibility = Field(description="Company slogan")
    foreign_connections: CitedValueWithCredibility = Field(
        description="A field denoting if company has foreign connections"
    )
    us_govt_contracts: CitedValueWithCredibility = Field(
        description="A field denoting if company has US government contracts"
    )
