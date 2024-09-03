from pydantic import BaseModel, Field
from typing import List


class SyntheticDocuments(BaseModel):
    documents: List[str] = Field(
        description="List of synthetic documents generated for the vector database"
    )
