from pydantic import BaseModel
from datetime import datetime


class WebPage(BaseModel):
    url: str
    created_at: datetime
    content: str
    raw: str
