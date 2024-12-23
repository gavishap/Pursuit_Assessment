from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
from datetime import datetime

class LinkBase(BaseModel):
    url: HttpUrl
    title: Optional[str] = None
    content_type: Optional[str] = None
    keywords: Optional[List[str]] = None
    link_metadata: Optional[Dict] = None

class LinkCreate(LinkBase):
    pass

class Link(LinkBase):
    id: int
    relevance_score: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ScrapeRequest(BaseModel):
    url: HttpUrl
    max_links: Optional[int] = 100

class HealthCheck(BaseModel):
    status: str
    version: str 
