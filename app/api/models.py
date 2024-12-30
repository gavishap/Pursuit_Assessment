from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict
from datetime import datetime

class LinkBase(BaseModel):
    url: HttpUrl
    title: Optional[str] = None
    content_type: Optional[str] = None
    keywords: Optional[List[str]] = None
    link_metadata: Optional[Dict] = None

class LinkCreate(LinkBase):
    base_url: HttpUrl

class LinkResponse(LinkBase):
    id: int
    base_url: HttpUrl
    semantic_score: float
    openai_score: float
    nlp_score: float
    deep_learning_score: float
    ensemble_score: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ScrapeRequest(BaseModel):
    url: HttpUrl
    min_relevance: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Minimum ensemble score threshold")
    max_links: Optional[int] = Field(100, gt=0, le=1000, description="Maximum number of links to return")
    keywords: List[str] = Field(..., min_items=1, description="Keywords to use for ranking relevance. These define the context for scoring.")

class HealthCheck(BaseModel):
    status: str
    version: str 
